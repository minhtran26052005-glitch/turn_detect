"""Resumable CPU-only input caches, independent of the model/GPU lock."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from . import translation_segmenter_full_data_v5 as data
from .translation_segmenter_v15c import LABELS,atomic_json


SPECS={'ids':(np.dtype('int32'),256),'lengths':(np.dtype('int64'),None),
       'labels':(np.dtype('int64'),None),'horizons':(np.dtype('int64'),None)}


def folder_for(manifest,language,split):
    return data.JOB/'datasets'/manifest['generation']/'encoded'/language/split


def arrays_at(folder,n,create=False):
    result={}
    for key,(dtype,width) in SPECS.items():
        path=folder/(key+'.npy');shape=(n,width) if width else (n,)
        if path.exists():
            array=np.load(path,mmap_mode='r+')
            assert array.shape==shape and array.dtype==dtype,(key,array.shape,array.dtype)
        else:
            assert create,'Missing partial cache array: '+key
            array=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
        result[key]=array
    return result


def ready(manifest,language,split='train'):
    folder=folder_for(manifest,language,split);p=folder/'complete.json'
    if not p.exists():return False
    proof=json.loads(p.read_text());assert proof['fingerprint']==manifest['fingerprints'][language][split]
    return True


def progress_state(manifest,language,split,stage,rows,total,**extra):
    atomic_json(data.JOB/'cache_status'/(language+'_'+split+'.json'),{
      'stage':stage,'at_utc':data.now(),'pid':os.getpid(),'generation':manifest['generation'],
      'language':language,'split':split,'rows_committed':rows,'total_rows':total,
      'CPU_only':True,'GPU_lock_held':False,**extra})


def adopt_prefix(manifest,language,split,rows):
    """Caller has frozen the old CPU writer at its published row boundary.

    The prefix is flushed and hashed before the old writer may exit. Rows at
    and after the conservative exclusive cursor are rebuilt by the new worker.
    """
    folder=folder_for(manifest,language,split);fp=manifest['fingerprints'][language][split];n=fp['rows']
    assert 0<=rows<n and not (folder/'complete.json').exists()
    assert not (folder/'progress.json').exists(),'Preserve an already journaled cache'
    arrays=arrays_at(folder,n)
    db=None
    try:
        lengths=arrays['lengths'][:rows]
        assert np.all((lengths>0)&(lengths<=256))
        assert np.all((arrays['labels'][:rows]>=0)&(arrays['labels'][:rows]<=1))
        assert np.all((arrays['horizons'][:rows]==1)|(arrays['horizons'][:rows]==2))
        hashes={}
        for key,array in arrays.items():
            array.flush();hashes[key]=hashlib.sha256(memoryview(array[:rows]).cast('B')).hexdigest()
        db=sqlite3.connect('file:'+str(folder.parents[2]/'dataset.sqlite')+'?mode=ro',uri=True,timeout=5)
        last=None
        if rows:
            last=db.execute("SELECT example_id FROM records INDEXED BY accepted_fingerprint_cover WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id LIMIT 1 OFFSET ?",(language,split,rows-1)).fetchone()[0]
        db.close()
        proof={'generation':manifest['generation'],'fingerprint':fp,'database_sha256':manifest['database_sha256'],
          'rows_committed':rows,'last_example_id':last,'adopted_published_prefix':True,'prefix_array_sha256':hashes}
        atomic_json(folder/'progress.json',proof);atomic_json(folder/'adoption_receipt.json',proof)
        return proof
    finally:
        if db is not None:db.close()
        for array in arrays.values():array.flush();array._mmap.close()


def cache(manifest,language,split,row_budget=None):
    folder=folder_for(manifest,language,split);folder.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (folder/'CACHE_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return build_locked(manifest,language,split,row_budget)


def build_locked(manifest,language,split,row_budget=None):
    folder=folder_for(manifest,language,split);fp=manifest['fingerprints'][language][split];n=fp['rows']
    if ready(manifest,language,split):return data.load_cache(folder)
    arrays=arrays_at(folder,n,create=True);p=folder/'progress.json'
    db=None
    try:
        previous=json.loads(p.read_text()) if p.exists() else {}
        if previous:
            assert previous['fingerprint']==fp and previous['database_sha256']==manifest['database_sha256']
        position=int(previous.get('rows_committed',0));last_id=previous.get('last_example_id')
        assert 0<=position<=n and (position==0 or last_id is not None)
        if previous.get('adopted_published_prefix'):
            for key,wanted in previous['prefix_array_sha256'].items():
                assert hashlib.sha256(memoryview(arrays[key][:position]).cast('B')).hexdigest()==wanted
        progress_state(manifest,language,split,'PREPARING_CPU_INPUT_CACHE',position,n)
        db=sqlite3.connect('file:'+str(folder.parents[2]/'dataset.sqlite')+'?mode=ro',uri=True,timeout=10)
        db.execute('PRAGMA cache_size=-131072')
        if position:
            actual=db.execute("SELECT example_id FROM records INDEXED BY accepted_fingerprint_cover WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id LIMIT 1 OFFSET ?",(language,split,position-1)).fetchone()[0]
            assert actual==last_id,'Cache ordinal and saved key disagree'
        sql="SELECT example_id,tokens,label,k FROM records WHERE language=? AND split=? AND disposition='accepted'"
        args=[language,split]
        if last_id is not None:sql+=' AND example_id>?';args.append(last_id)
        sql+=' ORDER BY example_id'
        count=0;began=time.monotonic();saved_at=began
        def save(stage):
            for array in arrays.values():array.flush()
            atomic_json(p,{'generation':manifest['generation'],'fingerprint':fp,'database_sha256':manifest['database_sha256'],
              'rows_committed':position,'last_example_id':last_id,'CPU_only':True})
            elapsed=max(.001,time.monotonic()-began)
            progress_state(manifest,language,split,stage,position,n,rows_per_second=count/elapsed)
        for eid,raw,label,k in db.execute(sql,args):
            assert position<n
            ids=np.frombuffer(raw,dtype='<u4');assert 0<len(ids)<=256
            arrays['ids'][position,:]=0;arrays['ids'][position,:len(ids)]=ids
            arrays['lengths'][position]=len(ids);arrays['labels'][position]=LABELS[label];arrays['horizons'][position]=k
            position+=1;count+=1;last_id=eid
            if count%20000==0 or time.monotonic()-saved_at>=30:
                save('PREPARING_CPU_INPUT_CACHE');saved_at=time.monotonic()
            if row_budget is not None and count>=row_budget and position<n:
                save('PAUSED_CACHE_BUILD');db.close();return None
        db.close();assert position==n,(language,split,position,n)
        save('COMPLETE')
        assert np.all((arrays['lengths']>0)&(arrays['lengths']<=256))
        actual={label+':'+str(k):int(np.count_nonzero((arrays['labels']==value)&(arrays['horizons']==k)))
          for label,value in LABELS.items() for k in (1,2)}
        if 'strata' in fp:assert actual==fp['strata'],'Cache strata disagree with frozen dataset'
        atomic_json(folder/'complete.json',{'fingerprint':fp,'rows':n,'source_database_sha256':manifest['database_sha256'],
          'resumable_cpu_cache':True,'class_horizon_counts':actual})
        return data.load_cache(folder)
    finally:
        if db is not None:db.close()
        for array in arrays.values():array.flush();array._mmap.close()


if __name__=='__main__':
    import argparse,traceback
    parser=argparse.ArgumentParser();parser.add_argument('--generation',required=True)
    parser.add_argument('--language',choices=data.LANGUAGES,required=True);parser.add_argument('--split',choices=('train','dev'),required=True)
    a=parser.parse_args();m=json.loads((data.JOB/'datasets'/a.generation/'manifest.json').read_text())
    try:cache(m,a.language,a.split)
    except Exception as exc:
        atomic_json(data.JOB/'cache_status'/(a.language+'_'+a.split+'_error.json'),{'error':repr(exc),'traceback':traceback.format_exc(),'generation':a.generation})
        raise
