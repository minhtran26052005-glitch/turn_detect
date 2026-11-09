"""Score only audited-clean inputs; protected/reserve overlaps never infer."""
from collections import Counter
import gc
import gzip
import hashlib
import json
from pathlib import Path

from . import translation_segmenter_v15d_test_v2 as ev


def infer_clean(model,encoded,clean,lang):
    return ev.infer(model,tuple(part[clean] for part in encoded),lang)


def score_language(db,lang,manifest):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    from .common import normalize
    from .metrics import classification_metrics
    from .translation_segmenter_v15c import encode
    folder=ev.JOB/'frozen'/lang
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    packed=db.execute("SELECT record_json FROM examples WHERE language=? AND split='test' ORDER BY example_id",(lang,)).fetchall()
    records_sha=hashlib.sha256();rows=[]
    for (raw,) in packed:
        records_sha.update(raw.encode());records_sha.update(b'\n');row=json.loads(raw)
        if row.get('punctuation_in_model_input') is not False:raise RuntimeError('Unexpected frozen input protocol')
        rows.append(row)
    del packed
    ev.status('ENCODING_FROZEN_DIAGNOSTIC',language=lang,test_rows=len(rows))
    encoded=encode(rows,tokenizer)
    clean,audit=ev.audit_test(db,lang,rows,encoded,tokenizer)
    ev.wait_for_gpu(language=lang)
    model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).cuda()
    probs=infer_clean(model,encoded,clean,lang)
    threshold=manifest['checkpoints'][lang]['threshold_from_dev']
    labels=encoded[2][clean];horizons=encoded[3][clean];predictions=(probs>=threshold).astype(np.int64)
    full_predictions=np.full(len(rows),-1,dtype=np.int64);full_predictions[clean]=predictions
    overall=classification_metrics(labels,probs,threshold)
    period=np.asarray([j for j,i in enumerate(clean) if rows[int(i)]['label']=='EOT' and rows[int(i)].get('target_punctuation') in {'.','。'}],dtype=np.int64)
    internal=np.asarray([j for j,i in enumerate(clean) if rows[int(i)]['label']=='NEOT' and rows[int(i)].get('target_punctuation') in {',','，',';','；',':','：','、'}],dtype=np.int64)
    trace=ev.JOB/f'test_predictions_{lang}.jsonl.gz';sampled=Counter()
    with gzip.open(trace,'wt',encoding='utf-8') as stream,(ev.JOB/'review_samples.jsonl').open('a',encoding='utf-8') as review:
        for j,i in enumerate(clean):
            row=rows[int(i)];predicted='EOT' if predictions[j] else 'NEOT'
            value={**row,'context':normalize(row['context']),'text':normalize(row['text']),
                'predicted':predicted,'p_eot':float(probs[j]),'correct':predicted==row['label'],
                'included_in_clean_test':True,'threshold_from_dev':threshold}
            stream.write(json.dumps(value,ensure_ascii=False)+'\n')
            key=(row['label'],row['lookahead_words'],value['correct'])
            if sampled[key]<15:review.write(json.dumps(value,ensure_ascii=False)+'\n');sampled[key]+=1
    result={**manifest['checkpoints'][lang],'threshold_selected_on_test':False,
        'test_rows_sha256':records_sha.hexdigest(),'leakage_audit':audit,'metrics':overall,
        'raw_test_metrics_before_overlap_filter':None,
        'excluded_rows_were_not_inferred':True,
        'by_lookahead_words':{str(k):classification_metrics(labels[horizons==k],probs[horizons==k],threshold) for k in (1,2)},
        'source_cluster_95pct_intervals':ev.cluster_intervals(rows,clean,encoded[2],full_predictions),
        'terminal_period_eot_recall':float((probs[period]>=threshold).mean()) if len(period) else None,
        'terminal_period_targets':int(len(period)),
        'internal_punctuation_no_eot_rate':float((probs[internal]<threshold).mean()) if len(internal) else None,
        'internal_punctuation_targets':int(len(internal)),'test_evaluated':True,
        'prose_test_accuracy_and_f1_90':min(overall['accuracy'],overall['eot_f1'])>=.9,
        'full_trace_remote_only':str(trace),'full_trace_sha256':ev.file_sha(trace),
        'scoring_source_sha256':ev.file_sha(Path(__file__))}
    del model,tokenizer,encoded,rows,probs,labels,predictions,full_predictions
    gc.collect();torch.cuda.empty_cache()
    return result
