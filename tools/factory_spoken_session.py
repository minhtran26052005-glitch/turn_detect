"""Rediscover a.ipynb, safely coalescing duplicate kernels on one runtime.

No kernel is created/interrupted. Multiple kernels are treated as equivalent
only after read-only boot/PID namespace/mount namespace/content-view probes.
Different runtimes or different views remain ambiguous and block mutation.
"""
import json
from pathlib import Path

from colab_runtime import assigned_servers, ColabRuntime


class SessionSelectionError(RuntimeError):
    def __init__(self, stage, **values):
        self.details = {'stage': stage, **values}
        super().__init__(json.dumps(self.details))


def choose_verified_view(matches, probes):
    if len({i for i, _ in matches}) != 1:
        raise SessionSelectionError('AMBIGUOUS_A_IPYNB_RUNTIMES', matches=len(matches))
    views = {(v['boot_id'], v['mount_namespace'], v['pid_namespace'], tuple(v['content_identity'])) for v in probes}
    if len(views) != 1 or len(probes) != len({s['kernel']['id'] for _, s in matches}):
        raise SessionSelectionError('A_IPYNB_KERNEL_VIEWS_DIFFER', matches=len(matches))
    # Once the same machine/filesystem/process view is proven, either idle
    # kernel can inspect the shared background worker without relaunching it.
    return sorted(matches, key=lambda item: (Path(item[1]['path']).name != 'a.ipynb', item[1]['path']))[0]


def select_current_a_session():
    matches = []; errors = []
    for index in range(len(assigned_servers())):
        rt = ColabRuntime(server_index=index)
        try:
            response = rt.http.get(rt.base_url+'api/sessions', timeout=15)
            response.raise_for_status()
            for session in response.json():
                name = Path(session.get('path', '')).name
                if name == 'a.ipynb' or name.startswith('a-jvsc-'):
                    matches.append((index, session))
        except Exception as error:
            errors.append({'runtime_index': index, 'error_type': type(error).__name__})
    if not matches:
        raise SessionSelectionError('NEED_CURRENT_A_IPYNB_SESSION', matches=0, errors=errors)
    if errors:
        raise SessionSelectionError('A_IPYNB_RUNTIME_DISCOVERY_INCOMPLETE', matches=len(matches), errors=errors)
    if len(matches) == 1:
        index, session = matches[0]
        if session['kernel'].get('execution_state') != 'idle':
            raise SessionSelectionError('KERNEL_BUSY_WAIT', matches=1)
        return index, session, {'matching_sessions': 1, 'selection': 'unique_current_session'}
    if len({i for i, _ in matches}) != 1:
        raise SessionSelectionError('AMBIGUOUS_A_IPYNB_RUNTIMES', matches=len(matches))
    if any(s['kernel'].get('execution_state') != 'idle' for _, s in matches):
        raise SessionSelectionError('KERNEL_BUSY_WAIT', matches=len(matches), note='No interrupt or new kernel attempted')
    unique = {}
    for index, session in matches:
        unique.setdefault(session['kernel']['id'], (index, session))
    code = """import pathlib,os,json
p=pathlib.Path('/content');s=p.stat()
print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
'mount_namespace':os.readlink('/proc/self/ns/mnt'),'pid_namespace':os.readlink('/proc/self/ns/pid'),
'content_identity':[s.st_dev,s.st_ino]}))"""
    probes = []
    for index, session in unique.values():
        rt = ColabRuntime(server_index=index, session_path=session['path'])
        # The ID was just discovered from this runtime's authenticated sessions
        # API; using it avoids a redundant metadata request, not a stale pin.
        rt.kernel_id = session['kernel']['id']
        try:
            result = rt.execute(code, timeout=30, echo=False)
            if result['status'] != 'ok':
                raise SessionSelectionError('A_IPYNB_VIEW_PROBE_FAILED', matches=len(matches))
            raw = ''.join(x.get('text', '') for x in result['outputs'] if x['output_type'] == 'stream')
            probes.append(json.loads(raw))
        except SessionSelectionError:
            raise
        except Exception as error:
            raise SessionSelectionError('A_IPYNB_VIEW_PROBE_FAILED', matches=len(matches), error_type=type(error).__name__) from None
        finally:
            rt.close()
    index, session = choose_verified_view(matches, probes)
    return index, session, {'matching_sessions': len(matches), 'distinct_kernels': len(unique),
                            'selection': 'same_runtime_same_process_and_filesystem_view',
                            'verified_boot_id': probes[0]['boot_id'], 'views_verified': True}


def select_owned_a_session(preferred_boot):
    """Select a known campaign owner across runtimes only after full probes.

    The older ambiguous-runtime guard stays unchanged. This stricter scoped
    entry point additionally requires verified Drive identity/owner and no
    generation process on any other runtime; it never chooses by tab order.
    """
    if not preferred_boot:raise SessionSelectionError('KNOWN_OWNER_BOOT_REQUIRED')
    matches=[];errors=[]
    for index in range(len(assigned_servers())):
        rt=ColabRuntime(server_index=index)
        try:
            response=rt.http.get(rt.base_url+'api/sessions',timeout=15);response.raise_for_status()
            for s in response.json():
                name=Path(s.get('path','')).name
                if name=='a.ipynb' or name.startswith('a-jvsc-'):matches.append((index,s))
        except Exception as e:errors.append({'runtime_index':index,'error_type':type(e).__name__})
        finally:rt.close()
    if errors:raise SessionSelectionError('OWNER_RUNTIME_DISCOVERY_INCOMPLETE',errors=errors)
    if not matches:raise SessionSelectionError('NEED_CURRENT_A_IPYNB_SESSION',matches=0)
    if any(s['kernel'].get('execution_state')!='idle' for _,s in matches):
        raise SessionSelectionError('OWNER_VIEW_BUSY_WAIT',matches=len(matches))
    code=r'''import pathlib,json,os
p=pathlib.Path('/content');s=p.stat();o=pathlib.Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026/factory_spoken_campaign_v9')
v={'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'mount_namespace':os.readlink('/proc/self/ns/mnt'),'pid_namespace':os.readlink('/proc/self/ns/pid'),'content_identity':[s.st_dev,s.st_ino],'identity_verified':False,'owner':None,'generation_processes':[]}
if os.path.ismount('/content/drive') and (o.parent/'destination_identity.json').exists():
 i=json.loads((o.parent/'destination_identity.json').read_text());v['identity_verified']=i.get('nonce')=='9e618b0f2db442a6a9ee3ee18f1d7143' and i.get('intended_path')==str(o.parent)
 if (o/'owner.json').exists():v['owner']=json.loads((o/'owner.json').read_text())
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  c=f.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if '/content/' in c and any(x in c for x in ['/factory_spoken_campaign_v','/factory_dialogue_campaign_v','/factory_paragraphs_v']):
   r=(f.parent/'stat').read_text();v['generation_processes'].append({'pid':int(f.parent.name),'start_ticks':r[r.rfind(')')+2:].split()[19],'cmdline':c[:300]})
 except OSError:pass
print(json.dumps(v))'''
    unique={}
    for index,s in matches:unique.setdefault((index,s['kernel']['id']),(index,s))
    probes=[]
    for index,s in unique.values():
        rt=ColabRuntime(server_index=index,session_path=s['path']);rt.kernel_id=s['kernel']['id']
        try:
            result=rt.execute(code,timeout=35,echo=False)
            if result['status']!='ok':raise SessionSelectionError('OWNER_VIEW_PROBE_FAILED')
            raw=''.join(x.get('text','') for x in result['outputs'] if x['output_type']=='stream')
            value=json.loads(raw);probes.append((index,s,value))
        finally:rt.close()
    selected=[(i,s,v) for i,s,v in probes if v['boot_id']==preferred_boot]
    if not selected:raise SessionSelectionError('KNOWN_OWNER_RUNTIME_NOT_ASSIGNED',preferred_boot=preferred_boot)
    indices={i for i,_,_ in selected}
    if len(indices)!=1:raise SessionSelectionError('KNOWN_OWNER_DUPLICATE_RUNTIME_ENDPOINTS')
    for i,s,v in probes:
        if i not in indices and v['generation_processes']:
            raise SessionSelectionError('OTHER_RUNTIME_HAS_GENERATION_WORKER',runtime_index=i)
    for _,_,v in selected:
        owner=v['owner']
        if not v['identity_verified'] or not owner or owner['boot_id']!=preferred_boot:
            raise SessionSelectionError('KNOWN_OWNER_DRIVE_IDENTITY_MISMATCH')
        for process in v['generation_processes']:
            if process['pid']!=owner['pid'] or process['start_ticks']!=owner['start_ticks']:
                raise SessionSelectionError('KNOWN_OWNER_PROCESS_MISMATCH')
    owned_matches=[(i,s) for i,s in matches if i in indices]
    index,session=choose_verified_view(owned_matches,[v for _,_,v in selected])
    return index,session,{'selection':'explicit_known_owner_boot_with_verified_drive_and_all_runtime_process_probes',
        'verified_boot_id':preferred_boot,'assigned_runtime_count':len({i for i,_,_ in probes}),
        'selected_matching_sessions':len(owned_matches),'selected_views_verified':True,
        'other_runtimes_have_no_generation_worker':True,'other_runtime_boots':[v['boot_id'] for i,_,v in probes if i not in indices]}
