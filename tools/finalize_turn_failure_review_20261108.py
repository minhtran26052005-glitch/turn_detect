"""Publish small review files, verify byte hashes and preserve the paused campaign."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/turn_failure_literature_20261108'
MAIN = ROOT / 'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
MARKER = 'USER_FAILED_CHECKPOINT_CLEANUP_AND_LITERATURE_COMPLETE_20261108'
NOTE = '''\n\n## USER_FAILED_CHECKPOINT_CLEANUP_AND_LITERATURE_COMPLETE_20261108
Theo yêu cầu user: đã xóa checkpoint/export của các nhánh được xác nhận fail;
receipt ở ../failed_checkpoint_cleanup_20261108/deletion_receipt.json.
Các file trong receipt bị xóa có chủ đích, không phải lỗi mới và không tự phục hồi/retrain.
Main full checkpoint, teacher và dữ liệu/báo cáo/traces được giữ.
Rà soát 10 paper cùng thực nghiệm đã lưu hoàn thành tại
../turn_failure_literature_20261108/README.md và review.html.
Thiết kế A/B context là đề xuất, chưa chạy. Main vẫn PAUSED; research không mở lại TRAIN/TEST.
'''


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs).get('href'))
        self.tables += tag == 'table'


def main():
    page = Page()
    page.feed((OUT / 'review.html').read_text(encoding='utf-8'))
    assert page.tables == 1 and len(page.links) >= 11
    for link in page.links:
        if not link.startswith(('https://', '#')):
            assert (OUT / link).exists(), link
    files = {}
    for path in OUT.iterdir():
        if path.is_file() and path.name != 'publication_verification.json':
            raw = path.read_bytes()
            files[path.name] = {'base64': base64.b64encode(raw).decode(),
                                'sha256': hashlib.sha256(raw).hexdigest()}
    assert sum(len(base64.b64decode(x['base64'])) for x in files.values()) < 5 * 1024**2
    metadata = {'complete': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
                'review': 'outputs/turn_failure_literature_20261108/README.md',
                'new_training_or_test': False, 'proposed_context_experiment_launched': False,
                'primary_papers': 10}
    code = '''import json,base64,pathlib,hashlib,datetime,shutil
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/turn_failure_literature_20261108';p.mkdir(exist_ok=True)
files=json.loads(FILES);meta=json.loads(META);checks=[]
assert shutil.disk_usage(r).free>=13*2**30
for name,x in files.items():
 assert pathlib.PurePath(name).name==name
 b=base64.b64decode(x['base64']);assert hashlib.sha256(b).hexdigest()==x['sha256']
 q=p/name;q.write_bytes(b);assert hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256'];checks.append({'name':name,'sha256':x['sha256']})
m=r/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
s=m/'task_state.json';state=json.loads(s.read_text());assert state['desired_state']=='PAUSED_BY_USER_FOR_TEST' and (m/'STOP_REQUESTED').exists()
state['user_requested_failure_research']=meta;t=s.with_name(s.name+'.tmp');t.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\\n');t.replace(s)
q=m/'README.md';old=q.read_text();note=NOTE
if MARKER not in old:q.write_text(old+note)
print(json.dumps({'complete':True,'files':checks,'main_desired_state':state['desired_state'],'stop_requested':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ssd_free_gib':shutil.disk_usage(r).free/2**30}))
'''.replace('FILES', repr(json.dumps(files))).replace('META', repr(json.dumps(metadata))).replace('NOTE', repr(NOTE)).replace('MARKER', repr(MARKER))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'edabk@100.91.209.124', 'python3 -'], input=code, text=True,
                            encoding='utf-8', capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr)
    proof = json.loads(result.stdout)
    assert proof['complete']
    state_path = MAIN / 'task_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8-sig'))
    state['user_requested_failure_research'] = metadata
    temporary = state_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(state_path)
    readme = MAIN / 'README.md'
    old = readme.read_text(encoding='utf-8-sig')
    if MARKER not in old:
        readme.write_text(old + NOTE, encoding='utf-8')
    proof.update(html_table_and_links_checked=True, new_training_or_test=False)
    (OUT / 'publication_verification.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'sha_verified_remote_files': len(proof['files']),
                      'main_desired_state': proof['main_desired_state'], 'new_training_or_test': False}))


if __name__ == '__main__':
    main()
