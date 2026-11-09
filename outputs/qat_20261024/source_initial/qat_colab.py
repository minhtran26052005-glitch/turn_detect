"""Start/inspect/collect bounded QAT jobs on the already connected Colab VM."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime


SESSION = "a-jvsc-f2543605-ce4e-4c3d-a99f-688dd99e9b33d84c1b1f-d7f4-4eb7-9690-6a3514d8ec38.ipynb"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["launch", "status", "collect"])
    p.add_argument("--group-size", type=int, choices=[32, 64], required=True)
    a = p.parse_args()
    name = f"qat_gru96_g{a.group_size}_20261024"
    evidence = Path("outputs/qat_20261024")
    evidence.mkdir(parents=True, exist_ok=True)
    r = ColabRuntime(0, SESSION)
    connection = r.connect()
    if a.action == "launch":
        sources = ["turn_v2/qat_student.py", "turn_v2/quantize_student4.py", "turn_v2/tiny_student.py", "turn_v2/distill.py", "turn_v2/common.py", "turn_v2/metrics.py", "tools/distill_job.py"]
        payload = {n: base64.b64encode(Path(n).read_bytes()).decode("ascii") for n in sources}
        code = f'''from pathlib import Path
import base64,hashlib,json,subprocess,sys
qat_root=Path('/content/turn_detect_v2')
qat_name={name!r}
assert not (qat_root/'jobs'/qat_name).exists(), 'QAT job already exists; inspect it instead of relaunching'
assert not (qat_root/'runs'/qat_name).exists(), 'QAT output already exists'
qat_sources={payload!r}
for relative,encoded in qat_sources.items():
    path=qat_root/relative
    data=base64.b64decode(encoded)
    if path.exists():
        assert path.read_bytes()==data, 'Existing remote source differs: '+relative
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
qat_command=[sys.executable,'tools/distill_job.py','--job','jobs/'+qat_name,'--timeout','900','--',sys.executable,'-u','-m','turn_v2.qat_student','--student','runs/distill_gru_h96_e64_l1_20261024','--output','runs/'+qat_name,'--group-size',{str(a.group_size)!r},'--epochs','8','--lr','0.0001','--batch-size','256','--seed','43']
qat_launcher=qat_root/'jobs'/(qat_name+'_launcher.log')
with qat_launcher.open('w') as stream:
    qat_process=subprocess.Popen(qat_command,cwd=qat_root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':qat_name,'launcher_pid':qat_process.pid,'command':qat_command,'source_sha256':{{n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in qat_sources.items()}}}}))
'''
    elif a.action == "status":
        code = f'''from pathlib import Path
import json,time
qat_root=Path('/content/turn_detect_v2');qat_name={name!r}
qat_files={{}}
for relative in ['jobs/'+qat_name+'/status.json','runs/'+qat_name+'/progress.json','runs/'+qat_name+'/history.json','runs/'+qat_name+'/training_report.json']:
    path=qat_root/relative
    if path.exists():qat_files[relative]=json.loads(path.read_text())
qat_log=qat_root/'jobs'/qat_name/'stdout.log'
print(json.dumps({{'current_unix':time.time(),'files':qat_files,'log_tail':qat_log.read_text()[-3000:] if qat_log.exists() else None}}))
'''
    else:
        code = f'''from pathlib import Path
import hashlib,json,zipfile
qat_root=Path('/content/turn_detect_v2');qat_name={name!r}
qat_status=json.loads((qat_root/'jobs'/qat_name/'status.json').read_text())
assert qat_status['status']=='COMPLETED' and qat_status['exit_code']==0
qat_archive=Path('/content/'+qat_name+'_artifacts.zip')
with zipfile.ZipFile(qat_archive,'w',zipfile.ZIP_DEFLATED) as out:
    for folder in ['runs/'+qat_name,'jobs/'+qat_name]:
        for path in sorted((qat_root/folder).rglob('*')):
            if path.is_file():out.write(path,path.relative_to(qat_root))
print(json.dumps({{'archive':'content/'+qat_archive.name,'bytes':qat_archive.stat().st_size,'sha256':hashlib.file_digest(qat_archive.open('rb'),'sha256').hexdigest()}}))
'''
    result = r.execute(code, timeout=90, echo=False)
    receipt = {"connection": connection, **result}
    (evidence / (name + "_" + a.action + ".json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result["status"] == "ok", "Inspect saved QAT receipt"
    output = "".join(o["text"] for o in result["outputs"] if o["output_type"] == "stream")
    if a.action == "collect":
        metadata = json.loads(output)
        archive = evidence / (name + "_artifacts.zip")
        r.download(metadata["archive"], archive)
        assert archive.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
        with zipfile.ZipFile(archive) as incoming:
            for member in incoming.infolist():
                target = (evidence / member.filename).resolve()
                assert target.is_relative_to(evidence.resolve())
                data = incoming.read(member)
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
    r.close()
    print(output)


if __name__ == "__main__":
    main()
