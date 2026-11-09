"""Bounded pilot jobs in the existing Colab VM, with SHA-verified collection."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["launch", "status", "collect"])
    p.add_argument("--phase", choices=["teacher", "student", "final", "real_pause"], default="teacher")
    a = p.parse_args()
    name = "tri_" + a.phase + "_20261024"
    evidence = Path("outputs/trilingual_20261024")
    evidence.mkdir(parents=True, exist_ok=True)
    r = ColabRuntime(0, SESSION)
    connection = r.connect()
    if a.action == "launch":
        archive = evidence / (name + "_upload.zip")
        files = [p for p in Path("turn_v2").glob("trilingual*.py")]
        if a.phase == "teacher":
            files += [Path("data_v3/pilot_trilingual_20261024") / n for n in ["train.jsonl", "dev.jsonl", "manifest.json"]]
        if a.phase == "final":
            files += [p for p in Path("models/turn_v3_trilingual_pilot_20261024/onnx").iterdir() if p.is_file()]
            files += [Path("data_v3/pilot_trilingual_20261024/test.jsonl")]
        if a.phase == "real_pause":
            files += [Path("data_v3/livekit_eot_20261024/cases.jsonl"), Path("data_v3/livekit_eot_20261024/sources.json")]
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in files:
                z.write(path, path.as_posix())
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        r.upload(archive, "content/turn_detect_v2/colab/" + archive.name)
        r.close(); r.connect()
        module = "turn_v2.trilingual_" + a.phase
        code = f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
tri_root=Path('/content/turn_detect_v2');tri_name={name!r}
assert not (tri_root/'jobs'/tri_name).exists(), 'Job already exists; inspect instead of duplicate launch'
tri_archive=tri_root/'colab'/{archive.name!r}
assert hashlib.file_digest(tri_archive.open('rb'),'sha256').hexdigest()=={checksum!r}
with zipfile.ZipFile(tri_archive) as archive:
    for member in archive.infolist():
        target=(tri_root/member.filename).resolve();assert target.is_relative_to(tri_root.resolve())
        data=archive.read(member)
        if target.exists():assert target.read_bytes()==data, 'Existing pilot file differs: '+member.filename
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
tri_command=[sys.executable,'tools/distill_job.py','--job','jobs/'+tri_name,'--timeout','1800','--',sys.executable,'-u','-m',{module!r},'--output','runs/'+tri_name]
with (tri_root/'jobs'/(tri_name+'_launcher.log')).open('w') as stream:
    tri_process=subprocess.Popen(tri_command,cwd=tri_root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':tri_name,'launcher_pid':tri_process.pid,'command':tri_command,'bundle_sha256':{checksum!r}}}))
'''
    elif a.action == "status":
        code = f'''from pathlib import Path
import json,time
tri_root=Path('/content/turn_detect_v2');tri_name={name!r};tri_files={{}}
for rel in ['jobs/'+tri_name+'/status.json','runs/'+tri_name+'/progress.json','runs/'+tri_name+'/history.json','runs/'+tri_name+'/training_report.json']:
    path=tri_root/rel
    if path.exists():tri_files[rel]=json.loads(path.read_text())
tri_log=tri_root/'jobs'/tri_name/'stdout.log'
print(json.dumps({{'current_unix':time.time(),'files':tri_files,'log_tail':tri_log.read_text()[-4500:] if tri_log.exists() else None}}))
'''
    else:
        code = f'''from pathlib import Path
import hashlib,json,zipfile
tri_root=Path('/content/turn_detect_v2');tri_name={name!r}
tri_status=json.loads((tri_root/'jobs'/tri_name/'status.json').read_text());assert tri_status['status']=='COMPLETED' and tri_status['exit_code']==0
tri_archive=Path('/content/'+tri_name+'_artifacts.zip')
with zipfile.ZipFile(tri_archive,'w',zipfile.ZIP_DEFLATED) as out:
    for prefix in ['runs/'+tri_name,'jobs/'+tri_name]:
        for path in sorted((tri_root/prefix).rglob('*')):
            if not path.is_file() or '/cache/' in str(path) or path.name.endswith('.safetensors'):continue
            out.write(path,path.relative_to(tri_root))
print(json.dumps({{'archive':'content/'+tri_archive.name,'bytes':tri_archive.stat().st_size,'sha256':hashlib.file_digest(tri_archive.open('rb'),'sha256').hexdigest()}}))
'''
    result = r.execute(code, timeout=120, echo=False)
    receipt = {"connection": connection, **result}
    (evidence / (name + "_" + a.action + ".json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result["status"] == "ok", "Inspect saved pilot receipt"
    output = "".join(o["text"] for o in result["outputs"] if o["output_type"] == "stream")
    value = json.loads(output)
    if a.action == "status":
        files = value["files"]
        status = files.get("jobs/" + name + "/status.json", {})
        print(json.dumps({"job": name, "status": status.get("status"), "exit_code": status.get("exit_code"),
            "elapsed_seconds": status.get("elapsed_seconds"), "progress": files.get("runs/" + name + "/progress.json"),
            "last_evaluation": files.get("runs/" + name + "/history.json", [None])[-1],
            "log_tail": value["log_tail"] if status.get("status") == "FAILED" or not files.get("runs/" + name + "/progress.json") else None}))
    elif a.action == "collect":
        archive = evidence / (name + "_artifacts.zip")
        r.download(value["archive"], archive)
        assert archive.stat().st_size == value["bytes"] and hashlib.sha256(archive.read_bytes()).hexdigest() == value["sha256"]
        with zipfile.ZipFile(archive) as incoming:
            for member in incoming.infolist():
                target = (evidence / member.filename).resolve(); assert target.is_relative_to(evidence.resolve())
                data = incoming.read(member)
                if target.exists():assert target.read_bytes() == data
                else:
                    target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(data)
        print(json.dumps(value))
    else:
        print(json.dumps(value))
    r.close()


if __name__ == "__main__":
    main()
