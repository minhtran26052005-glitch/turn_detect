"""Launch/inspect/collect the causal retraining campaign without restarting Colab."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME = "stream_v4_campaign_20261025"


def main():
    p = argparse.ArgumentParser();p.add_argument("action", choices=["launch", "status", "collect"]);a=p.parse_args()
    evidence = Path("outputs/stream_v4_20261025");evidence.mkdir(parents=True, exist_ok=True)
    r = ColabRuntime(0, SESSION);connection = r.connect()
    if a.action == "launch":
        files = [Path("turn_v2/stream_v4_data.py"), Path("turn_v2/stream_v4_train.py")]
        files += [Path("data_v4/causal_20261025") / n for n in ["train.jsonl", "dev.jsonl", "manifest.json"]]
        archive = evidence / "campaign_upload.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in files:z.write(path, path.as_posix())
        checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
        r.upload(archive, "content/turn_detect_v2/colab/stream_v4_upload.zip");r.close();r.connect()
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
assert not (root/'jobs'/name).exists(), 'Existing campaign must be inspected; no duplicate launch'
source=root/'colab/stream_v4_upload.zip'
assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()=={checksum!r}
with zipfile.ZipFile(source) as archive:
    for item in archive.infolist():
        target=(root/item.filename).resolve();assert target.is_relative_to(root.resolve())
        data=archive.read(item)
        if target.exists():assert target.read_bytes()==data
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--',sys.executable,'-u','-m','turn_v2.stream_v4_train','--output','runs/'+name,'--rounds','3','--steps','600']
with (root/'jobs'/(name+'_launcher.log')).open('w') as out:
    process=subprocess.Popen(command,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'command':command,'bundle_sha256':{checksum!r}}}))
'''
    elif a.action == "status":
        code=f'''from pathlib import Path
import json,time
root=Path('/content/turn_detect_v2');name={NAME!r};result={{'current_unix':time.time()}}
for rel in ['jobs/'+name+'/status.json','runs/'+name+'/progress.json','runs/'+name+'/campaign.json','runs/'+name+'/initial_dev.json']:
    path=root/rel
    if path.exists():result[rel]=json.loads(path.read_text())
log=root/'jobs'/name/'stdout.log'
result['log_tail']=log.read_text()[-2500:] if log.exists() else None
evaluations=sorted((root/'runs'/name).glob('round_*/evaluations.json'))
if evaluations:result['latest_evaluation']=json.loads(evaluations[-1].read_text())[-1]
print(json.dumps(result))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
status=json.loads((root/'jobs'/name/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
archive=Path('/content/'+name+'_receipts.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['jobs/'+name,'runs/'+name]:
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and not p.name.endswith('.safetensors'):z.write(p,p.relative_to(root))
print(json.dumps({{'archive':'content/'+archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}}))
'''
    result = r.execute(code, timeout=90, echo=False)
    receipt = {"connection":connection, **result}
    (evidence/(a.action+".json")).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    assert result["status"]=="ok", "Inspect saved campaign receipt"
    text="".join(o["text"] for o in result["outputs"] if o["output_type"]=="stream");value=json.loads(text)
    if a.action=="status":
        status=value.get("jobs/"+NAME+"/status.json",{});campaign=value.get("runs/"+NAME+"/campaign.json",{})
        print(json.dumps({"status":status.get("status"),"exit_code":status.get("exit_code"),"elapsed_seconds":status.get("elapsed_seconds"),
            "progress":value.get("runs/"+NAME+"/progress.json"),"rounds_complete":len(campaign.get("rounds",[])),"best_minimum_dev_score":campaign.get("best_minimum_dev_score"),
            "latest_evaluation":value.get("latest_evaluation"),
            "log_tail":value.get("log_tail") if status.get("status")=="FAILED" or not value.get("runs/"+NAME+"/progress.json") else None}))
    elif a.action=="collect":
        archive=evidence/"campaign_receipts.zip";r.download(value["archive"],archive)
        assert archive.stat().st_size==value["bytes"] and hashlib.sha256(archive.read_bytes()).hexdigest()==value["sha256"]
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                target=(evidence/item.filename).resolve();assert target.is_relative_to(evidence.resolve())
                data=z.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        print(json.dumps(value))
    else:print(text)
    r.close()


if __name__=="__main__":main()
