"""Download a pinned public teacher; never allocates GPU or alters training."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import model_info,snapshot_download


def main():
    repo='Qwen/Qwen3-4B-Instruct-2507'
    info=model_info(repo,token=False)
    path=Path(snapshot_download(repo,revision=info.sha,token=False,max_workers=2,
        allow_patterns=['*.json','*.safetensors','tokenizer.*','vocab*','merges.txt','README.md','LICENSE']))
    files=[]
    for item in sorted(path.iterdir()):
        if item.is_file():
            with item.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files.append({'name':item.name,'bytes':item.stat().st_size,'sha256':digest})
    output=Path('outputs/natural_v5_20261025/qwen_download.json');output.parent.mkdir(parents=True,exist_ok=True)
    receipt={'repo':repo,'revision':info.sha,'local_path':str(path),'license':'Apache2.0 model card','files':files,'GPU_allocated':False}
    output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'status':'TEACHER_WEIGHTS_READY','revision':info.sha,'bytes':sum(f['bytes'] for f in files),'files':len(files)}),flush=True)


if __name__=='__main__':main()
