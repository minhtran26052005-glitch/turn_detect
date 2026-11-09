"""Pin and download the Vietnamese ASR control; no GPU allocation."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import model_info,snapshot_download


def main():
    repo='vinai/PhoWhisper-large';revision=model_info(repo,token=False).sha
    path=Path(snapshot_download(repo,revision=revision,token=False,max_workers=2,
        allow_patterns=['*.json','pytorch_model*.bin','*.safetensors','tokenizer.*','vocab*','merges.txt','README.md','LICENSE*','normalizer*']))
    files=[]
    for item in sorted(path.iterdir()):
        if item.is_file():
            with item.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files.append({'name':item.name,'bytes':item.stat().st_size,'sha256':digest})
    output=Path('outputs/natural_v6_20261025/pho_download.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'repo':repo,'revision':revision,'local_path':str(path),'files':files,'license':'BSD3Clause model card',
        'attribution':'Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen. PhoWhisper: Automatic Speech Recognition for Vietnamese. ICLR2024 Tiny Papers.'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ASR_READY','revision':revision,'bytes':sum(f['bytes'] for f in files)}),flush=True)


if __name__=='__main__':main()
