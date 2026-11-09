"""Pinned small multilingual encoder candidate; download only, no GPU."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import model_info,snapshot_download


def main():
    repo='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';revision=model_info(repo,token=False).sha
    path=Path(snapshot_download(repo,revision=revision,token=False,max_workers=2,allow_patterns=['config.json','model.safetensors',
        'tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','README.md',
        'config_sentence_transformers.json','1_Pooling/config.json','modules.json','LICENSE']))
    files=[]
    for item in sorted(path.rglob('*')):
        if item.is_file():
            with item.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files.append({'name':str(item.relative_to(path)),'bytes':item.stat().st_size,'sha256':digest})
    output=Path('outputs/natural_v6_20261025/encoder_download.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'repo':repo,'revision':revision,'local_path':str(path),'files':files,'license':'Apache2.0 model card'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ENCODER_READY','revision':revision,'bytes':sum(f['bytes'] for f in files)}),flush=True)


if __name__=='__main__':main()
