import json,time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from turn_v2.common import read_jsonl,encode_pair,LABEL2ID,write_json,write_jsonl,file_hash
from turn_v2.metrics import probabilities,classification_metrics
root=Path('/content/turn_detect_v2');out=root/'runs/trial_gold_20261024_093324'
model_dir=out/'best';contract=json.loads((model_dir/'turn_detector_config.json').read_text())
rows=list(read_jsonl(root/'test.jsonl'));tokenizer=AutoTokenizer.from_pretrained(model_dir)
model=AutoModelForSequenceClassification.from_pretrained(model_dir).to('cuda').eval()
collate=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8)
probs=[];started=time.monotonic()
for start in range(0,len(rows),16):
    batch=collate([encode_pair(tokenizer,row.get('context',''),row['text'],contract['max_length']) for row in rows[start:start+16]]).to('cuda')
    with torch.inference_mode():probs.extend(probabilities(model(**batch).logits.float().cpu().numpy()).tolist())
result={'status':'DESCRIPTIVE_LEGACY_REGRESSION','model_selected_on':'frozen dev','threshold_frozen':contract['threshold'],'test_sha256':file_hash(root/'test.jsonl'),'metrics':classification_metrics([LABEL2ID[row['label']] for row in rows],probs,contract['threshold']),'elapsed_seconds':time.monotonic()-started,'limits':'Legacy test was viewed previously. Results are regression diagnostics; no model or threshold changes are made from it.'}
destination=out/'legacy_regression';write_json(destination/'report.json',result)
write_jsonl(destination/'predictions.jsonl',[{**row,'prob_EOT':prob,'prediction':'EOT' if prob>=contract['threshold'] else 'NEOT'} for row,prob in zip(rows,probs,strict=True)])
print(json.dumps(result,indent=2),flush=True)
