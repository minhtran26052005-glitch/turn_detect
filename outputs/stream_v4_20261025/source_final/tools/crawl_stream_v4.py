"""Collect fresh three-language streaming review cases; labels stay unassigned."""
from collections import Counter
import json
from pathlib import Path
import re
import time

import requests

from turn_v2.common import digest, normalize, write_json, write_jsonl


TITLES={"vi":["Huế","Cần Thơ","Vịnh Hạ Long","Cà phê Việt Nam"],
        "en":["Sydney","Coffee","Public library","Hangzhou"],
        "zh":["杭州市","成都市","咖啡","图书馆"]}


def main():
    output=Path("data_v4/fresh_crawl_20261025");output.mkdir(parents=True,exist_ok=False)
    http=requests.Session();http.headers["User-Agent"]="TurnDetectorResearch/0.4 (bounded provenance-preserving streaming review)"
    sources=[];cases=[]
    for lang,titles in TITLES.items():
        response=http.get(f"https://{lang}.wikipedia.org/w/api.php",params={"action":"query","format":"json","formatversion":2,"prop":"extracts|info","explaintext":1,"exintro":1,"redirects":1,"inprop":"url","titles":"|".join(titles)},timeout=45)
        response.raise_for_status()
        for page in response.json().get("query",{}).get("pages",[]):
            text=page.get("extract","")
            if len(text)<100:continue
            source={"language":lang,"title":page["title"],"source_id":f"{lang}wiki:{page['pageid']}","source_url":page.get("fullurl",f"https://{lang}.wikipedia.org/?curid={page['pageid']}"),
                    "text":text,"sha256":digest(text),"license":"Wikipedia CC BY-SA terms, attribution retained"}
            sources.append(source)
            # Freeze source and all candidate labels BEFORE any new-model score.
            punctuation=list(re.finditer(r"[,，、；;：:.。!?！？]",text))
            for position in punctuation[:20]:
                prefix=text[:position.end()]
                if len(normalize(prefix))<20:continue
                cases.append({"candidate_id":digest(source["source_id"]+"\0"+str(position.end())),"language":lang,
                    "source_id":source["source_id"],"source_url":source["source_url"],"source_title":source["title"],
                    "text":prefix,"context":"","punctuation":position.group(),"observed_suffix_sha256":digest(text[position.end():]),
                    "label":None,"label_status":"PENDING_INDEPENDENT_REVIEW","is_gold":False,
                    "instruction":"Decide semantic turn completeness from observed text only; comma/period is not a turn gold label. AMBIGUOUS is allowed."})
        time.sleep(.5)
    assert all(any(r["language"]==lang for r in cases) for lang in TITLES)
    write_jsonl(output/"sources.jsonl",sources);write_jsonl(output/"review_cases.jsonl",cases)
    write_json(output/"manifest.json",{"sources":len(sources),"cases":dict(Counter(r["language"] for r in cases)),"labels_fabricated":False,
        "acceptance_eligible":False,"scope":"Fresh crawled prose streaming cases, pending labels; NOT real dialogue gold","training_use":False})
    print(json.dumps({"sources":len(sources),"cases":dict(Counter(r["language"] for r in cases)),"gold_labels":0}),flush=True)


if __name__=="__main__":main()
