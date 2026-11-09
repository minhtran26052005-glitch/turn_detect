"""Fetch only compact reports/reviews; full traces and model pins stay on SSH."""
from html import escape
import base64
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/multilingual_student_test_20261031_v2'
REMOTE='/mnt/hdd500/turn_detect_training/multilingual_distill_test_20261031_v2'
LIMIT=5*1024**2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remote-output',default=REMOTE)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();remote=args.remote_output;destination=args.output_dir
    assert remote.startswith('/mnt/hdd500/turn_detect_training/multilingual_distill_test_')
    assert shutil.disk_usage(ROOT).free>=10*1024**3+LIMIT,'Keep local storage reserve'
    code=f'''import pathlib,json,hashlib,base64
p=pathlib.Path({remote!r});names=['report.json','frozen_dev.json','input_manifest.json','checkpoint_receipt.json','status.json','launcher.json','review_vi.json','review_en.json','review_zh.json']
out={{}}
for name in names:out[name]=json.loads((p/name).read_text())
assert out['report.json']['complete'] and out['status.json']['stage']=='COMPLETE'
out['report_sha256']=hashlib.sha256((p/'report.json').read_bytes()).hexdigest()
out['report_bytes_b64']=base64.b64encode((p/'report.json').read_bytes()).decode()
raw=json.dumps(out,ensure_ascii=False);assert len(raw.encode())<={LIMIT};print(raw)
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],
        input=code,capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    assert len(result.stdout.encode())<=LIMIT
    bundle=json.loads(result.stdout);out=destination;out.mkdir(parents=True,exist_ok=True)
    for name,value in bundle.items():
        if name.endswith('.json'):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Preserve Linux LF bytes instead of Windows text-mode CRLF conversion.
    (out/'report.json').write_bytes(base64.b64decode(bundle['report_bytes_b64']))
    assert hashlib.sha256((out/'report.json').read_bytes()).hexdigest()==bundle['report_sha256'],'Report bytes differ from the remote SHA'
    report=bundle['report.json'];step=report['checkpoint']['step'];threshold=report['threshold_from_same_checkpoint_dev']
    def pct(x):return '—' if x is None else f'{100*x:.2f}%'
    rows=[]
    for lang,d in report['languages'].items():
        m=d['metrics_after_cycle84_audit'];raw=d['existing_C75_subset_metrics_before_cycle84_audit']
        rows.append(f"<tr><td>{lang.upper()}</td><td>{d['input_rows']:,}</td><td>{m['samples']:,}</td><td>{pct(m['accuracy'])}</td><td>{pct(m['eot_f1'])}</td><td>{pct(raw['accuracy'])} / {pct(raw['eot_f1'])}</td></tr>")
    samples=[]
    comparison=''
    if report.get('comparison_to_step_106000'):
        cells=[]
        for lang,d in report['comparison_to_step_106000']['by_language'].items():
            cells.append(f"<tr><td>{lang.upper()}</td><td>{pct(d['old']['accuracy'])} / {pct(d['old']['eot_f1'])}</td><td>{pct(d['new']['accuracy'])} / {pct(d['new']['eot_f1'])}</td><td>{d['eot_f1_change_percentage_points']:+.2f}</td></tr>")
        comparison='<h2>So với checkpoint 106.000</h2><p>Mỗi checkpoint dùng ngưỡng chọn riêng trên DEV; không chọn lại theo test.</p><table><tr><th>Tiếng</th><th>Cũ: accuracy / F1</th><th>Mới: accuracy / F1</th><th>Δ F1, điểm %</th></tr>'+''.join(cells)+'</table>'
    for lang in ('vi','en','zh'):
        samples.append(f'<h2>Ví dụ {lang.upper()}</h2>')
        for row in bundle[f'review_{lang}.json']:
            label=escape(row['label']);prediction=escape(row['predicted'])
            samples.append(f'<article><p><b>{"Đúng" if row["correct"] else "Sai"}</b> · Nhãn {label} · Dự đoán {prediction} · P(EOT) {row["p_eot"]:.4f} · nhìn trước {row["lookahead_words"]} từ</p><p><b>Context:</b> {escape(row["context"])}</p><p><b>Text:</b> {escape(row["text"])}</p></article>')
    html=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test student đa ngôn ngữ</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f6f8fb;color:#19263a;max-width:1100px;margin:32px auto;padding:0 20px}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border:1px solid #d8e0eb;text-align:left}}article,.note{{background:white;padding:16px 22px;margin:15px 0;border-radius:10px;overflow-wrap:anywhere}}.note{{border-left:4px solid #d69a24}}code{{font-size:13px}}</style>
<h1>Test chẩn đoán student chung VI / EN / ZH</h1>
<p>Checkpoint bước <b>{step:,}</b> · ngưỡng chung <b>{threshold:.2f}</b>, chọn trên DEV của chính checkpoint này trước khi chấm test.</p>
<div class="note">Đây là C75 nhãn theo dấu câu, đã từng được xem. Không phải test hội thoại/audio độc lập hoặc human gold. Không dùng kết quả này để chọn checkpoint hay ngưỡng. Training chạy độc lập và tiếp tục.</div>
<table><thead><tr><th>Tiếng</th><th>C75 ban đầu</th><th>Sau audit cycle84</th><th>Accuracy</th><th>F1 EOT</th><th>Accuracy / F1 trước audit</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Audit loại đầu vào 256 token của teacher trùng train/DEV cycle84, nguồn trùng hoặc bản sao nguồn thuộc non-test. Chưa kiểm hết paraphrase hoặc va chạm chỉ xảy ra với tokenizer WordPiece của student.</p>
<p>Đầu vào chỉ gồm context và text không dấu câu; không đưa nhãn, dấu câu nguồn hoặc dự đoán cũ vào model. Full traces và checkpoint giữ trên SSH.</p>
<p><a href="report.json">Report JSON</a> · <a href="frozen_dev.json">DEV dùng khóa ngưỡng</a> · <a href="checkpoint_receipt.json">Biên nhận checkpoint</a></p>
{comparison}{''.join(samples)}<p>Report SHA256: <code>{escape(bundle['report_sha256'])}</code></p></html>'''
    (out/'review.html').write_text(html,encoding='utf-8')
    receipt={'remote':remote,'report_sha256_remote':bundle['report_sha256'],'transfer_bytes':len(result.stdout.encode()),
        'local_files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()},
        'models_or_full_traces_downloaded':False}
    (out/'transfer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'review':str(out/'review.html'),'checkpoint_step':step,'threshold':threshold,
        'languages':{l:v['metrics_after_cycle84_audit'] for l,v in report['languages'].items()},'transfer_bytes':receipt['transfer_bytes']},ensure_ascii=False))


if __name__=='__main__':main()
