"""Update a small status dashboard and append-only Colab notebook receipt."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
from build_notebook import cell
from colab_runtime import ColabRuntime

ROOT=Path('outputs/continuous_v13_20261026')


def main():
    p=argparse.ArgumentParser();p.add_argument('--notebook',action='store_true');a=p.parse_args()
    receipt=json.loads((ROOT/'status_receipt.json').read_text(encoding='utf-8'));value=json.loads(''.join(o['text'] for o in receipt['outputs'] if o['output_type']=='stream'))
    s=value.get('status.json',{});rows=[]
    for lang,r in value.get('champions',{}).items():
        semantic=r['semantic_dev']['metrics'];floor=r['natural_floor_dev'];rows.append({'language':lang,'semantic_samples':semantic['samples'],'semantic_accuracy':semantic['accuracy'],'semantic_eot_f1':semantic['eot_f1'],
            'floor_samples':floor['samples'],'floor_accuracy':floor['accuracy'],'floor_eot_f1':floor['eot_f1'],'path':r['path'],'precision':r.get('precision','FP32_LATENT')})
    summary={'status':s.get('status'),'stage':s.get('stage'),'cycle':value.get('cycle'),'process_verified':value.get('process_verified'),
        'queried_at_utc':datetime.now(timezone.utc).isoformat(),'worker_heartbeat_utc':s.get('heartbeat_utc'),'GPU':value.get('GPU'),'data_counts':value.get('pool_counts',[]),
        'reference_disagreements':value.get('reference_disagreements',[]),'champions':rows,'snapshot':value.get('recovery/latest_snapshot.json'),'quality_target_certified':False,
        'automation_id':'theo-d-i-v-ti-p-t-c-turn-detector-tr-n-colab','continuous':True,'max_cycles':0,'checkpoint_policy':'Compact snapshots, Drive backup by15minute monitor; no full Qwen download to Windows'}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    tr=''.join('<tr><td>'+r['language']+'</td><td>'+str(r['semantic_samples'])+'</td><td>'+f"{r['semantic_accuracy']:.2%} / {r['semantic_eot_f1']:.2%}"+'</td><td>'+str(r['floor_samples'])+'</td><td>'+f"{r['floor_accuracy']:.2%} / {r['floor_eot_f1']:.2%}"+'</td></tr>' for r in rows)
    escaped=lambda v:html.escape(json.dumps(v,ensure_ascii=False,indent=2))
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Colab continuous turn detection</title><style>body{font:18px system-ui;max-width:1200px;margin:32px auto;padding:24px;color:#18313e;background:#f3f7fb}section{background:white;padding:24px;border-radius:12px;margin:18px 0}p{line-height:1.7}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto}</style><h1>Crawl và train liên tục trên Colab</h1><p>Không giới hạn số vòng, không dừng vì plateau hoặc silver90%. Ba model riêng. Dữ liệu và training nằm trên Colab; bản sao khôi phục nhỏ được theo dõi đưa lên Drive.</p><section><h2>Trạng thái lần kiểm tra gần nhất</h2><pre>'+escaped({k:summary[k] for k in ['queried_at_utc','status','stage','cycle','process_verified','GPU','worker_heartbeat_utc']})+'</pre><p>Đây là snapshot lần kiểm tra, lịch theo dõi15phút sẽ cập nhật khi app có thể chạy. Ô cuối notebook đọc trạng thái kernel hiện tại. Trang tự tải lại không phải keepalive Colab.</p></section><section><h2>Hai mục tiêu được báo riêng</h2><div class="scroll"><table><tr><th>Tiếng</th><th>Mẫu semantic silver dev</th><th>Accuracy/F1 semantic</th><th>Mẫu natural floor dev</th><th>Accuracy/F1 floor</th></tr>'+tr+'</table></div><p>Semantic là đủ ý để dừng, nhãn Qwen tự động. Floor là người nói thực sự nhường lượt theo VAD. Không lấy điểm semantic thay cho chứng nhận hội thoại thật; chưa có human gold. Candidate học FP32 latent, chưa là packedINT4 được nghiệm thu.</p></section><section><h2>Data tích lũy</h2><pre>'+escaped(summary['data_counts'])+'</pre><h2>Khác biệt hai nhãn trên cùng text</h2><pre>'+escaped(summary['reference_disagreements'])+'</pre></section><section><h2>Checkpoint</h2><pre>'+escaped(summary['snapshot'])+'</pre><p><a href="summary.json">JSON trạng thái</a> · <a href="../../README_CONTINUOUS.md">Cách chạy và khôi phục</a> · <a href="../auto_rl_v12_20261025/review.html">Vòng thử cũ</a></p></section></html>'
    (ROOT/'review.html').write_text(page,encoding='utf-8')
    if a.notebook:
        path=Path('a.ipynb');nb=json.loads(path.read_text(encoding='utf-8'));tag='continuous_v13_status';source=Path('tools/templates/continuous_v13_status_cell.py').read_text(encoding='utf-8')
        found=[c for c in nb['cells'] if c['cell_type']=='code' and tag in c.get('metadata',{}).get('tags',[])]
        if not found:
            shutil.copy2(path,ROOT/'a_before_continuous.ipynb');nb['cells'].extend([cell('markdown','''
            ## Colab — crawl và train liên tục

            Chạy đến khi người dùng yêu cầu dừng, không giới hạn2vòng. SFT/RL từ checkpoint tốt nhất của từng tiếng, data tích lũy, optimizer/RNG có thể resume. Semantic silver dev và natural-floor dev báo riêng. [Dashboard](outputs/continuous_v13_20261026/review.html) · [Cách khôi phục](README_CONTINUOUS.md).
            ''',tag),cell('code',source,tag)]);found=[nb['cells'][-1]]
        current=found[0];assert ''.join(current['source'])==source
        runtime=ColabRuntime(0);connection=runtime.connect();result=runtime.execute(source,timeout=90,echo=False);runtime.close();assert result['status']=='ok'
        current['outputs']=result['outputs'];current['metadata']['verified_utc']=datetime.now(timezone.utc).isoformat()
        proof={'connection':connection,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result};destination=ROOT/('notebook_receipt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
        destination.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');current['metadata']['verification_receipt']=str(destination)
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'stage':summary['stage'],'process_verified':summary['process_verified'],'champions':len(rows),'notebook_refreshed':a.notebook}),flush=True)


if __name__=='__main__':main()
