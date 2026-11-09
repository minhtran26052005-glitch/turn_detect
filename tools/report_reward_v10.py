"""Report matched data/reward evidence and refresh the campaign notebook cell."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from build_notebook import cell
from colab_runtime import ColabRuntime

ROOT=Path('outputs/reward_v10_20261025')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['live','final'],default='live');args=p.parse_args()
    reports={lang:json.loads((ROOT/lang/'report.json').read_text()) for lang in ['vi','en','zh'] if (ROOT/lang/'report.json').exists()}
    data=json.loads(Path('data_v10/reward_20261025/manifest.json').read_text());qwen=json.loads((ROOT/'qwen/report.json').read_text()) if (ROOT/'qwen/report.json').exists() else {}
    summary={'status':'DEV_ABLATION_COMPLETE' if len(reports)==3 else 'RUNNING','quality_target_certified':False,'language_models_separate':True,'reports':reports,'qwen':qwen,
        'local_policy':'10GiB reserve; no full teacher download; compact student artifacts only','crawl':'Training candidate pool; low-weight two-view silver; never independent gold',
        'RL_scope':'Sampled one-step REINFORCE from binary automatic labels. No environment interaction, measured latency or PPO.',
        'fresh_test_evaluated':False,'old384_regression_reused_for_selection':False}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for lang,report in reports.items():
        for phase,record in [('baseline',report['baseline'])]+[(r['phase'],r['dev']) for r in report['phases']]:
            m=record['slices']['natural'];rows.append({'language':lang,'phase':phase,'accuracy':m['accuracy'],'eot_f1':m['eot_f1'],'false_eot_rate':m['false_eot_rate'],'selection_score':record['selection_score']})
    markdown='''# Data mở rộng, crawl Qwen và thử reward — V10

## Material Passport

Experiment type: CODE_TRAINING_ABLATION. Ba model riêng. Input text-only. GPU Colab L4 23.034MiB; không tải teacher lớn về Windows. Mục tiêu >90% accuracy/F1 EOT trên hội thoại độc lập chưa được chứng nhận.

## Thiết kế khóa trước khi chạy

Mỗi ngôn ngữ khởi tạo từ đúng student QAT riêng đã có. Vocabulary và scale INT4 giữ nguyên; cập nhật latent weights có fake quant mỗi forward. Codec giữ token hiện tại mới nhất trong256token. Không có dữ liệu người nói tương lai/timing/audio trong input.

`data_sft`:450bước với dữ liệu mở rộng, balanced nguồn/nhãn. Sau đó `ce_control`, `cost_ce`, `reinforce` cùng bắt đầu từ checkpoint data_sft được chọn bằng dev,225bước/nhánh, batch256, cùng seed sinh chỉ số batch và LR. `cost_ce` tăng trọng số NEOT để kiểm tra lợi ích có chỉ đến từ việc ưu tiên tránh ngắt sớm hay không.

RL là one-step REINFORCE: hành động wait/stop, đúng+1, stop trên NEOT−2, wait trên EOT−0,5. Baseline reward kỳ vọng được detach để giảm phương sai; thêm CE và KL tới data_sft để tránh quên. Vì có nhãn binary, reward của cả hai hành động có thể suy ra; đây không phải bằng chứng rằng RL tốt hơn phân loại theo cost. Không có simulator thời gian hoặc latency thật, nên−0,5 là sở thích cost cố định, không phải số mili giây. Crawl silver không nhận policy reward; chỉ vào CE với weight0,25. Nhãn floor tự động và prefix proxy vẫn có lỗi.

## Nguồn dữ liệu và crawl

Tận dụng thêm dialogue official train trong MultiWOZ2.2 và CrossWOZ đã tải, giữ revision/provenance. Cuối lượt và prefix có tiếp lời chỉ là nhãn cấu trúc proxy, không tự là human semantic EOT. Nhánh Việt thêm628mẫu natural train trước lọc và24.000mẫu replay train. Dữ liệu dev, toàn bộ test/holdout đã xem và204prefix cũ được bảo vệ khỏi training theo exact visible-input hash; official dialogue train/dev khác nhau. Đây chưa chứng minh không có trùng paraphrase hoặc pretraining contamination.

Crawl12bài Wikipedia mới có giới hạn, loại source của204prefix review cũ, giữ page ID/revision/URL/hash. Thu110prefix (Việt43, Anh47, Trung20); hai request Trung bị429, không lặp request dồn dập. Dấu câu chỉ chọn vị trí quan sát, không tạo nhãn.

Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554` chạy BF16, tối đa24prompt/batch và16.384token/batch. Chỉ nhận hai view cùng EOT/NEOT và min relative A/B/C score≥0,8. Hai view cùng model có tương quan, score chưa hiệu chỉnh; nhãn vẫn silver. Qwen chỉ thấy text/context hiện tại, không thấy suffix. Không dùng nhãn Qwen làm test gold.

## Phạm vi đánh giá

Dev natural Việt145/Anh79/Trung69; các slice replay/legacy báo riêng và cùng ràng buộc ngưỡng/chọn checkpoint. Tất cả nhánh được chấm lại trên cùng dev của vòng này. Các số cũ dùng codec/split khác không là đối chứng nhân quả. Không chấm lại384mẫu regression để tune; không gọi dữ liệu crawl không có nhãn độc lập là accuracy/F1 thực.

| Tiếng | Nhánh | Accuracy natural dev | F1 EOT natural dev | EOT nhầm/hold | Min dev slice score |
|---|---|---:|---:|---:|---:|
'''
    for row in rows:markdown+=f"| {row['language']} | {row['phase']} | {row['accuracy']:.2%} | {row['eot_f1']:.2%} | {row['false_eot_rate']:.2%} | {row['selection_score']:.2%} |\n"
    if qwen:markdown+='\nQwen accepted silver: '+json.dumps(qwen.get('by_language',{}),ensure_ascii=False)+'. GPU peak '+f"{qwen['CUDA_peak_bytes']/2**30:.2f}"+'GiB.\n'
    if reports:
        markdown+='\n## So sánh reward với đối chứng cùng ngân sách\n\n'
        for lang,r in reports.items():markdown+='- '+lang+': chọn dev `'+r['selected_phase']+'`; REINFORCE thắng CE theo min-slice score: '+str(r['RL_beats_matched_CE_on_selection_score'])+'; thắng cost-CE: '+str(r['RL_beats_cost_CE_on_selection_score'])+'.\n'
    markdown+='''

Kích thước/model và pipeline cũ giữ nguyên. Không tự promote nhánh chưa vượt tiêu chí. [Kết quả](outputs/reward_v10_20261025/review.html) · [Số liệu JSON](outputs/reward_v10_20261025/summary.json).

Nguồn: [MultiWOZ upstream](https://github.com/budzianowski/multiwoz/tree/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b), dữ liệu CC BY4 theo Cambridge, code MIT; [CrossWOZ upstream](https://github.com/thu-coai/CrossWOZ/tree/df82c9fdff91b9b130f2d6b89110d3870ba6260e), Apache2. [Qwen model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554), Apache2. Wikipedia CC BY-SA, URL/revision từng bài trong `data_v10/reward_20261025/crawl/sources.jsonl`.

[Contextual Bandit Algorithms with Supervised Learning Guarantees](https://proceedings.mlr.press/v15/beygelzimer11a/beygelzimer11a.pdf) mô tả partial-feedback bandit. V10 khác ở chỗ reward suy từ nhãn nên có full-feedback giả định; code không triển khai thuật toán Exp4.P hay tuyên bố các bảo đảm của bài báo áp dụng.
'''
    Path('README_REWARD_V10.md').write_text(markdown,encoding='utf-8')
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]) if k in ['language','phase'] else f'{row[k]:.2%}')+'</td>' for k in ['language','phase','accuracy','eot_f1','false_eot_rate','selection_score'])+'</tr>' for row in rows)
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Data và reward V10</title><style>body{font:17px system-ui;max-width:1200px;margin:35px auto;padding:24px;color:#18313e;background:#f3f7fb}p{line-height:1.7}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{background:white;padding:24px;border-radius:12px;margin:18px 0}.scroll{overflow:auto}</style><h1>Ba model riêng: thêm data, crawl và reward</h1><p>Đây là development ablation; chưa chứng nhận90% hội thoại thực. Nhãn tự động/proxy và silver có giới hạn. Không dùng384regression đã xem để chọn nhánh.</p><section><div class="scroll"><table><tr><th>Tiếng</th><th>Nhánh</th><th>Accuracy natural dev</th><th>F1 EOT natural dev</th><th>EOT nhầm/hold</th><th>Min slice score</th></tr>'+body+'</table></div></section><section><h2>Reward và đối chứng</h2><p>REINFORCE một quyết định: đúng+1, ngắt sớm−2, chờ trênEOT−0,5. CE, cost-CE và reward cùng bắt đầu từ data_sft với cùng ngân sách. Nhãn crawl chỉ dùng CE trọng số thấp, không tạo reward.</p><pre>'+html.escape(json.dumps({lang:{'selected_phase':r['selected_phase'],'RL_beats_CE':r['RL_beats_matched_CE_on_selection_score'],'RL_beats_cost_CE':r['RL_beats_cost_CE_on_selection_score']} for lang,r in reports.items()},indent=2))+'</pre></section><section><h2>Crawl Qwen</h2><pre>'+html.escape(json.dumps(qwen,ensure_ascii=False,indent=2))+'</pre><p><a href="summary.json">JSON số liệu</a> · <a href="../../README_REWARD_V10.md">Recipe và phạm vi</a></p></section></html>'
    (ROOT/'review.html').write_text(page,encoding='utf-8')
    notebook=Path('a.ipynb');nb=json.loads(notebook.read_text(encoding='utf-8'));tag='reward_v10_status'
    source=Path('tools/templates/reward_v10_status_cell.py').read_text(encoding='utf-8');matches=[c for c in nb['cells'] if c['cell_type']=='code' and tag in c.get('metadata',{}).get('tags',[])]
    if not matches:
        shutil.copy2(notebook,ROOT/'a_before_v10.ipynb');nb['cells'].extend([cell('markdown','''
        ## V10 — data mở rộng, crawl Qwen, đối chứng reward

        Ba model riêng, QAT INT4. So sánh data_sft/CE/cost-CE/REINFORCE trên cùng dev. Nhãn tự động/silver, chưa có chứng nhận90% thực. [Báo cáo](outputs/reward_v10_20261025/review.html) · [Recipe](README_REWARD_V10.md).
        ''',tag),cell('code',source,tag)]);matches=[nb['cells'][-1]]
    current=matches[0];assert ''.join(current['source'])==source
    r=ColabRuntime(0);connection=r.connect();result=r.execute(source,timeout=90,echo=False);r.close();destination=ROOT/('notebook_'+args.phase+'_receipt.json');assert not destination.exists()
    destination.write_text(json.dumps({'connection':connection,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result},ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['status']=='ok';current['outputs']=result['outputs'];current['metadata']['verification_receipt']=str(destination);notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'phase':args.phase,'completed_languages':list(reports),'notebook_execution':result['status'],'outputs':len(result['outputs'])}))


if __name__=='__main__':main()
