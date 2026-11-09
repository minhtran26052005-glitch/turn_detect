"""Readable results, streaming replay, and an additive notebook result cell."""
from collections import Counter
import html
import json
from pathlib import Path
import re
import shutil

from build_notebook import cell

ROOT = Path('outputs/stream_v4_20261025')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(values, markdown=True):
    if markdown:
        lines = ['| Ngôn ngữ | Mẫu | Accuracy | F1 EOT | EOT nhầm | Bỏ lỡ EOT |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for lang, value in values.items():
            lines.append(f"| {lang} | {value['samples']} | {value['accuracy']:.2%} | {value['eot_f1']:.2%} | {value['false_eot']} | {value['missed_eot']} |")
        return '\n'.join(lines)
    rows = ''.join(f"<tr><td>{lang}</td><td>{m['samples']}</td><td>{m['accuracy']:.2%}</td><td>{m['eot_f1']:.2%}</td><td>{m['false_eot']}</td><td>{m['missed_eot']}</td></tr>" for lang, m in values.items())
    return '<table><tr><th>Tiếng</th><th>Mẫu</th><th>Accuracy</th><th>F1 EOT</th><th>EOT nhầm</th><th>Bỏ lỡ EOT</th></tr>'+rows+'</table>'


def main():
    final = load(ROOT/'colab_final/report.json')
    campaign = load(ROOT/'runs/stream_v4_campaign_20261025/campaign.json')
    followup = load(ROOT/'runs/stream_v4_balanced_20261025/report.json')
    local = load(ROOT/'local_final/report.json')
    natural = {**final['metrics']['fresh_vi_natural_auto_labels'], **final['metrics']['fresh_en_zh_natural_auto_labels']}
    natural = {lang: natural[lang] for lang in ['vi', 'en', 'zh']}
    baseline = final['metrics']['original_teacher_matched_fresh_auto_labels']
    checkpoint = final['frozen']['selected_path']
    selected = load(ROOT/Path(checkpoint).parent/'best_dev.json')
    summary = {'status':'TRAINED_AND_EVALUATED_REAL_TARGET_NOT_ACHIEVED', 'user_target_achieved':False,
        'colab_initial_optimizer_steps':1800, 'colab_followup_optimizer_steps':300, 'local_optimizer_steps':600,
        'selected_checkpoint':checkpoint,'selected_round':selected['round'],'selected_step':selected['step'],
        'weights_sha256':final['frozen']['weights_sha256'],'weights_bytes':1115268200,
        'thresholds':final['frozen']['thresholds'],'fresh_natural_auto_label_metrics':natural,
        'original_teacher_matched_fresh':baseline,'full_legacy_vi_dev':final['metrics']['vi_full_legacy_dev'],
        'dev_slices':final['dev_slices'],'held_read_speech':final['metrics']['sealed_read_speech'],
        'followup_stop_reason':followup['stop_reason'],'fresh_crawl_cases':204,'fresh_crawl_gold_labels':0,
        'fresh_crawl_accuracy':None,'fresh_crawl_eot_f1':None,'accepted_serving_default_changed':False,
        'compression_of_v4_teacher_performed':False,'human_gold_available':False,
        'fresh_natural_conversation_counts':{'vi':2,'en':1,'zh':1},
        'test_status_after_this_run':'SEEN; retained as regression, do not tune on these pilot scores',
        'next_training_architecture':'TEXT_ONLY; user selected more public natural conversation training data',
        'local_branch_report':'local_final/report.json','colab_report':'colab_final/report.json'}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md = f'''# Streaming V4 — train và kiểm thử 25/10/2026

**Chưa đạt accuracy và F1 EOT >90% đồng thời ở cả ba ngôn ngữ.** Colab L4 đã chạy xong 3 vòng ×600 bước và 300 bước cân bằng nhãn; nhánh local chạy thêm 3 vòng ×200 bước. Vòng cân bằng dừng do ba lần dev liên tiếp không cải thiện. Checkpoint chọn bằng dev là vòng {selected['round']}, bước {selected['step']}; không chọn lại sau khi xem test.

**Pilot hội thoại mới, nhãn tự động từ audio**

{table(natural)}

Việt: 30 khoảng dừng từ 2 cuộc gọi DataHive. Anh/Trung: mỗi tiếng 20 khoảng dừng từ 10 turn CallHome thuộc official-test, một hold và một EOT mỗi turn; các turn được lấy đầu file nên chỉ thuộc **1 cuộc gọi mỗi tiếng**. ASR chỉ nghe tối đa15 giây trước thời điểm quyết định `silence.start+0,2s`. Không đưa tiếng nói/phần chữ tương lai vào model. Nhãn do VAD và hành vi đổi người nói suy ra, chưa được người chấm xác nhận; mẫu nhỏ và các điểm trong cùng cuộc gọi không độc lập. Đây là chẩn đoán ban đầu, chưa là benchmark gold đại diện.

**Teacher gốc trên đúng các mẫu mới và cùng cách mã hóa causal**

{table(baseline)}

Ngưỡng của mỗi model được chọn riêng từ dev và khóa trước pilot. Sau train, Việt tốt hơn đối chứng trên pilot nhỏ, Anh/Trung thấp hơn; không có bằng chứng để thay model triển khai hoặc tuyên bố đạt mục tiêu. Khác nguồn và nhãn giữa dev cũ với hội thoại tự nhiên là vấn đề cần giải quyết; chưa thể quy toàn bộ lỗi cho một nguyên nhân từ 70 mẫu tự động.

**Dev Việt cũ đủ17.833 mẫu**

{table(final['metrics']['vi_full_legacy_dev'])}

Dev này là nhãn prefix suy ra. Điểm >90% ở đây không đồng nghĩa đã >90% trên hội thoại thật. Bản Việt INT4 đang có đạt accuracy46,67% / F1 EOT38,46% trên cùng30mẫu Việt mới; con số này cũng chỉ là pilot tự động. Nhánh local mới đạt56,67% /60,61% trên30mẫu đó và chưa được chọn làm bản triển khai.

**Dữ liệu và cách train**

Train104.143 mẫu, dev5.992 mẫu; cân bằng mức xuất hiện ba tiếng. Giữ riêng1.855 mẫu EN/ZH đọc sẵn trước lượt kiểm tra cuối. LiveKit đã được xem ở V3 nên từ V4 là train/dev, chia theo toàn cuộc hội thoại; không gọi là test độc lập. Việt rehearsal lấy từ frozen train/dev cũ, không đọc lại holdout đã dùng. Loại toàn câu hiện tại khỏi `messages` của Scicom đọc sẵn để tránh lộ phần lời tương lai. Đầu vào dài giữ phần mới nhất của current text, tối đa256tokens gồm ngữ cảnh và special tokens; dấu câu được chuẩn hóa khỏi đầu vào.

Colab: full teacher mDeBERTa12 lớp, FP32/TF32 tắt, AdamW/warmup/cosine/gradient checkpointing. Ba vòng LR1e-5,5e-6,2e-6; microbatch16 ×2. Vòng cân bằng lấy EOT/NEOT đều trong từng nguồn, LR2e-6; checkpoint trước vẫn là đối chứng hợp lệ. Local RTX3050: chỉ học3lớp cuối/pooler/head, microbatch4 ×2, lưu delta kèm hash base.

Tập đọc sẵn tiếng Trung bị lệch nhãn mạnh: lượt held có498EOT,36NEOT. Accuracy93,07% / F196,38% vẫn đi cùng31/36hold bị kết thúc nhầm; không dùng điểm này để tuyên bố hội thoại Trung đã tốt.

**Crawl streaming**

204 prefix tại dấu phẩy/chấm từ11bài mới, đủ Việt/Anh/Trung, đã dự đoán và giữ từng xác suất. Cả204mẫu vẫn `label:null`, vì dấu câu không phải nhãn nhường lượt. Accuracy/F1 crawl hiện là `null`. [Player và bảng kết quả](outputs/stream_v4_20261025/review.html), [chấm semantic-completeness độc lập](outputs/stream_v4_20261025/fresh_crawl_review.html), [nghe30điểm dừng Việt](outputs/stream_v4_20261025/vi_natural_review.html).

**Nguồn và giấy phép**

- [Scicom semantic-vad-eot](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/tree/98d1445eab71e4084ccb838dbd87a205f6f2e293): read speech CC BY4.0. Subset CallHome hội thoại giữ giấy phép nguồn CC BY-NC-SA4.0, chỉ dùng kiểm thử nghiên cứu trong vòng này.
- [DataHive conversations](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/tree/1120cd7533b16ba8dd7a7c07e6d316b74b60f72a): CC BY4.0, attribution DataHive, không kèm transcript.
- [LiveKit EOT data](https://huggingface.co/datasets/livekit/eot-bench-data/tree/ca9d98a9686b920a2d8c9eb984224ba9be74e4dd): dữ liệu CC BY4.0; giấy phép Apache2.0 của harness không phải giấy phép dataset. Sửa nhầm lẫn trong README V3.
- [Whisper small CTranslate2](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120): snapshot ASR được khóa trong pilot EN/ZH. Nguồn Wikipedia, URL và hash từng bài nằm trong `data_v4/fresh_crawl_20261025/sources.jsonl`.

**File đã lưu và cách kiểm tra**

Teacher FP32: `outputs/stream_v4_20261025/{checkpoint}/model.safetensors` (1.115.268.200byte), SHA256 `{summary['weights_sha256']}`. Đây là teacher lớn; chưa xuất một bản distill/INT4 V4 đạt tiêu chuẩn. Bản Việt QAT mặc định và các checkpoint cũ giữ nguyên.

`outputs/stream_v4_20261025/summary.json` chứa bảng tổng; `colab_final/report.json` và NPZ/JSONL chứa kết quả tính; `completion_validation.json` kiểm tra kỹ thuật. Đã chạy45unit tests pass. Notebook `a.ipynb` có ô `stream_v4_result` cuối file để xác minh hash và đọc đúng kết quả này. Nguồn và hash dữ liệu, source launch/final, receipts, audio và dự đoán được giữ lại để kiểm tra.

Mốc >90% vẫn chưa đạt. Các test vừa xem chuyển thành regression; vòng sau cần nguồn hội thoại train bổ sung và một cohort kiểm tra mới, hoặc thay đầu vào theo lựa chọn text-only/âm thanh. Không lặp tối ưu trên chính70mẫu này để biến chúng thành điểm chấp nhận.
'''
    Path('README_STREAM_V4.md').write_text(md,encoding='utf-8')
    passport = '# Hồ sơ thí nghiệm streaming V4\n\nMaterial Passport: **TRAINED_AND_EVALUATED_REAL_TARGET_NOT_ACHIEVED**.\n\n'+md.split('\n',1)[1]
    Path('EXPERIMENT_STREAM_V4_20261025.md').write_text(passport,encoding='utf-8')
    rows = [json.loads(line) for line in (ROOT/'colab_final/fresh_crawl_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0,str(Path.cwd()))
    from turn_v2.common import normalize
    tokenizer=AutoTokenizer.from_pretrained(ROOT/checkpoint,local_files_only=True)
    for row in rows:
        tokens=tokenizer.encode(normalize(row['text']),add_special_tokens=False)
        row['current_tokens']=len(tokens);row['current_truncated']=len(tokens)>253
        row['current_model_text']=tokenizer.decode(tokens[-253:])
    data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    content = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kết quả streaming V4</title><style>body{font:17px system-ui;color:#18313e;background:#f3f7fb;margin:0}main{max-width:1100px;margin:auto;padding:28px}section{background:white;padding:24px;border:1px solid #c8d9e4;border-radius:12px;margin:18px 0}h1{font-size:29px}p{line-height:1.65}button,select{font:inherit;padding:9px;border:1px solid #abc2d0;border-radius:6px;background:white;margin:4px;cursor:pointer}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #dce6ee}#text{font-size:22px;white-space:pre-wrap;line-height:1.8}input[type=range]{width:100%}a{color:#126b92}details{margin:14px 0}.bad{color:#9c3b22;font-weight:bold}.note{color:#496677}pre{white-space:pre-wrap;word-break:break-all}</style>
<main><h1>Streaming V4: chưa đạt >90% cả ba tiếng</h1><p>Đã train trên Colab L4 và giữ checkpoint được chọn trên dev. Pilot hội thoại mới có nhãn suy ra bằng audio/VAD, chưa được người chấm xác nhận.</p><section><h2>Pilot hội thoại mới</h2>__NATURAL__<p>Việt30mẫu từ2cuộc gọi; Anh/Trung20mẫu mỗi tiếng từ1cuộc gọi mỗi tiếng. Mẫu nhỏ, nhiều điểm cùng cuộc gọi; đây chưa phải ước lượng chất lượng trên mọi hội thoại.</p><details><summary>Teacher gốc trên cùng mẫu/cách mã hóa</summary>__BASELINE__</details><p><a href="vi_natural_review.html">Nghe và kiểm tra30mẫu Việt</a> · <a href="colab_final/report.json">JSON kết quả gốc</a> · <a href="../../README_STREAM_V4.md">Đọc báo cáo</a></p></section>
<section><h2>204 điểm crawl tại dấu câu</h2><p class="note">Chưa có nhãn độc lập, nên accuracy và F1 crawl chưa được tính. Dấu chấm/phẩy chỉ là điểm gọi model. Thanh trượt giữ thứ tự prefix; tự động phát sẽ luôn chạy hết bài, kể cả khi model phát EOT sớm.</p><select id="lang"><option value="vi">Tiếng Việt</option><option value="en">English</option><option value="zh">中文</option></select><select id="article"></select><button id="prev">←</button><button id="next">→</button><button id="play">Phát</button><input id="position" type="range" min="0" value="0"><p id="meta"></p><div id="text"></div><p id="result"></p><details><summary>Phần chữ chuẩn hóa được model quan sát</summary><p id="tail"></p></details><p><a id="source">Nguồn bài</a> · <a href="fresh_crawl_review.html">Chấm mẫu độc lập</a></p></section>
<section><h2>Phạm vi kết luận</h2><p>Dev Việt cũ F1 khoảng90,09% không xác nhận tiếng Việt hội thoại đã đạt90%. Vòng cân bằng nhãn dừng sau ba lần dev không cải thiện. Các pilot đã xem được giữ làm regression; không chỉnh ngưỡng theo các kết quả này.</p><p id="identity"></p><p><a href="completion_validation.json">Kiểm tra kỹ thuật</a> · <a href="summary.json">Bảng số liệu</a></p></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const R=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let rows=[],i=0,timer=null;
function pause(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Phát'}
function show(){const r=rows[i];if(!r)return;$('position').value=i;$('meta').textContent=(i+1)+'/'+rows.length+' · dấu '+r.punctuation+' · '+r.current_tokens+' tokens'+(r.current_truncated?' · dùng phần mới nhất':'');$('text').textContent=r.text;$('result').textContent=r.prediction+' · p(EOT)='+r.prob_EOT.toFixed(4)+' · ngưỡng='+r.threshold.toFixed(3)+' · chưa có nhãn';$('result').className=r.prediction==='EOT'?'bad':'';$('tail').textContent=r.current_model_text;$('source').href=r.source_url}
function article(){pause();rows=R.filter(r=>r.language===$('lang').value&&r.source_id===$('article').value).sort((a,b)=>a.text.length-b.text.length);i=0;$('position').max=rows.length-1;show()}
function language(){pause();$('article').replaceChildren();const seen=new Set();R.filter(r=>r.language===$('lang').value).forEach(r=>{if(!seen.has(r.source_id)){seen.add(r.source_id);const o=document.createElement('option');o.value=r.source_id;o.textContent=r.source_title;$('article').append(o)}});article()}
$('lang').onchange=language;$('article').onchange=article;$('position').oninput=()=>{pause();i=Number($('position').value);show()};$('prev').onclick=()=>{pause();i=Math.max(0,i-1);show()};$('next').onclick=()=>{pause();i=Math.min(rows.length-1,i+1);show()};$('play').onclick=()=>{if(timer){pause();return}$('play').textContent='Dừng';timer=setInterval(()=>{if(i===rows.length-1){pause();return}i++;show()},700)};$('identity').textContent=__IDENTITY__;language();
</script></html>'''
    content=content.replace('__NATURAL__',table(natural,False)).replace('__BASELINE__',table(baseline,False)).replace('__DATA__',data)
    content=content.replace('__IDENTITY__',json.dumps('Teacher1.115.268.200byte · SHA256 '+summary['weights_sha256']))
    (ROOT/'review.html').write_text(content,encoding='utf-8')
    notebook=Path('a.ipynb');nb=load(notebook)
    if not any('stream_v4_result' in c.get('metadata',{}).get('tags',[]) for c in nb['cells']):
        backup=ROOT/'a_before_stream_v4.ipynb'
        if not backup.exists():shutil.copy2(notebook,backup)
        code='''
        import hashlib, json
        from pathlib import Path
        V4_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
        v4_summary = json.loads((V4_ROOT/'outputs/stream_v4_20261025/summary.json').read_text(encoding='utf-8'))
        v4_weights = V4_ROOT/'outputs/stream_v4_20261025'/v4_summary['selected_checkpoint']/'model.safetensors'
        if not v4_weights.exists():
            v4_weights = V4_ROOT/v4_summary['selected_checkpoint']/'model.safetensors'
        with v4_weights.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == v4_summary['weights_sha256']
        assert not v4_summary['user_target_achieved']
        print(json.dumps({'hash_verified':True, 'status':v4_summary['status'],
            'fresh_natural_AUTO_labels':v4_summary['fresh_natural_auto_label_metrics'],
            'fresh_crawl_gold_labels':v4_summary['fresh_crawl_gold_labels'],
            'fresh_crawl_accuracy':None, 'serving_default_changed':False},ensure_ascii=False,indent=2))
        '''
        nb['cells'].extend([cell('markdown','''
        ## Streaming V4: đã train/test lại, chưa đạt >90% cả ba tiếng

        Colab3×600bước +300bước cân bằng nhãn; local3×200bước. Checkpoint chọn ở vòng2,bước200, đã backup đầy đủ. Pilot audio mới70mẫu: Việt accuracy50,0%/F157,1%; Anh70,0%/66,7%; Trung55,0%/66,7%. Nhãn VAD tự động, không gọi là human gold.204prefix crawl mới chưa có nhãn nên accuracy/F1=null.

        [Báo cáo và player](outputs/stream_v4_20261025/review.html) · [Nghe30điểm Việt](outputs/stream_v4_20261025/vi_natural_review.html) · [Hướng dẫn](README_STREAM_V4.md). Ô dưới chỉ kiểm tra hash/đọc kết quả đã lưu; không chạy lại test hoặc thay model mặc định.
        ''','stream_v4_result'),cell('code',code,'stream_v4_result')])
        notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    result_cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'stream_v4_result' in c.get('metadata',{}).get('tags',[]))
    (ROOT/'notebook_result_cell.py').write_text(''.join(result_cell['source']),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'notebook_cells':len(nb['cells']),'fresh_natural':natural},ensure_ascii=True))


if __name__=='__main__':
    main()
