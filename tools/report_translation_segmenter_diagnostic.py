"""Interpret compact frozen-test evidence; no model execution or raw-corpus I/O."""
from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path

JOB=Path('outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1')


def read(name):return json.loads((JOB/name).read_text(encoding='utf-8'))


def balanced_accuracy(metrics):
    (tn,fp),(fn,tp)=metrics['confusion_matrix_NEOT_EOT']
    return .5*(tn/max(tn+fp,1)+tp/max(tp+fn,1))


def main():
    report=read('report.json');analysis=read('error_analysis_v1.json');parser=read('parser_audit_v1.json')
    if not all(x.get('complete') for x in (report,analysis,parser)):
        raise RuntimeError('Complete scored/control/parser evidence is required')
    sha=hashlib.sha256((JOB/'report.json').read_bytes()).hexdigest()
    if analysis['diagnostic_report_sha256']!=sha:raise RuntimeError('Analysis points to another diagnostic report')
    lines=['# Kết quả test chẩn đoán cycle75 và hướng crawl', '', '## Material Passport', '',
        '- Dữ liệu: public prose, nhãn từ dấu câu; input chữ thường bỏ dấu câu, lookahead1/2 từ.',
        '- Model: ba checkpoint cố định cycle75/100, mỗi model1.100 bước optimizer; ngưỡng chỉ chọn trên dev.',
        '- Test: tài liệu thêm sau45 đến75,80% trang dùng chẩn đoán;20% trang giữ riêng chưa chấm.',
        '- Phạm vi: phân tích khám phá để thiết kế dữ liệu train; không phải chứng nhận hội thoại/audio90%.',
        '- Verification: scored records, frozen hashes, overlap audit and original-source alignment inspected; no performance rerun or causal intervention.',
        '', '## Kết quả chấm trên dữ liệu test mới', '',
        '| Ngôn ngữ | Mẫu sạch | Accuracy | EOT F1 | Balanced accuracy |95% CI accuracy theo trang|',
        '|---|---:|---:|---:|---:|---|']
    total=0;facts={};priorities={}
    for lang,result in report['languages'].items():
        m=result['metrics'];interval=result['source_cluster_95pct_intervals']['accuracy'];total+=m['samples']
        lines.append(f'|{lang}|{m["samples"]:,}|{m["accuracy"]:.2%}|{m["eot_f1"]:.2%}|{balanced_accuracy(m):.2%}|{interval[0]:.2%}–{interval[1]:.2%}|')
    lines+=['',f'Tổng{total:,} mẫu sạch. Không so trực tiếp với test cycle45 vì model và thành phần dữ liệu đều khác. Bộ cycle45 không được chấm lại.',
        '', '## Phần sai và bằng chứng', '']
    for lang,result in report['languages'].items():
        m=result['metrics'];errors=result['error_analysis'];fit=analysis['train_fit_controls'][lang]['metrics'];coverage=analysis['training_coverage'][lang]
        pc=parser['languages'][lang]['counts'];top=errors['slices']['punctuation'][:5]
        total_error=m['false_eot']+m['missed_eot']
        lines += [f'### {lang}', '',
            f'- Cắt sớm false EOT:{m["false_eot"]:,}; bỏ sót cuối câu:{m["missed_eot"]:,}; tổng sai:{total_error:,}.',
            f'- Recall cuối dấu chấm:{result["terminal_period_eot_recall"]:.2%}; giữ NEOT tại dấu câu nội bộ:{result["internal_punctuation_no_eot_rate"]:.2%}.',
            f'- Nhóm dấu câu góp nhiều lỗi nhất: '+ '; '.join(f'{repr(x["feature"])}:{x["errors"]:,}/{x["samples"]:,} ({x["error_rate"]:.1%})' for x in top)+'.',
            f'- Lỗi lookahead không liền nguồn đã căn chỉnh:{pc.get("non_contiguous_lookahead_rows",0):,}/{pc.get("aligned_EOT_rows",0):,} mẫu EOT; chưa căn chỉnh:{pc.get("could_not_align_to_legacy_extraction",0):,}.',
            f'- Train-pool control2.000 mẫu: accuracy{fit["accuracy"]:.2%}, F1{fit["eot_f1"]:.2%}, balanced accuracy{balanced_accuracy(fit):.2%}. Test balanced accuracy{balanced_accuracy(m):.2%}. Đây là các mẫu thuộc pool train, không chắc đã được optimizer lấy trúng.',
            f'- Volume học:70.400 lượt lấy mẫu /{coverage["rows"]:,} mẫu hiện có; số mẫu khác nhau đã thấy không thể vượt{coverage["upper_bound_unique_current_rows_seen_fraction"]:.2%} pool hiện tại. Lượt lặp và pool cũ khiến độ phủ thực còn thấp hơn cận trên này.', '']
        if pc.get('non_contiguous_lookahead_rows'):
            lines.append('Lỗi trích xuất đã xác nhận bằng nguồn: câu ngắn bị loại trước khi ghép cặp, làm từ tương lai nhảy qua nội dung thật. Cần sửa dữ liệu mới và đánh giá lại trên dev sạch; chưa có thí nghiệm chứng minh mức tăng điểm sau sửa.')
        gap=balanced_accuracy(fit)-balanced_accuracy(m)
        lines.append('Train-pool control và test cùng thấp hoặc gần nhau phù hợp với giả thuyết tối ưu còn ít/nhãn khó; chưa đủ chứng minh chỉ thiếu dữ liệu.' if balanced_accuracy(fit)<.90 and gap<.05 else 'Khoảng cách control/test cần được đọc cùng khác biệt nguồn, nhãn và độ dài; không tự quy toàn bộ lỗi cho thiếu crawl.')
        lines += ['', '|Nguồn|Mẫu|Sai|Tỷ lệ sai|','|---|---:|---:|---:|']
        for x in errors['slices']['source']:
            lines.append(f'|{x["feature"]}|{x["samples"]:,}|{x["errors"]:,}|{x["error_rate"]:.2%}|')
        lines += ['', 'Lookahead1/2 trên cùng ranh giới: '+json.dumps(errors['paired_lookahead'],ensure_ascii=False)+'.',
            'Va chạm nhãn trên cùng text nhìn thấy: '+json.dumps(errors['normalized_visible_input_label_conflicts'],ensure_ascii=False)+'.', '']
        facts[lang]={'accuracy':m['accuracy'],'f1':m['eot_f1'],'samples':m['samples'],
            'balanced_accuracy':balanced_accuracy(m),'false_eot':m['false_eot'],'missed_eot':m['missed_eot'],
            'train_pool_control_balanced_accuracy':balanced_accuracy(fit),'lookahead_defect':pc,
            'training_exposure':coverage,'one_pool_size_draw_volume_steps':math.ceil(coverage['rows']/64)}
        priorities[lang]={'punctuation_slices_by_error_contribution':top,
            'source_slices':errors['slices']['source'],
            'training_pool_coverage_upper_bound':coverage['upper_bound_unique_current_rows_seen_fraction'],
            'same_pool_size_draw_volume_steps':math.ceil(coverage['rows']/64),
            'step_count_is_not_unique_coverage_or_quality_guarantee':True}
    balance=read('crawl_training_balance_v1.json')
    lines+=['## Tốc độ thêm dữ liệu so với lượng học', '',
        '|Ngôn ngữ|Mẫu TRAIN mới cycle73–75|Lượt lấy mẫu để học trong100 bước|', '|---|---:|---:|']
    for value in balance['actual_new_rows_by_split']:
        if value['split']=='train':
            lines.append(f'|{value["language"]}|{value["new_rows"]:,}|6.400|')
            priorities[value['language']]['actual_new_train_rows_per_three_crawls']=value['new_rows']
            priorities[value['language']]['steps_for_same_new_row_draw_volume']=math.ceil(value['new_rows']/64)
    lines+=['', 'Pool tăng nhanh hơn lượng lấy mẫu học ở cả3 ngôn ngữ trong giai đoạn này. Mọi lượt lấy mẫu có thể lặp và lấy từ toàn pool, nên đây chỉ là so sánh volume, không phải số mẫu mới đã học.', '',
        '## Kế hoạch crawl và train có thể kiểm chứng', '',
        '1. Dùng extractor phiên bản mới giữ nguyên chuỗi nguồn: câu ngắn và đoạn cuối chưa có dấu chấm vẫn cung cấp đúng1/2 từ tương lai. Tách phiên bản dữ liệu; không ghi đè corpus/checkpoint đang chạy.',
        '2. Với tài liệu mới, lấy cặp EOT và NEOT theo các nhóm dấu câu có nhiều lỗi bên trên, cân bằng riêng lookahead1/2; giữ đủ câu ngắn, liệt kê, mệnh đề nối và nội dung có số.',
        '3. Lấy tài liệu TRAIN mới từ nguồn công khai đúng ngôn ngữ, kiểm tra SHA nguồn và actual256-token input với dev/test/reserve. Không đưa câu, URL hoặc nhãn từ diagnostic/test trở lại train; BAAI EN tiếp tục cách ly.',
        '4. Tăng mức học/kiểm độ phủ trên pool đã có trước khi tăng tốc crawl thô.70.400 lượt lấy mẫu không phải đã học hết hàng trăm nghìn mẫu. Lượng draw bằng một pool cũng không bảo đảm nhìn hết khi lấy có hoàn lại.',
        '5. Mọi lựa chọn model/ngưỡng/cấu hình mới dùng dev. Sau phát triển, chỉ mở bộ xác nhận mới một lần; giữ reserve hiện tại chưa chấm và loại trùng lại. Muốn kết luận hội thoại thật cần human/audio gold độc lập.',
        '', '## Kiểm tra cách diễn giải thống kê', '',
        '11/11 hạng mục đã xem: Simpson (tách nguồn/nhãn/lookahead); ecological (không suy sang hội thoại cá nhân); Berkson và collider (nêu rõ bộ lọc lấy mẫu/loại trùng); base rate (confusion và balanced accuracy); regression-to-mean (không chọn từ test); survivorship (báo số bị loại); look-elsewhere và forking paths (slice khám phá, không tuyên bố p-value); correlation/causation và reverse causality (không quy tương quan thành hiệu quả sửa lỗi).',
        '', 'Ngưỡng và weights không được chọn lại theo kết quả này. Confidence interval bootstrap theo trang chỉ phản ánh tập prose chẩn đoán và quy trình lấy mẫu hiện tại. Các nhóm ít mẫu và paraphrase chưa khử hết vẫn là giới hạn.',
        '',f'Report SHA256:{sha}', '']
    (JOB/'error_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    plan={'schema':1,'evidence_report_sha256':sha,'scope':'TRAIN_ONLY_FUTURE_DATA_DESIGN_NOT_AUTOMATIC_TEST_TUNING',
        'status':'REVIEWABLE_RECIPE_NOT_YET_ACTIVATED_IN_LIVE_WORKER',
        'causal_extraction_module':'turn_v2.translation_segmenter_causal_extraction_v1',
        'requirements':{'lowercase_punctuation_free_inputs':True,'future_words':[1,2],
            'preserve_short_sentences_and_trailing_future':True,'new_data_recipe_identity_required':True,
            'old_test_and_reserve_never_enter_train':True,'crawl_test_urls_or_copy_test_errors':False,
            'keep_baai_english_quarantined':True,'choose_models_and_thresholds_on_dev':True,
            'colab_only_model_execution':True,'large_artifacts_destination':'mounted_google_drive'},
        'priorities':priorities,'language_results':facts,'quality_target_certified':False}
    (JOB/'crawl_plan_v1.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    samples=[json.loads(x) for x in (JOB/'review_samples.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    cases=[]
    for row in samples:
        head=f'{row["language"]} | target {row["label"]} | predicted {row["predicted"]} | lookahead {row["lookahead_words"]} | correct {row["correct"]}'
        cases.append('<details><summary>'+html.escape(head)+'</summary><p>'+html.escape(row['context']+' ▏ '+row['text'])+'</p><p>Target punctuation: '+html.escape(row.get('target_punctuation',''))+'</p><pre>'+html.escape(row.get('original_source_paragraph_for_posthoc_label_audit',''))+'</pre></details>')
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Cycle75 test và phân tích lỗi</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.55}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ccd3dc;padding:12px;margin:12px 0}summary{cursor:pointer}</style><h1>Cycle75: test chẩn đoán và hướng crawl</h1><p>Phần test dùng để phân tích lỗi;20% trang giữ riêng chưa được chấm. Nhãn lấy từ dấu câu văn bản, không phải human/audio conversational gold.</p><pre>'+html.escape('\n'.join(lines))+'</pre><h2>Các mẫu để kiểm tra nguồn và nhãn</h2>'+''.join(cases)+'</html>'
    (JOB/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps({'analysis':str(JOB/'error_analysis.md'),'crawl_plan':str(JOB/'crawl_plan_v1.json'),'results':facts},ensure_ascii=False))


if __name__=='__main__':main()
