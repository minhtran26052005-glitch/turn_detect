"""Make a small inspectable end-only report after the paused-model test completes."""
from collections import Counter,defaultdict
import hashlib
import html
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'outputs/sentence_mix_end_only_test_20261107_v1'


def main():
    report=json.loads((P/'report.json').read_text(encoding='utf-8'));assert report['complete']
    independent=json.loads((P/'independent_end_only_verification.json').read_text(encoding='utf-8'));assert independent['complete']
    assert hashlib.sha256((P/'report.json').read_bytes()).hexdigest()==independent['report_sha256']
    analysis=json.loads((P/'end_only_analysis.json').read_text(encoding='utf-8'))
    mt=json.loads((P/'translation_input_end_only_report.json').read_text(encoding='utf-8'))
    rows=json.loads((P/'translation_input_segmentation.json').read_text(encoding='utf-8'))
    locked=json.loads((P/'translation_sentence_count_review_locked.json').read_text(encoding='utf-8'))
    old=json.loads((P/'previous_translation_end_only_recount.json').read_text(encoding='utf-8'))
    cohort_ids={
        'format_cleaned_period':set(map(tuple,locked['multi_message_ids_period_cleaned'])),
        'format_cleaned_all_terminal':set(map(tuple,locked['multi_message_ids_all_terminals']))}
    old_by_id={(r['conv'],r['id']):r for r in old['multi_messages']}
    # One question-first case was in the old single-span cohort; obtain it from
    # a bounded source-only recount receipt when present, without old inference.
    extra=P/'previous_translation_question_first.json'
    if extra.exists():
        for r in json.loads(extra.read_text(encoding='utf-8')):old_by_id[r['conv'],r['id']]=r
    refined={};sample=[]
    for cohort,ids in cohort_ids.items():
        counts=defaultdict(Counter)
        for r in rows:
            if (r['conv'],r['id']) not in ids:continue
            prior=old_by_id.get((r['conv'],r['id']))
            for key in ['ALL','language:'+r['language']]:
                c=counts[key];c['messages']+=1;c['end_only']+=int(r['end_only']);c['has_internal_cut']+=int(not r['end_only'])
                c['chunks']+=len(r['chunks'])
                if prior is not None:
                    c['old_known_messages']+=1;c['old_end_only']+=int(len(prior['old_chunks'])==1)
            if cohort=='format_cleaned_all_terminal':sample.append(r)
        refined[cohort]={k:dict(**v,end_only_percent=100*v['end_only']/v['messages']) for k,v in counts.items()}
        assert refined[cohort]['ALL']['messages']==len(ids)
    selected=[r for r in analysis['rows'] if r['lookahead_words']==2 and r['cohort'] in ('multi_sentence','single_sentence_excluded','at_least_two_completed_fullstops','multi_without_known_legacy_overlap') and r['group'] in ('ALL','language:vi','language:en')]
    metrics=[r for r in report['metric_rows'] if r['domain']=='ALL' and r['lookahead_words']==2 and r['label_policy']=='fullstop_matches_training' and r['cohort']=='all_requested']
    out=dict(complete=True,checkpoint=report['checkpoint'],report_sha256=independent['report_sha256'],
             translation_input_refined_cohorts=refined,translation_raw_period_cohorts=mt['counts'],
             original_test_actual_turn_census_k2=selected,new_test_f1_rows=metrics,
             translation_model_not_rerun=True,training_paused=True,
             caution='End-only is absence of an internal detector cut and caller flush at message end. It is not a translator quality score or real audio endpoint latency. Source-only sentence-count refinement changes denominators only, never training or EOT scoring labels.')
    (P/'user_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    esc=html.escape
    content='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TEST model tạm dừng — loại lượt một câu</title><style>body{font:17px/1.6 system-ui;max-width:1120px;margin:28px auto;padding:0 18px;background:#f6f8fa;color:#17212e}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d1d5db;padding:10px;text-align:left}article{background:white;border-left:5px solid #2563eb;padding:18px;margin:16px 0;border-radius:6px}.end{border-color:#d97706}pre{white-space:pre-wrap}h2{font-size:22px}</style><h1>TEST sau khi tạm dừng TRAIN</h1>'''
    cp=report['checkpoint'];content+='<p>mALBERT 11.941.122 tham số, checkpoint '+str(cp['step'])+' / branch '+str(cp['branch_step'])+', threshold '+str(cp['threshold'])+' chọn từ DEV của cùng checkpoint trước TEST. Trainer vẫn dừng; full optimizer/RNG/order/lanes được giữ. Không chạy lại model dịch.</p>'
    content+='<h2>310 đầu vào của bộ test dịch</h2><p>Theo dấu chấm thô: 289 đoạn đơn, 21 đoạn nhiều đơn vị. Rà định dạng nguồn loại 3 dấu chấm thuộc số thứ tự/No.; còn 18 đoạn nhiều câu theo dấu chấm. Thêm 1 đoạn có câu hỏi trước dấu chấm cuối thì tổng 19. Đây là các nhóm cố định từ văn bản trước khi chạy TEST.</p><table><tr><th>Cách đếm</th><th>Tiếng</th><th>Đoạn nhiều câu</th><th>Không cắt nội bộ, phải flush cuối</th><th>Có cắt nội bộ</th></tr>'
    for cohort,groups in refined.items():
        for group,c in groups.items():
            content+='<tr><td>'+esc(cohort)+'</td><td>'+esc(group)+'</td><td>'+str(c['messages'])+'</td><td>'+str(c['end_only'])+' ('+format(c['end_only_percent'],'.2f')+'%)</td><td>'+str(c['has_internal_cut'])+'</td></tr>'
    content+='</table><h2>TEST điểm ngắt lớn hơn — actual source turns</h2><p>Nhóm multi_sentence loại mọi lượt chỉ một đơn vị câu theo dấu chấm. Nhóm at_least_two_completed_fullstops chỉ giữ lượt có ít nhất hai dấu chấm kết thúc đầy đủ. Context vẫn chạy liên tục qua lượt gốc: số không cắt nội bộ chỉ có nghĩa phát cả lượt nếu caller flush ở cuối lượt; evaluator không tự flush/reset ở ranh giới người nói.</p><table><tr><th>Nhóm</th><th>Tiếng</th><th>Tổng lượt</th><th>Không cắt</th><th>Tỷ lệ</th><th>Có cắt nhưng không trúng fullstop nào</th></tr>'
    for r in selected:
        content+='<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [r['cohort'],r['group'],r['turns'],r['no_internal_model_cut'],format(r['end_only_percent'],'.2f')+'%',r['has_cuts_but_no_correct_fullstop_cut']])+'</tr>'
    content+='</table><h2>Toàn bộ 19 đoạn nhiều câu đã rà định dạng</h2><p>Gốc có dấu chỉ dùng để xem và đếm câu. Model nhận nguồn ASR không dấu câu, tối đa 2 từ tương lai. Phần cuối mỗi message do caller flush, không được tính là model phát hiện EOT. Có cắt không tự đảm bảo cắt đúng nghĩa hoặc chất lượng dịch tốt.</p>'
    for r in sample:
        content+='<article class="'+('end' if r['end_only'] else '')+'"><b>'+esc(str(r['conv'])+' / '+str(r['id'])+' / '+r['language'])+' — '+('Chờ hết message' if r['end_only'] else 'Có cắt trước cuối')+'</b><p>Gốc: '+esc(r['original_written'])+'</p><p>Nguồn ASR: '+esc(r['asr_source'])+'</p><ol>'+''.join('<li>'+esc(x)+'</li>' for x in r['chunks'])+'</ol></article>'
    content+='<p>Đã kiểm mọi input/encoded hash/future/predicted reset/confusion/word conservation, canonical FP32 parity và census độc lập. Nhãn dấu câu là silver, không phải human/audio gold; TEST có nhóm trùng lineage cũ được ghi riêng. Không dùng TEST để chọn model/ngưỡng/loss.</p></html>'
    (P/'review.html').write_text(content,encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
