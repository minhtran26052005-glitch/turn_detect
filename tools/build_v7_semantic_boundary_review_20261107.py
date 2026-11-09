"""Attach explicit assistant reading judgments to the fixed saved-trace sample."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

JOB=Path(__file__).resolve().parents[1]/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
NOTES={
'D01':('missed_clear_early_boundary','Đã nêu xong mức điện áp; câu sau giải thích đầu vào/đầu ra. Có thể ngắt sớm. Ghép để dịch cả hai câu vẫn có thể giữ nghĩa.'),
'D02':('uncertain_source','The job of a labor safety supervisor giống tiêu đề hoặc mảnh câu. Dấu chấm có thể là cách chép lời; chưa đủ để kết luận model sai nghĩa.'),
'D03':('plausible_translation_boundary','Now I am letting it run, and it is all good đã có mệnh đề trọn ý. everyone là lời gọi người nghe; ngắt trước đó có thể dùng cho dịch nếu giữ ngữ cảnh.'),
'D04':('uncertain_source','The two small ends of it là cụm danh từ; cả câu ASR gốc cũng rời rạc. Ngắt ở đây chưa đẹp, nhưng khó tách lỗi model khỏi chất lượng transcript chỉ bằng văn bản.'),
'D06':('uncertain_source','Also love it có thể là câu đáp đầy đủ, nhưng Red all over kế tiếp khó hiểu trong ngữ cảnh. Không phán nhãn đúng/sai chắc chắn khi chưa có lời nói gốc.'),
'D07':('missed_clear_early_boundary','Okay là lời xác nhận độc lập, sau đó chuyển sang hỏi khu vực thành phố. Bỏ lỡ điểm ngắt sớm hợp lý.'),
'D08':('plausible_translation_boundary','I am not much of a fan of Tom Brady là phát biểu trọn ý. Phần the quarterback for the Patriots bổ nghĩa tên người. Có thể dịch phần đầu, nhưng nối phần bổ nghĩa cần giữ ngữ cảnh.'),
'D09':('plausible_translation_boundary','That is interesting I always wondered that xong ý; do you use Google mở câu hỏi mới. Dấu phẩy nguồn không làm đây thành một chỗ ngắt vô nghĩa.'),
'D11':('missed_clear_early_boundary','Fine, maintenance is late, I get it là ý nhượng bộ hoàn chỉnh. But mở mệnh đề tiếp theo; giữ chung cũng được nhưng ngắt sớm hoàn toàn hợp lý.'),
'D12':('missed_clear_early_boundary','Crack is on record là thông báo hoàn chỉnh, sau đó I am going now là hành động mới. Bỏ sót cơ hội phát dịch sớm.'),
'D13':('plausible_translation_boundary','Give me the lockout là câu mệnh lệnh hoàn chỉnh. I will have it done by lunch là cam kết khác. Đổi dấu phẩy thành chỗ ngắt không làm đứt cụm từ.'),
'D14':('plausible_translation_boundary','No stop order là quyết định ngắn, rồi let us just sign off là đề nghị tiếp theo. Điểm ngắt có thể chấp nhận khi dịch hội thoại.'),
'D16':('missed_clear_early_boundary','That is concerning là phản hồi trọn ý. Câu sau hỏi ngưỡng an toàn, nên có thể phát bản dịch phần phản hồi trước.'),
'D17':('missed_clear_early_boundary','Thanks là lời cảm ơn độc lập; câu sau là yêu cầu. Tiếp tục chờ không tự chứng minh không hiểu nghĩa, nhưng làm mất điểm dịch sớm.'),
'D18':('premature_semantic_split','Nguồn nói good progress; model dừng ở That is good và đẩy progress sang đoạn sau. Dù That is good đứng riêng được, ở nguồn này nó tách cụm danh từ good progress, có nguy cơ đổi cách diễn đạt và gắn kết.'),
'D19':('plausible_translation_boundary','That is wonderful là lời khen hoàn chỉnh; anh là lời gọi người nghe. Có thể ngắt, dù phần xưng hô có thể bị gắn sang câu sau.'),
'D21':('uncertain_source','Transcript có các chuỗi khó hiểu như bơm định loạn và Mắt en yêu. Phần sử dụng là sáu con có thể đủ ý với ngữ cảnh, nhưng cần âm thanh/soát nguồn để phán chắc.'),
'D22':('uncertain_source','Làm là ship mình dạy mình như vậy đó và Là cái câu đó có lỗi/đứt gãy chép lời. Không quy toàn bộ lỗi điểm ngắt thành lỗi hiểu nghĩa của model.'),
'D23':('premature_semantic_split','Zeelim sáu mươi ba một là tên/mã máy trong cùng cụm chủ ngữ; sau đó còn vị ngữ là dòng máy. Cắt sau Zeelim làm tách mã khỏi tên và chủ ngữ khỏi vị ngữ.'),
'D24':('uncertain_source','Rồi ở đây là phần mở ý mới. Model cắt sau Rồi thay vì trước Rồi hoặc sau mệnh đề; lệch một từ nối, ảnh hưởng dịch tùy cách giữ ngữ cảnh. Không xếp là điểm ngắt rõ đẹp hay lỗi nghĩa chắc chắn.'),
'D26':('uncertain_source','Red all over đứng một mình trong bản Việt vẫn chưa rõ là tên/tựa, chữ chép nhầm, hay mảnh câu. Cần kiểm nguồn trước khi xem fullstop này là một kết thúc nghĩa chắc chắn.'),
'D27':('missed_clear_early_boundary','Phát biểu về chuyển dịch ý thức hệ đã kết thúc; Mình cũng từng nghĩ mở một suy nghĩ khác. Có thể ngắt tại dấu chấm.'),
'D28':('plausible_translation_boundary','Mệnh đề nêu tên Tellus và Telluris đã đủ ý; không biết tại sao nữa là nhận xét tiếp. Có thể tách thành hai nhịp dịch.'),
'D29':('plausible_translation_boundary','Lựa chọn chuyến bay và giờ bay đã rõ; có hai điểm dừng là thông tin bổ sung. Có thể dịch theo hai đoạn nếu vẫn giữ liên hệ với cùng chuyến bay.'),
'D31':('missed_clear_early_boundary','Đã xác nhận hoàn thành thao tác/kiểm tra rồi chuyển sang câu hỏi/kết luận mới. Đây là cơ hội ngắt sớm hợp lý.'),
'D32':('missed_clear_early_boundary','Vậy là xong hoàn tất chỉ dẫn trước; Lọc nước mở chỉ dẫn tiếp. Model giữ lại không tự làm sai nghĩa, nhưng bỏ lỡ điểm phát dịch.'),
'D33':('plausible_translation_boundary','Em cần xem bề mặt kỹ chứ đã thành lời yêu cầu trọn ý. anh Tom là lời gọi người nghe; ngắt ở đây có thể chấp nhận nhưng cần giữ xưng hô.'),
'D34':('premature_semantic_split','Nguồn là em chờ chị Linda qua kéo cầu chì nhé. Cắt sau em chờ tách động từ khỏi người/sự việc đang chờ; đoạn sau có thể bị hiểu thành lời gọi/yêu cầu khác.'),
'D36':('missed_clear_early_boundary','Đó là tin tốt là phản hồi hoàn chỉnh; Tôi sẽ giới thiệu là hành động mới. Có thể dịch phản hồi trước.'),
'D37':('missed_clear_early_boundary','Tốt quá là lời đánh giá độc lập, tiếp theo là yêu cầu Cho tôi biết. Bỏ lỡ một điểm ngắt sớm rõ ràng.'),
'D38':('plausible_translation_boundary','Tốt là lời xác nhận ngắn; anh vui lòng gửi báo cáo là yêu cầu mới. Dấu phẩy nguồn vẫn cho phép ngắt theo nhịp dịch.'),
'D39':('premature_semantic_split','Thời gian ... của chiếc xe này mới là cụm chủ ngữ, thật ấn tượng là nhận xét/vị ngữ. Cắt trước thật ấn tượng làm phần đầu chưa nói xong điều gì về chủ ngữ.'),
}
LABELS={'plausible_translation_boundary':'Có thể ngắt để dịch',
        'premature_semantic_split':'Cắt đứt cụm/quan hệ nghĩa',
        'missed_clear_early_boundary':'Bỏ lỡ điểm dịch sớm rõ',
        'uncertain_source':'Nguồn/điểm ngắt chưa rõ',
        'valid_positive_control':'Đối chứng cắt đúng hợp nghĩa'}


def main():
    source=JOB/'semantic_boundary_review_sample_20261107.json'
    sample=json.loads(source.read_text(encoding='utf-8'))
    counts=defaultdict(Counter)
    for row in sample['rows']:
        if row['case']=='tp':
            category='valid_positive_control';note='Ý trước đã thành phát biểu/lời đáp hoàn chỉnh; phần sau mở ý hoặc câu hỏi khác. Điểm cắt phù hợp cả dấu chấm và cách đọc nghĩa.'
        else:category,note=NOTES[row['id']]
        row.update(review_category=category,review_label=LABELS[category],review_rationale=note)
        counts[row['case']][category]+=1
    assert len(NOTES)==32
    assert counts['fp']==Counter(plausible_translation_boundary=10,premature_semantic_split=4,uncertain_source=2)
    assert counts['fn']==Counter(missed_clear_early_boundary=11,uncertain_source=5)
    sample.update(review_complete=True,review_utc=datetime.now(timezone.utc).isoformat(),
        review_method='Assistant qualitative reading of 32 source-stratified errors plus 8 positive controls; not blinded, not human gold, no audio and no translation model outputs evaluated.',
        reviewer_saw_extended_source=True,model_saw_only_saved_causal_context_and_two_future_words=True,
        error_sample_not_population_representative=True,review_counts={k:dict(v) for k,v in counts.items()},
        sample_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inference='Punctuation false positives include plausible translation boundaries and genuine premature splits. False negatives can miss an early translation opportunity without proving mistranslation or inability to understand. No global corrected F1 can be inferred from this sample.')
    target=JOB/'semantic_boundary_review_20261107.json'
    target.write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    esc=html.escape
    cards=[]
    for row in sample['rows']:
        cards.append('<article class="'+row['review_category']+'"><h2>'+esc(row['id']+' · '+row['language']+' · '+row['domain']+' · '+row['case'].upper())+'</h2><b>'+esc(row['review_label'])+'</b><blockquote>'+esc(row['original_source_excerpt'])+'</blockquote><p>'+esc(row['review_rationale'])+'</p><details><summary>Đầu vào thực tế và định danh</summary><p>Context không dấu câu (đuôi tối đa 70 từ): '+esc(row['actual_model_context_tail'])+'</p><p>Hai từ tương lai model thấy: <b>'+esc(row['actual_model_future'])+'</b></p><p>p(EOT)='+str(round(row['p_eot'],6))+'; threshold=0.1</p><p>'+esc(row['stream_key'])+' · boundary '+str(row['boundary'])+'</p></details></article>')
    content='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rà nghĩa điểm cắt DEV v7</title><style>body{font:17px/1.6 system-ui;max-width:1040px;margin:32px auto;padding:0 18px;color:#1f2937;background:#f5f7fa}article{padding:20px;margin:16px 0;background:white;border-left:6px solid #9ca3af;border-radius:8px}h2{font-size:18px;margin:0 0 10px}blockquote{margin:14px 0;padding:12px;background:#f3f4f6}details{font-size:14px}.premature_semantic_split{border-color:#dc2626}.plausible_translation_boundary{border-color:#16a34a}.missed_clear_early_boundary{border-color:#d97706}.valid_positive_control{border-color:#2563eb}button{padding:8px 14px;margin:5px}</style><h1>Điểm cắt: sai dấu câu hay chưa trọn nghĩa?</h1><p>DEV v7 checkpoint branch 480.000 / global 3.711.046, threshold 0,1. Đọc lại trace đã lưu; không chạy lại model và không đổi TRAIN/DEV/nhãn/ngưỡng.</p><p>Dấu <b>⟂</b> là vị trí đang xét. FP: model cắt dù không có dấu chấm; FN: model không cắt dù nguồn có dấu chấm. Trích nguồn giữ dấu câu để người rà đọc; model chỉ thấy context không dấu câu và đúng hai từ tương lai ghi trong mỗi mục. Phần nguồn xa hơn chỉ phục vụ rà offline.</p><p>Chọn cố định bằng hash: mỗi nguồn × mỗi tiếng lấy 2 FP + 2 FN + 1 TP. Tổng 32 lỗi + 8 đối chứng. Đây là nhận định đọc nghĩa của trợ lý, không phải nhãn người kiểm độc lập, không có audio hay chấm bản dịch. Không ngoại suy tỷ lệ này cho toàn bộ DEV.</p><p><b>16 FP:</b> 10 chỗ có thể ngắt để dịch, 4 chỗ cắt đứt cụm/quan hệ nghĩa, 2 chỗ chưa rõ. <b>16 FN:</b> 11 điểm có thể dịch sớm rõ, 5 điểm nguồn chưa rõ. Ngắt được theo nghĩa không đảm bảo bản dịch thực tế giữ nguyên chất lượng.</p><div>'''
    for category,label in [('all','Tất cả')]+list(LABELS.items()):
        content+='<button onclick="filter(\''+category+'\')">'+esc(label)+'</button>'
    content+='</div>'+''.join(cards)+'''<script>function filter(c){document.querySelectorAll('article').forEach(e=>e.hidden=c!=='all'&&!e.classList.contains(c))}</script></html>'''
    (JOB/'semantic_boundary_review_20261107.html').write_text(content,encoding='utf-8')
    print(json.dumps(sample['review_counts'],ensure_ascii=False))


if __name__=='__main__':main()
