# Tự động crawl và RL

## Material Passport

Experiment type: BOUNDED_AUTOMATIC_CRAWL_AND_ONE_STEP_REINFORCE. Ba model riêng, input text-only. GPU Colab L4 23.034MiB. Không tải Qwen hoặc full teacher về Windows. Chưa chứng nhận accuracy/F1 EOT90%hội thoại thực.

Qwen làm người chấm tự động cho trạng thái: lời hiện tại đủ ý để kết thúc (EOT), rõ ràng dang dở (NEOT), hoặc AMBIGUOUS. Model GRU nhỏ riêng của từng tiếng mới là policy được cập nhật/triển khai. Crawl chỉ đem về văn bản, không tự đem về reward đúng/sai. Ngắt tại dấu câu không tự là nhãn.

Vòng chạy: crawl các bài có provenance → tạo prefix theo từng câu tại35%/65%/100%đơn vị text → model chọn hành động wait/stop → Qwen chấm state theo hai cách hỏi → tính reward → REINFORCE kèm CE/KL và replay tự động → chấm dev → chỉ giữ nếu min dev-slice score tăng. Code chấm Qwen trước theo batch rồi sample action/update; không cần người bấm từng mẫu, nhưng đây không phải tương tác với người dùng thật hay simulator độ trễ.

Reward: đúng+1, ngắt sớm−2, chờ trênEOT−0,5. Chi phí chờ là ưu tiên cố định, chưa đo mili giây. Khi có nhãn binary, reward cho hai action đều có thể suy ra; do đó thử RL phải so với CE/cost-CE cùng ngân sách. V10 đã làm đối chứng và chưa thấy REINFORCE vượt hai nhánh đó trên dev.

## Người chấm và giới hạn

V11 dùng relative A/B/C score từ Qwen4B, không vượt bộ6probe rõ ràng đầy đủ/dang dở. Vòng đó dừng với0policy update. Kết quả và code giữ nguyên.

V12 dùng [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218), Apache2, revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16. Tắt thinking theo chat template; sinh đáp án A/B/C trực tiếp, hai view phải đồng ý EOT/NEOT. Bất đồng, AMBIGUOUS hoặc lỗi format không tạo reward. Không đưa nhãn dev, suffix hoặc dự đoán policy vào prompt người chấm. Probe6/6 chỉ kiểm tra đường chạy trên ví dụ rõ ràng; chưa chứng nhận độ chính xác judge trên crawl. Thay cả model/cách lấy nhãn, nên không quy kết riêng lợi ích cho số tham số.

Qwen chấm **semantic completeness**, còn dev natural hiện có tham chiếu **observed-floor/VAD tự động**. Hai mục tiêu khác nhau: câu đã đủ ý nhưng người nói vẫn tiếp tục có thể là EOT semantic và NEOT observed-floor. Reward tăng vì học theo judge không chứng minh nhường lượt thực tốt hơn. Không biến nhãn judge thành gold, không lấy accuracy với chính judge làm chứng nhận90%.

## Điều khiển và artifacts

Đã khóa tối đa2vòng,75bước/mỗi tiếng/mỗi vòng. Mỗi vòng crawl tối đa2bài mỗi tiếng; giới hạn timeout1800s. Có ít nhất2mẫu reward mỗi lớp mới cập nhật; thiếu đa dạng thì bỏ qua. Các checkpoint/dự đoán/số liệu nằm dưới `outputs/auto_rl_v12_20261025/round_01` và `round_02`. Nhánh dev giảm bị bỏ, model mặc định hiện có không bị thay.

`python tools/auto_rl_v11_colab.py status --revision 2` đọc job hiện có. `collect --revision 2` thu sau khi hoàn tất và kiểm hash. Không gọi lại launch với cùng run; code từ chối ghi đè. Muốn mở chiến dịch mới phải chọn run mới và khóa cohort/protocol trước; không tự chạy vô hạn.

Nguồn Wikipedia CC BY-SA, lưu URL/page revision/hash từng bài trong các file `sources.jsonl`. Các nguồn204prefix cũ và input dev/seen regression được bảo vệ. Request429 được ghi lỗi và bỏ qua, không lặp dồn dập. Không chạy lại384mẫu đã xem để chọn checkpoint/ngưỡng.

[Bảng vòng tự động](outputs/auto_rl_v12_20261025/review.html) · [Số liệu](outputs/auto_rl_v12_20261025/summary.json) · [Đối chứng V10](outputs/reward_v10_20261025/review.html). Ô cuối `auto_rl_status` trong `a.ipynb` đọc kết quả Colab, không tự khởi chạy thêm job.
