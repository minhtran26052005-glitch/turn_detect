# Chiến dịch Colab chạy liên tục

## Material Passport

Experiment: `continuous_v13_20261026`. Chạy crawl → Qwen judge → data tích lũy → supervised/RL → dev → snapshot → vòng tiếp theo. Ba model riêng, text-only. Không dừng vì hết hai vòng, vì dev chững hoặc vì silver dev đạt90%. Chưa có chứng nhận accuracy/F1 EOT90% trên hội thoại thực.

## Thay đổi so với các vòng trước

Vòng V12 chỉ có183prefix và107reward được nhận, mỗi nhánh75bước, giới hạn2vòng. Mục tiêu Qwen là **đủ ý về ngữ nghĩa**; nhãn natural dev là **nhường lượt quan sát qua VAD/âm thanh**. Hai việc có thể khác nhau dù cùng text. Việc đó góp phần giải thích sự lệch điểm, nhưng chưa tách được đóng góp riêng của ASR, kiến trúc nhỏ, nhãn hoặc lượng data.

V13 đóng băng một tập **semantic silver dev** trước khi train, gồm text natural dev được Qwen gán lại và các nguồn Wikipedia riêng. Giữ nguyên natural-floor dev như cột chẩn đoán riêng, không đổi tên nó thành semantic gold. Dữ liệu natural train được gán lại một lần theo cùng định nghĩa semantic, rồi tích lũy các nguồn crawl mới. Hai prompt cùng Qwen vẫn có tương quan. Silver dev không phải kiểm chứng người chấm độc lập và không tự chứng nhận90%.

Mỗi tiếng có checkpoint riêng. Mỗi vòng dùng cùng checkpoint và dữ liệu để thử SFT và one-step REINFORCE; mỗi5vòng có thêm cost-CE. Chỉ cập nhật champion khi min(accuracy,F1 EOT) trên semantic dev tăng. Vòng không tăng điểm vẫn tiếp tục thu thập data. Natural-floor metrics dùng ngưỡng semantic để bộc lộ chênh lệch nhiệm vụ. Tập384regression trước đó không được mở lại để chọn nhánh.

Candidate V13 học **FP32 latent** để tránh khóa chất lượng vào scale QAT cũ ngay từ đầu. Vocabulary/kiến trúc student riêng giữ nguyên. Đây là model nghiên cứu nhỏ, chưa là file ONNX INT4 đã được nghiệm thu; chưa promote vào registry serving. QAT/export và kiểm packed parity chỉ có ý nghĩa sau khi chất lượng phù hợp, không dùng kích thước nhỏ làm chứng cứ accuracy.

## Chạy, theo dõi và khôi phục

`python tools/continuous_v13_colab.py status` đọc tiến độ hiện tại, xác minh cmdline/start ticks của tiến trình, GPU, data tích lũy và champion. `resume` chỉ khởi chạy khi không còn worker đúng cmdline; khóa hệ điều hành trong worker chống chạy trùng.

Worker chạy `--max-cycles 0`, không có timeout toàn chiến dịch. Một vòng mặc định tối thiểu5phút,3trang/mỗi tiếng,200bước/mỗi candidate. Các lần crawl có rate limit và backoff429/503. Thiếu data hai lớp thì chờ, lỗi khôi phục được thì lưu lỗi/backoff rồi thử lại. Thiếu15GiB trên Colab thì chờ giải phóng, không xóa tự động các model cũ. Không có thao tác lách giới hạn/keepalive Colab.

State SQLite WAL lưu cursor crawl, dữ liệu được nhận, nguồn, nhóm/dev được bảo vệ, cycle và champion. Optimizer/RNG được lưu mỗi25bước. Cuối vòng tạo `recovery/latest_snapshot.zip` gồm code, database snapshot, data cần thiết và checkpoint nhỏ; không kèm trọng số Qwen hoặc credentials. Bản V13 đầu tiên gặp legacy row không có source_id trước khi train, đã dừng nhẹ và tiếp tục bằng module `continuous_v13_r2` (legacy source_id) rồi `continuous_v13_r3` (clone frozen target cho KL backward); source/log cũ vẫn được giữ.

Gói snapshot nhỏ được theo dõi và sao lưu ngoài VM qua kết nối Google Drive vào thư mục `turn_detect_continuous_20261026`. Nếu đã mount `/content/drive/MyDrive`, worker cũng ghi snapshot vào folder đó. Mount Drive vẫn cần người dùng cấp quyền theo flow Colab chuẩn; không tự đọc credentials ngoài phiên kết nối đang có.

Nếu Colab VM mới mất `/content/turn_detect_v2`, dùng snapshot đã kiểm hash: `python tools/continuous_v13_colab.py restore`, rồi `resume`. Restore từ chối ghi đè database đang có. Nếu HTTP404/503, trạng thái remote là chưa biết, cần kết nối lại kernel trước; không đoán job còn sống từ PID/log cũ. Chỉ code không thể bảo đảm VM Colab hoạt động24/7: [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Để dừng theo yêu cầu người dùng: tạo file `outputs/continuous_v13_20261026/STOP_REQUESTED` trên Colab hoặc gửi SIGTERM sau khi xác minh đúng worker. Worker lưu trạng thái rồi dừng; không kết thúc process khác. Không tạo file dừng chỉ vì dev chững hoặc đang chờ network.

## Artifacts

`outputs/continuous_v13_20261026/status_receipt.json` là snapshot lần kiểm tra gần nhất, không tự cập nhật. Chạy status để lấy trạng thái mới. Mỗi cycle có audit crawl, report, prediction và optimizer receipt trên Colab. Notebook `a.ipynb` có ô cuối `continuous_v13_status` đọc kernel thật, không khởi chạy lại job. Các báo cáo cũ giữ nguyên để so sánh đúng protocol.
