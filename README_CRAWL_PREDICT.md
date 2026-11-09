# Crawl về và dự đoán ngay

`turn_v2.crawl_predict` lấy từng ứng viên từ nguồn crawl, dự đoán ngay và flush từng dòng vào `predictions.jsonl`. Không cần chờ crawl hết các trang mới bắt đầu suy luận. Notebook `a.ipynb` có thêm ô **Crawl và dự đoán ngay, rồi Qwen xét một số mẫu**; các cấu hình và output cũ được giữ lại.

## Chạy bằng model gốc trên Windows

```powershell
python -u -m turn_v2.crawl_predict --model .\checkpoint-23000 --limit 30 --output .\outputs\crawl_predict_run01
```

Mở `outputs/crawl_predict_run01/results.html` để xem bảng dự đoán. Mỗi lần chạy chọn thư mục output mới. Có thể thêm `--titles "Hà Nội" "Đà Nẵng"` để chọn chủ đề crawl.

## Chạy trên Colab với model mới

```bash
cd /content/turn_detect_v2
python -u -m turn_v2.crawl_predict --model runs/gold_v1/onnx --limit 30 --review-budget 4 --qwen-review --output outputs/crawl_predict_run01
```

`--qwen-review` dùng Qwen trên GPU để xét một số mẫu. Nếu chỉ có checkpoint smoke, code yêu cầu thêm `--allow-smoke` và đánh dấu kết quả `SMOKE_INFERENCE_ONLY`; điều đó không biến checkpoint smoke thành model dùng thật.

Hàng đợi xét mẫu gồm khoảng một nửa gần ngưỡng quyết định, một mẫu đoạn cắt được báo EOT với xác suất cao, phần còn lại lấy theo một cách chọn cố định. Nhờ vậy còn kiểm tra được các mẫu model rất tự tin. Đây là mẫu audit có giới hạn, không chứng minh những mẫu chưa được Qwen xét là đúng.

Các trường/file kết quả:

- `predictions.jsonl`: dự đoán trực tiếp; có `prediction`, `prob_EOT`, `threshold`, nguồn và độ trễ. Không tạo trường `label` để tự biến dự đoán thành nhãn train.
- `review_queue.jsonl`: các mẫu chọn để xét, kèm lý do chọn.
- `qwen/audit.jsonl`: hai câu trả lời Qwen cho từng mẫu được xét.
- `qwen/silver.jsonl`: nhãn được Qwen chấp nhận theo tiêu chí hiện có; vẫn là silver.
- `reviewed_predictions.jsonl`: dự đoán model kèm trạng thái/nhãn teacher, nếu đã bật Qwen.
- `results.html`: bảng dễ đọc, có link nguồn.
- `report.json`: số mẫu, số bất đồng, model/hash nguồn, và các cờ `training_performed=false`, `rl_performed=false`, `holdout_evaluated=false`.

Code lọc trùng với train/dev/holdout/test cũ bằng nội dung chuẩn hóa, nguồn được bảo vệ và prefix dài trước khi lập hàng đợi. Chỉ đọc các tập này để lọc trùng/hash; không dùng nhãn holdout để tối ưu model hoặc tính điểm ở bước này. Dữ liệu crawl chưa có nhãn tham chiếu độc lập, vì vậy `accuracy` trong report là `null`. `prob_EOT=0.95` không có nghĩa model đạt accuracy 95%.

## RL thì sao?

Crawl → dự đoán → Qwen gán nhãn → fine-tune bằng các nhãn đó là một luồng bổ sung dữ liệu cho supervised learning. Đoạn crawl/predict hiện tại chưa cập nhật trọng số. RL cần state, action và reward rồi tối ưu policy theo reward; chỉ dự đoán trên văn bản mới chưa tạo ra quá trình học RL. Xem [khung RL của Hugging Face](https://huggingface.co/learn/deep-rl-course/unit1/rl-framework).

Một thiết kế RL có thể thử riêng cho turn detection:

| Thành phần | Đề xuất cần kiểm chứng |
| --- | --- |
| State | Lịch sử hội thoại, partial ASR, tín hiệu im lặng/VAD và thời gian |
| Action | `WAIT` hoặc `END_TURN` |
| Reward | Thưởng kết thúc đúng; phạt kết thúc sớm; phạt trễ sau mốc kết thúc thật; phạt không kết thúc trước deadline |
| Dữ liệu tham chiếu | Log streaming có mốc kết thúc thật và thời gian, được người duyệt hoặc hệ thống đáng tin cậy xác nhận |
| Đánh giá | Cùng holdout riêng, đo lỗi cắt sớm, độ trễ và accuracy/F1, so với baseline supervised |

Đây là đề xuất thí nghiệm, chưa phải thuật toán RL đã được cài/chạy. Tập văn bản crawl và nhãn tự khai của Qwen chưa cung cấp mốc kết thúc thật trong hội thoại. Hiện ưu tiên chạy và kiểm tra luồng dữ liệu mới; không coi xác suất của model hoặc nhãn silver là reward chuẩn.

Nếu dùng nhãn do chính classifier dự đoán để train lại, đó là self-training; lựa chọn mẫu và hiệu chỉnh xác suất có vai trò quan trọng theo [tài liệu scikit-learn](https://scikit-learn.org/stable/modules/semi_supervised.html#self-training). Luồng hiện tại giữ nhãn Qwen riêng và không tự nhận dự đoán model làm nhãn đúng.
