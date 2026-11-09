# Bản thử Việt–Anh–Trung: teacher → distill → QAT INT4

Vòng train streaming V4 ngày 25/10/2026 được ghi riêng trong [README_STREAM_V4.md](README_STREAM_V4.md). Các số bên dưới là kết quả V3 tại thời điểm chốt; từ V4, LiveKit đã được phân lại theo cuộc hội thoại thành train/dev và không còn là test độc lập.

**Chưa đạt mục tiêu 90% cả ba ngôn ngữ. Không thay bản Việt QAT đang có.** Bản này dùng kiểm chứng đường train và lỗi dữ liệu/streaming.

Teacher mDeBERTa12lớp/768: khôi phục encoder/head và embedding đã có của bản Việt, mở lại vocabulary đa ngôn ngữ pretrained rồi fine-tune cân bằng ba tiếng800bước trên Colab L4. Train **71,329 mẫu**, dev **7,190 mẫu**. Student GRU128/embedding64/1lớp: distill 11epoch, tiếp QAT 6epoch. Vocabulary **18,711**, chỉ fit train, giữ fallback Latin/CJK. ONNX **728,710byte**, cả thư mục `onnx/` **1,888,631byte**; không tính thư viện đã cài. Trọng số matrix4-bit, phép tính FP32.

## Kết quả trên chính file INT4

| Tập | Ngôn ngữ | Mẫu | Accuracy | F1 EOT |
| --- | --- | ---: | ---: | ---: |
| Dev pilot nhãn suy ra | vi | 2400 | 90.92% | 87.21% |
| Dev pilot nhãn suy ra | en | 2387 | 87.56% | 83.08% |
| Dev pilot nhãn suy ra | zh | 2403 | 92.30% | 88.85% |
| Official-test hội thoại, nhãn suy ra | en | 2407 | 86.87% | 82.09% |
| Official-test hội thoại, nhãn suy ra | zh | 2405 | 92.31% | 88.66% |
| LiveKit: văn bản tại khoảng dừng thực | en | 1105 | 56.20% | 46.81% |
| LiveKit: văn bản tại khoảng dừng thực | zh | 952 | 62.71% | 47.41% |

Full Vietnamese dev17.833: student accuracy **90.42%**, F1 **85.94%**; bản Việt QAT riêng F1 **90,19%**. Teacher ba tiếng giữ F1Việt **90.31%**. Student nhỏ bị regression tiếng Việt.

## Dùng để kiểm tra

```powershell
python -m pip install -r models/turn_v3_trilingual_pilot_20261024/onnx/requirements.txt
python models/turn_v3_trilingual_pilot_20261024/onnx/run_model.py --language en --text "I would like to book a flight to London tomorrow"
python models/turn_v3_trilingual_pilot_20261024/onnx/run_model.py --language zh --text "我想订一张明天从北京到上海的火车票"
```

Python: `TinyOnnxDetector(path).predict(text, context, language='vi'|'en'|'zh')`. Ngôn ngữ cần truyền rõ; không có bộ nhận diện ngôn ngữ tự động. Ngưỡng từng tiếng lấy từ dev, đã khóa trước cả hai lượt kiểm tra test/external. Runner không cần Torch/Transformers.

## Cách đọc kết quả

- MultiWOZ2.2/CrossWOZ có lượt hội thoại, không có nhãn semantic EOT cho mọi prefix. Bản thử lấy cuối lượt làm EOT, cắt phần có lời tiếp làm NEOT proxy. Một prefix có thể đã đủ ý; không gọi các nhãn này là gold chuẩn semantic.
- LiveKit EOT Bench có transcript/timing và khoảng im lặng thực. Bài kiểm tra chỉ dùng văn bản quan sát tại `silence.start+0,2s`, không tải audio, không dùng tương lai, không quét threshold/policy. Đây không phải điểm leaderboard audio chính thức; transcript lag=0 là điều kiện lý tưởng hơn ASR thực.
- Crawl39đoạn Wikipedia, replay12đoạn/2.253prefix cho mỗi model. Dấu phẩy/dấu chấm không tự là EOT. Model bỏ dấu câu nên chỉ thêm dấu ở cuối cho delta0. Đầu vào quá253token hiện giữ phần đầu; runner gắn `input_truncated` nhưng chưa sửa chính sách này trong bản thử.
- Dữ liệu tiếngViệt cũ/holdout đã dùng giữ nguyên, không chấm lại holdout. Official-test EN/ZH và LiveKit chưa được đưa vào train hay dùng chọn ngưỡng.

## Nguồn và artefacts

- [MultiWOZ2.2](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2), revision `fe0c8e65cfcd8462bd33c86e35f21addc84ca82b`. Attribution dữ liệu CC BY4.0 theo Cambridge; code repo MIT.
- [CrossWOZ](https://github.com/thu-coai/CrossWOZ), revision `df82c9fdff91b9b130f2d6b89110d3870ba6260e`; upstream Apache2.0.
- [LiveKit EOT Bench](https://huggingface.co/datasets/livekit/eot-bench-data), revision `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`; dữ liệu CC BY4.0 theo dataset card (khác giấy phép Apache2.0 của code benchmark).400turnAnh và344turnTrung, tổng2.057điểm dừng đủ200ms.
- Wikipedia: nguồn/attribution trong `outputs/trilingual_stream_20261024/crawl/sources.json` (CC BY-SA theo điều khoản Wikipedia).

[Báo cáo tổng](outputs/trilingual_20261024/review.html) · [Phát lại từng prefix](outputs/trilingual_stream_20261024/review.html) · [Hồ sơ thí nghiệm](EXPERIMENT_TRILINGUAL_20261024.md).

Teacher FP32 checkpoint da luu nguyen ven tai `outputs/trilingual_20261024/runs/tri_teacher_20261024/best/` (config/tokenizer/model.safetensors). SHA256 weights: `e962d06668aeceaeb7c05116a1a55e664f5f3f21e9c7eaf1d250b922028a4b80`. [Kiem tra hoan tat](outputs/trilingual_20261024/completion_validation.json) va [receipt backup](outputs/trilingual_20261024/teacher_checkpoint_backup.json).
