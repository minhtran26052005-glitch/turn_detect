# Hội thoại nhà máy: nguồn tham khảo và mẫu kiểm tra

Phạm vi: tiếng Việt, phân đoạn câu từ chữ thường không có dấu câu, nhìn trước 1 hoặc 2 từ. Bộ này dùng để kiểm tra dạng dữ liệu; chưa được đưa vào train hoặc test.

## Nguồn tham khảo

- [TexPrax](https://huggingface.co/datasets/UKPLab/TexPrax): hội thoại thu thập từ công nhân nhà máy tại Đức, về máy hỏng, thiếu vật tư và các vấn đề trong ca. Giấy phép ghi CC BY-NC 4.0. Tham khảo chủ đề và cấu trúc; không chép dữ liệu nguồn vào tập sinh.
- [COALA English](https://zenodo.org/records/8268928): kịch bản công nghiệp và audio tiếng Anh do nhà cung cấp dịch vụ tạo ra. Có thể tham khảo dạng kịch bản và audio, nhưng không gán là hội thoại tự nhiên của công nhân. Xem điều kiện trên trang nguồn trước khi dùng dữ liệu trực tiếp.
- [Manufacturing English](https://englishladder.com/efsp-manufacturing.html): các tình huống hội thoại hư cấu trong nhà máy, ví dụ báo máy dừng, bảo trì và điều phối sản xuất. Tham khảo bối cảnh, không sao chép nguyên văn.

## Mẫu minh họa tự viết — chưa phải đầu ra Qwen

**Tình huống 1: băng tải dừng**

- Người vận hành: Anh ơi, băng tải lại dừng rồi. Em đã báo tổ trưởng.
- Kỹ thuật viên: Em thấy đèn báo lỗi gì không?
- Người vận hành: Đèn đỏ nhấp nháy anh ạ.
- Kỹ thuật viên: Để anh kiểm tra phiếu nhé.

Các điểm xét, với lookahead 2 từ:

| Context | Text nhìn trước | Nhãn |
| --- | --- | --- |
| anh ơi | băng tải | NEOT |
| anh ơi băng tải lại dừng rồi | em đã | EOT |
| em đã báo tổ trưởng | em thấy | EOT |

**Tình huống 2: bàn giao ca**

- Ca trước: Máy đóng gói bị dừng hai lần. Lô này vẫn đang làm dở.
- Ca sau: Có ghi mã lỗi lại chưa anh?
- Ca trước: Anh ghi trong phiếu rồi. Em kiểm tra giúp nhé.
- Ca sau: Vâng, em xem ngay.

| Context | Text nhìn trước | Nhãn |
| --- | --- | --- |
| máy đóng gói bị dừng | hai lần | NEOT |
| máy đóng gói bị dừng hai lần | lô này | EOT |
| lô này vẫn đang làm dở | có ghi | EOT |

Các ví dụ này là hội thoại giả do trợ lý viết để kiểm tra quy ước, không phải dữ liệu nguồn hay đầu ra Qwen. Đoạn nhìn trước có thể thuộc người nói tiếp theo, giả định chữ được đưa vào một luồng liên tục. EOT ở đây là **kết thúc câu**, không tự động tương đương kết thúc lượt nói theo audio.

## Pilot Qwen trên Colab

Script `turn_v2/factory_dialogue_pilot_v1.py` tạo tối đa 8 hội thoại pilot theo các tình huống khác nhau, xác thực JSON và chuyển thành mẫu 1/2 từ. Tệp hội thoại, ví dụ, lỗi và manifest được ghi riêng trên Colab. Mẫu bị lỗi cấu trúc được giữ ở tệp rejected; không tự động đưa vào dữ liệu train. Nhãn là từ cấu trúc câu do Qwen sinh, không phải human/audio gold. Kết quả cần được đọc và kiểm tra tự nhiên, đúng chuyên môn và ranh giới câu trước khi mở rộng.

Sau khi kết nối `a.ipynb` với Colab, dùng `python -X utf8 tools/factory_dialogue_pilot_colab.py start`. Helper tìm phiên hiện tại, chỉ khởi chạy khi kernel rảnh, GPU đủ bộ nhớ và còn ít nhất 43 GiB trống để dành chỗ tải model cùng mức dự phòng 25 GiB. Chạy `python -X utf8 tools/factory_dialogue_pilot_colab.py status` để xem trạng thái; lệnh start không tạo worker thứ hai khi đã có lock. Pilot ghi riêng trong `/content/factory_dialogue_pilot_20261030_v1`, không sửa worker train hoặc dữ liệu gốc. Đây là lưu trữ tạm trên runtime, chưa phải bản sao lưu Drive.

Đầu câu, hết luồng và thay đổi người nói cần xử lý riêng ở ứng dụng streaming; mẫu cuối không đủ từ nhìn trước không được gắn nhãn từ suy đoán.

## Kết quả pilot Qwen 30/10

Qwen3-8B đã sinh 32 hội thoại trên Colab L4 trong một lô; peak VRAM PyTorch reserved 21,75/22,03 GiB theo báo cáo runtime. Bước kiểm tra ban đầu chỉ nhận 3/32 vì 29 hội thoại có nhiều câu trong cùng một phần tử JSON. Bước hậu xử lý tách thận trọng theo dấu kết câu Qwen đã sinh: 31/32 hội thoại qua kiểm tra định dạng, 1 bị loại vì câu quá ngắn; tạo 5.069 mẫu EOT/NEOT hai lookahead. Dữ liệu vẫn chưa được con người duyệt hay đưa vào train. Có mẫu xưng hô thiếu tự nhiên và gợi ý thao tác cần xem xét; đạt định dạng không có nghĩa đạt chất lượng hội thoại/chuyên môn.

- [Đọc hội thoại pilot](qwen_v3_review/review.html)
- [Manifest pilot](qwen_v3_review/manifest.json)
- [Mẫu điểm xét](qwen_v3_review/examples.jsonl)
- [Đầu ra gốc và mẫu bị loại](qwen_v2/)

## Pilot đoạn dài, đơn ngữ 30/10

Theo yêu cầu mới: mỗi người nói một đoạn gồm nhiều câu; các tình huống tập trung vào chuyên gia, giám sát, kỹ sư và chủ quản nhà máy. Mỗi hội thoại chỉ dùng một ngôn ngữ; không sinh bản dịch. Dữ liệu được ghi riêng theo `vi`, `en`, `zh`.

Tổng hai lượt sinh có 95 hội thoại qua kiểm tra cấu trúc: 32 Việt, 31 Anh, 32 Trung, với 3.468 mẫu điểm xét. Lượt Trung đầu có 29 mẫu sai ngôn ngữ nên bị loại; lượt bổ sung dùng prompt tiếng Trung và chấp nhận 29/32 mẫu. Phần lớn lượt nói gồm 3–5 câu; một số có 2 hoặc nhiều hơn 5 câu sau khi tách.

Trong mẫu, EOT lấy ở dấu chấm (`.` hoặc `。`); NEOT ở dấu phẩy hoặc chấm phẩy bên trong câu. Chỉ lấy 1 hoặc 2 từ sau điểm xét trong cùng lượt nói, không mượn lời người khác. Cuối lượt thiếu từ nhìn trước không tạo mẫu. Dấu câu không đi vào model. Nhãn do dấu câu Qwen sinh, chưa qua người duyệt, chưa xác nhận là ranh giới ngữ nghĩa hay audio.

- [Xem các đoạn của ba ngôn ngữ](qwen_v4_paragraphs_review/review.html).
- Mẫu nhỏ nằm trong `qwen_v4_paragraphs_review` và `qwen_v5_zh_review`; dữ liệu đầy đủ từng tiếng và bản ghi lỗi đã được sao lưu riêng trong Drive vừa xác minh nhận diện. Chưa đưa vào train.
