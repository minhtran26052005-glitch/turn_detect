# V5: text-only trên hội thoại công khai

**Mục tiêu vẫn là accuracy và F1 EOT đều >90% ở từng tiếng Việt, Anh, Trung. Chưa có bản được chấp nhận đạt mục tiêu này.**

Theo lựa chọn của người dùng, model nhận text và ngữ cảnh chữ đã quan sát. Audio được dùng ngoại tuyến để tạo transcript ASR và nhãn tham chiếu; model phân loại không nhận audio, thời gian dừng, người nói tương lai hoặc lời nói tương lai.

| Tập | Mẫu | Việt | Anh | Trung |
| --- | ---: | ---: | ---: | ---: |
| Natural train | 1.433 | 628 / 4 cuộc gọi | 423 / 101 cuộc gọi | 382 / 93 cuộc gọi |
| Natural dev | 293 | 145 / 1 cuộc gọi | 79 / 17 cuộc gọi | 69 / 14 cuộc gọi |
| Giữ riêng | 384 | 210 / 1 cuộc gọi | 93 / 13 cuộc gọi | 81 / 4 cuộc gọi |

Nhãn là **automatic observed-floor/VAD**, chưa có người chấm xác nhận. Pilot Việt chỉ có ít cuộc gọi; không dùng các số này để khẳng định chất lượng trên mọi hội thoại. 70 mẫu V4 đã xem được giữ làm regression, không vào train V5. 384 mẫu mới được giữ để kiểm tra sau lựa chọn model; ô trạng thái notebook cho biết đã chấm hay chưa.

## Các sửa đổi về dữ liệu

- DataHive Việt có 8 cuộc gọi công khai. Cuộc 1–2 giữ làm regression; 3–6 train; 7 dev; 8 giữ riêng. Xử lý tối đa 900 giây đầu mỗi cuộc 3–8.
- CallHome Anh/Trung lấy audio từ phần train và validation của Scicom đã khóa revision. Phần train chia tiếp theo **call_id chính thức**; giữ cả hai kênh và các kiểu kết thúc của cùng cuộc gọi trong một tập. ID clip có cả kênh/kiểu kết thúc nên không dùng làm call_id.
- Với CallHome chỉ giữ các turn có `eot_kind=response`, loại ranh giới do im lặng dài hoặc hết cuộc gọi khỏi tập natural này. Có thể giữ tối đa hai hold mỗi turn, chọn theo vị trí trước khi biết ASR/model score.
- Sửa quy tắc Việt: phản hồi của người kia dài ít nhất 0,5 giây trong 3 giây đầu là EOT tham chiếu, kể cả người ban đầu chỉ nói tiếp sau hơn 4 giây. NEOT cần người đó nói tiếp trong 2 giây và không có phản hồi đáng kể của người kia. Bỏ trường hợp mơ hồ/ngắt chồng. Không đổi nhãn hoặc điểm pilot V4 đã công bố.
- ASR Whisper-small chỉ nghe tối đa 15 giây trên kênh hiện tại, kết thúc ở `pause.start+0,2s`. Ngữ cảnh chữ chỉ lấy các cửa sổ ASR đã kết thúc trước khi cửa sổ hiện tại bắt đầu, không sử dụng nhãn EOT/NEOT để chọn lịch sử.
- Loại input trùng theo thứ tự giữ riêng → dev → train trước khi train. Nhãn, các trường thời gian và `eot_kind` không đi vào classifier.

Pool ASR ban đầu `data_v5/natural_20261025` và pool Việt mở rộng `data_v5/natural_vi_expanded_20261025` là dữ liệu trung gian. **Chỉ `data_v5/canonical_20261025` được dùng để train/dev V5.** Manifest giữ hash, source revision, số cuộc gọi, nhãn và các trường hợp loại.

## Train và đối chứng

mDeBERTa bắt đầu từ checkpoint V4 đã backup, SHA256 `89ebf4e8c8b99e4ea1ad2eac19f9132b0ce8506ee0836943dba343fc849d6509`. Replay chỉ lấy 104.143 mẫu V4 train; 3.600 mẫu legacy dev Việt được giữ làm kiểm tra regression trong lựa chọn dev. Ngưỡng được chọn trên dev, không chọn từ tập giữ riêng.

FP32, TF32 tắt, AdamW, warmup/cosine, gradient checkpointing, microbatch 16 ×2. Tối đa 4 vòng ×600 bước, đánh giá mỗi 150 bước, giảm LR. Giữ checkpoint trước trong tập ứng viên; dừng khi dev đạt tiêu chuẩn hoặc sáu lần dev liên tiếp không cải thiện sau ít nhất hai vòng. Train loss thấp không được coi là chất lượng trên cuộc gọi mới.

Đối chứng tuyến tính TF-IDF/logistic đã chạy local trên cùng dev, không mở tập giữ riêng. Bundle nghiên cứu chọn theo từng tiếng khoảng 1,67 MB nhưng không đạt chất lượng, chưa là bản triển khai. Qwen3-4B-Instruct-2507 được tải khóa revision để chẩn đoán teacher trên cùng dev; dùng cùng phần text causal và ví dụ từ train, không nạp tập giữ riêng. Qwen được nạp GPU sau khi job mDeBERTa kết thúc.

## Theo dõi trong notebook

Mở `a.ipynb`, chạy ô cuối có tag `natural_v5_status` bằng kernel Colab đang kết nối. Ô chỉ đọc trạng thái job, tiến độ, best dev và thống kê dữ liệu; không khởi chạy/ngắt train.

Các lệnh cho người cần thao tác bằng terminal:

```powershell
python tools/natural_v5_train_colab.py status
python tools/natural_v5_qwen_colab.py status
```

Các script launch có chặn trùng job. Không chạy lại `prepare`, `train`, `expand-vi` hoặc `canonicalize` khi job/thư mục cùng tên đã có; hãy đọc trạng thái và receipts trước.

## Nguồn và artifacts

- [Scicom](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/tree/98d1445eab71e4084ccb838dbd87a205f6f2e293), CallHome giữ giấy phép nguồn CC BY-NC-SA4.0, thí nghiệm nghiên cứu này lưu đủ attribution.
- [DataHive](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/tree/1120cd7533b16ba8dd7a7c07e6d316b74b60f72a), CC BY4.0, attribution DataHive.
- [Whisper-small](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d), snapshot ASR dùng trong pool mới.
- [Qwen teacher](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554), Apache2.0, dùng nghiên cứu teacher và khả năng distill; không phải gói triển khai nhỏ.

[Kết quả V4 và player 204 prefix](outputs/stream_v4_20261025/review.html), [dữ liệu V5](data_v5/canonical_20261025/manifest.json), [receipts V5](outputs/natural_v5_20261025). Dấu phẩy/chấm trong crawl chưa có nhãn độc lập nên accuracy/F1 crawl vẫn chưa xác định. Bản Việt QAT mặc định được giữ nguyên.
