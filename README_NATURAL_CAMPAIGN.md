# Hội thoại tự nhiên: các vòng text-only V5–V8

## Material Passport

Experiment run: **FINAL_PUBLIC_HELD_EVALUATED, QUALITY_TARGET_UNMET**. Mục tiêu: accuracy và F1 EOT đều >90% riêng cho Việt/Anh/Trung. Nguồn công khai; nhãn nhường lượt VAD/observed-floor tự động, chưa có human gold. Mô hình chỉ nhận text và lời đối diện đã xuất hiện trong quá khứ.

Train natural1.433mẫu từ4cuộc gọi Việt,101cuộc gọi Anh,93cuộc gọi Trung; dev293mẫu.384mẫu giữ riêng từ1cuộc gọi Việt,13cuộc gọi Anh,4cuộc gọi Trung. Các điểm cùng cuộc gọi không độc lập, nhất là Việt dev chỉ có1cuộc gọi. Replay chỉ lấy V4 train; pilot70mẫu V4 đã xem là regression. Điểm dev không phải điểm test mới.

| Model | Tiếng | Accuracy dev | F1 EOT dev | EOT nhầm/hold | Phạm vi |
| --- | --- | ---: | ---: | ---: | --- |
| V5 mDeBERTa | vi | 60.00% | 58.57% | 52.08% | V5 dev; sampled prior context, Whisper-small |
| V5 mDeBERTa | en | 72.15% | 81.36% | 65.38% | V5 dev; sampled prior context, Whisper-small |
| V5 mDeBERTa | zh | 78.26% | 87.39% | 87.50% | V5 dev; sampled prior context, Whisper-small |
| V6 MiniLM fine-tune | vi | 53.79% | 48.06% | 51.04% | V6 dev; current-turn/immediate-partner text, Whisper-small |
| V6 MiniLM fine-tune | en | 68.35% | 80.62% | 92.31% | V6 dev; current-turn/immediate-partner text, Whisper-small |
| V6 MiniLM fine-tune | zh | 76.81% | 86.89% | 100.00% | V6 dev; current-turn/immediate-partner text, Whisper-small |
| V6 Qwen feature head | vi | 51.03% | 48.92% | 58.33% | V6 dev; frozenQwen + class/call weighted logistic head |
| V6 Qwen feature head | en | 73.42% | 82.64% | 69.23% | V6 dev; frozenQwen + class/call weighted logistic head |
| V6 Qwen feature head | zh | 81.16% | 88.70% | 68.75% | V6 dev; frozenQwen + class/call weighted logistic head |
| V6 MiniLM feature head | vi | 45.52% | 43.97% | 63.54% | V6 dev; frozenMiniLM + class/call weighted logistic head |
| V6 MiniLM feature head | en | 67.09% | 80.30% | 100.00% | V6 dev; frozenMiniLM + class/call weighted logistic head |
| V6 MiniLM feature head | zh | 76.81% | 86.89% | 100.00% | V6 dev; frozenMiniLM + class/call weighted logistic head |
| V7 mDeBERTa + PhoWhisper | vi | 48.28% | 47.55% | 62.50% | V7 dev; samecases/callgroups, repairedVIASR |
| V7 mDeBERTa + PhoWhisper | en | 67.09% | 76.36% | 57.69% | V7 dev; samecases/callgroups, repairedVIASR |
| V7 mDeBERTa + PhoWhisper | zh | 62.32% | 76.79% | 100.00% | V7 dev; samecases/callgroups, repairedVIASR |
| V8 Qwen supervised LoRA | vi | 58.62% | 56.52% | 52.08% | SamePhoWhisper dev; supervisedLoRA, fulltext-only teacher |
| V8 Qwen supervised LoRA | en | 74.68% | 82.46% | 53.85% | SamePhoWhisper dev; supervisedLoRA, fulltext-only teacher |
| V8 Qwen supervised LoRA | zh | 76.81% | 86.44% | 87.50% | SamePhoWhisper dev; supervisedLoRA, fulltext-only teacher |
| Chốt ngưỡng exact dev | vi | 58.62% | 56.52% | 52.08% | SamePhoWhisper dev; weights fixed, exact threshold boundaries |
| Chốt ngưỡng exact dev | en | 74.68% | 82.46% | 53.85% | SamePhoWhisper dev; weights fixed, exact threshold boundaries |
| Chốt ngưỡng exact dev | zh | 76.81% | 86.44% | 87.50% | SamePhoWhisper dev; weights fixed, exact threshold boundaries |

V5: full mDeBERTa, balanced language/label sampling, replay,3vòng đã chạy1.800bước rồi dừng theo6lần dev không cải thiện. Checkpoint tốt nhất vòng2,bước300, không chọn checkpoint cuối đơn thuần.

V6 dựng lại current turn và immediate previous partner theo audio/time; text chỉ đến thời điểm quyết định. Vẫn cùng case, nhãn và call split; hai người nói của một cuộc gọi nằm trong cùng tập. Encoder feature heads học train rồi chọn regularization/ngưỡng trên dev. MiniLM cross-encoder có21,44triệu tham số học, embeddings cố định; đã train300bước, dừng dev plateau. Đây là thử nghiệm riêng, không thay bản Việt QAT mặc định.

Không coi Qwen là nhãn gold. Qwen lớn và các head nhỏ chưa vượt mục tiêu nên chưa dùng chúng để tạo một gói distill/INT4 được chấp nhận. MiniLM FP32 khoảng470,6MB có tiềm năng nén dưới112MB ở4bit, nhưng kích thước chưa phải chứng cứ chất lượng. Chưa xuất gói V6 đạt90%.

V6 ASR repetition diagnostic:37/628mẫu Việt train và6/145mẫu Việt dev có lặp3-gram trên30%. Đây không phải WER và chưa có transcript người chấm. V7 đổi tuyến ASR Việt sang PhoWhisper-large trên đúng cửa sổ đã quan sát; không bỏ mẫu theo điểm model, không sửa nhãn/call split. Anh/Trung giữ nguyên transcript V6. So sánh ASR matched-dev cố định teacher/codec/case/label/ngưỡng; tuyến ASR bao gồm model và beam decoding, chưa tách được đóng góp riêng của từng yếu tố.

PhoWhisper diagnostics (không phải WER):

```json
{
  "train": {
    "cases": 628,
    "old_high_repetition": 37,
    "new_high_repetition": 10,
    "changed_texts": 620,
    "empty_new_texts_kept": 0,
    "human_transcript_available": false
  },
  "dev": {
    "cases": 145,
    "old_high_repetition": 6,
    "new_high_repetition": 0,
    "changed_texts": 145,
    "empty_new_texts_kept": 0,
    "human_transcript_available": false
  }
}
```

V7: dừng `DEV_PLATEAU`; checkpoint chọn bằng dev `runs/stream_v4_campaign_20261025/round_02/best`, SHA256 `89ebf4e8c8b99e4ea1ad2eac19f9132b0ce8506ee0836943dba343fc849d6509`.

V8: Qwen3-4B đã học supervised LoRA thật sự (khác frozen head ở V6), chạy180bước, dừng `MAX_STEPS`. Adapter cần đi cùng base8GB đã khóa revision; kích thước adapter không phải kích thước triển khai. Không phải RL, chưa QAT/exportINT4.

Sau khi cố định checkpoint, ngưỡng cuối được chọn tại ranh giới xác suất chính xác trên cùng dev; lưới0,005 trước đó có thể bỏ sót các score rất sát0,5. Không đổi trọng số, không dùng dự đoán test. Toàn bộ xác suất dev và hash được giữ trong `exact_dev_calibration.json`.

384mẫu giữ riêng chỉ được chấm sau khi khóa lựa chọn cuối, không dùng để tune dev. Sau lượt chấm này chúng chuyển thành regression.204prefix crawl vẫn chưa có nhãn độc lập nên accuracy/F1 crawl=null. Không gọi model đạt90% từ train-fit, từ tỷ lệ EOT lớn hoặc từ điểm judge tự gán.

[Bảng kết quả](outputs/natural_v7_20261025/review.html) · [Player 204 prefix: model đã chọn](outputs/natural_v7_20261025/crawl_review.html) · [Dữ liệu/recipe V5](README_NATURAL_V5.md). Ô cuối notebook `natural_campaign_status` đọc trạng thái các job, không khởi chạy lại chúng.

Nguồn: [DataHive CC BY4.0](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/tree/1120cd7533b16ba8dd7a7c07e6d316b74b60f72a); [Scicom/CallHome](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/tree/98d1445eab71e4084ccb838dbd87a205f6f2e293), CallHome giữ giấy phép nguồn CC BY-NC-SA4.0; [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554), Apache2.0; [MiniLM đa ngôn ngữ](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42), Apache2.0; [PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large/tree/b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb), BSD3Clause. PhoWhisper attribution: Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen, ICLR2024 Tiny Papers.

Đã tìm thêm VietMed/VietMed-Sum có transcript người chấm. Chưa đưa vào train EOT: đoạn audio ASR được cắt không tự là ranh giới nhường lượt, cần gắn được timing và chuyển người nói trước. Không tự lấy cuối đoạn làm EOT.

## Kiểm tra cuối trên384mẫu giữ riêng

Mô hình chọn trước test: `qwen_lora`, `runs/natural_v8_lora_20261025_r3/best_adapter`. Nhãn automatic observed-floor/VAD, chưa có human gold; Việt1cuộc gọi, Anh13, Trung4. Chỉ là point estimates trên cohort nhỏ, không khẳng định chất lượng mọi hội thoại.

| Tiếng | Mẫu | Accuracy | F1 EOT | EOT nhầm/hold |
| --- | ---: | ---: | ---: | ---: |
| vi | 210 | 71.90% | 41.58% | 26.97% |
| en | 93 | 72.04% | 82.67% | 76.19% |
| zh | 81 | 79.01% | 87.94% | 88.24% |

Numeric target trên nhãn tự động: `False`.204prefix crawl chưa có nhãn nên accuracy/F1 vẫn=null. Không thay model serving hoặc quảng bá bản distill đạt90%.

Nguồn kiểm chứng Việt tiếp theo đã khảo sát: [ViYT-Diar](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar/tree/c66b118709d68c7538408d086232d850356db374), 100 bản ghi, tác giả công bố mốc diarization gán thủ công. Đã lưu dataset card và khóa revision, chưa tải audio/chạy classifier. Vì đây là official test, giữ riêng cho đánh giá. Mốc người nói không tự là nhãn semantic EOT: còn cần recipe handoff/pause, loại đoạn chồng lấn theo quy tắc khóa trước và ASR chỉ lấy lời đã xuất hiện. Không lấy nó làm train rồi gọi điểm trên cùng tập là test.
