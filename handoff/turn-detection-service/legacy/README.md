# Vietnamese INT4 student

Quy trình nhánh student GRU 96, weight-only INT4 QAT. Cần teacher ONNX, train/dev, cache teacher và vocabulary của nhánh cũ. Kết quả DEV đã lưu tại `results/vi_student_legacy.json`; nhánh này không dùng checkpoint hay input contract cycle 75.

```bash
pip install -r requirements.txt -r legacy/requirements.txt

python -m legacy.turn_v2.distill_teacher --teacher "$TEACHER" --data "$DATA" --output "$CACHE"
python -m legacy.turn_v2.distill --teacher "$TEACHER" --data "$DATA" \
  --cache "$CACHE" --vocabulary "$VOCAB" --output "$STUDENT" --hidden-size 96
python -m legacy.turn_v2.qat_student --student "$STUDENT" --data "$DATA" \
  --cache "$CACHE" --vocabulary "$VOCAB" --output "$QAT" --group-size 64
python -m legacy.turn_v2.export_qat_student --student "$QAT" --output "$INT4" --dev "$DATA/dev.jsonl"
```

`DATA` cần `train.jsonl`, `dev.jsonl`, `manifest.json` và `holdout_consumed.json` của nhánh cũ. `VOCAB`, `CACHE`, `STUDENT`, `QAT` và `INT4` là các đường dẫn chưa tồn tại khi tạo mới.
