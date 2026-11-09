# INT4 student experiment — 2026-10-24

Material Passport: **VERIFIED: FULL DEV, WINDOWS, COLAB, ZIP**.

Mục tiêu: trọng số 4-bit và model nhỏ hơn INT8, accuracy và F1 EOT ≥90% trên frozen dev; không mở lại holdout đã dùng. Nguồn FP32 là checkpoint GRU96 distill trước đó, SHA256 `1d1b6ce737def64f1a18e9c090d2e32b0dc24394e8e4833b6829d7f61bc48ad0`. INT8 đối chứng SHA256 `846e6c9e606813b013d2574f148962a331645acf3984d04cc78839b8b075d182`.

Phương pháp: ONNX version converter 18→21; năm ma trận INT4 signed [-8,7], hai mã mỗi byte. Scale FP16 cho mỗi nhóm theo trục cuối; chọn trong hệ số [0,8;0,85;0,9;0,95;1,0] bằng MSE trọng số, không dùng activation dev để định lượng. Giữ bias/norm FP32. DequantizeLinear block và Cast đưa về FP32 lúc load; không có kernel GRU INT4. Seed42 cho kiểm tra kỹ thuật batch1/3/2, sequence8/24/256, thêm8padding; delta0.

Đã chấm đủ 17.833 dev cho bốn nhóm [64,32,16,8], cùng tokenizer và cùng lưới ngưỡng 181 điểm 0,05..0,95 như đối chứng. Chọn file ONNX nhỏ nhất đạt cả hai, nhóm32: **185,933 byte**, cả gói **412,682 byte**. Không có train/QAT/RL/crawl mới trong lượt này.

| Cấu hình | ONNX KB | Accuracy dev | F1 EOT dev | Cả hai ≥90% |
| --- | ---: | ---: | ---: | --- |
| int4_g64 | 176.999 | 93.3606% | 89.9183% | Không đạt |
| int4_g32 | 185.933 | 93.4335% | 90.0620% | Đạt |
| int4_g16 | 204.191 | 93.3887% | 89.9394% | Không đạt |
| int4_g8 | 240.703 | 93.3887% | 90.0076% | Đạt |

F1 tại ngưỡng INT8 0,545 là89,8236%; sau hiệu chỉnh dev0,430 đạt90,06195%. False EOT tăng744→864, missed EOT giảm404→307. Chỉ chênh0,06195 điểm phần trăm so với yêu cầu90, nên không khẳng định chắc chắn trên dữ liệu mới. Data/holdout limitations kế thừa nguyên bản distillation. Không coi cờ size_target_1/10 của evaluate_compressed là điều kiện lần này; budget lần này là nhỏ hơn328.761byte/555.479byte INT8.

Nguồn: [ONNX DQ21](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-21), [ORT INT4](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4). Receipts subprocess ghi exit_code thật; nguồn và frozen identities nằm trong `outputs/quant4_20261024/`. Mọi đánh giá là dev selection, không phải chứng nhận độc lập.

Hoàn tất kiểm tra **64/64**: model/ZIP hashes, hai nền tảng mỗi nền tảng đủ17.833 dev (Windows delta0, Colab2,22e-16), runner không Torch/HF, bốn cấu hình, marker/frozen bytes còn nguyên, notebook đã thực thi Colab, **35 unit tests pass**. Lần pytest đầu quét cả bản chụp mã nguồn trong outputs gây trùng tên module; `pytest.ini` giới hạn discovery vào `tests/` rồi chạy lại thành công. ZIP204,705byte, SHA256 `a6046ec6d65e7349442d8dfb1eb199391eff1907b04661f8b1b44e97870063f8`.
