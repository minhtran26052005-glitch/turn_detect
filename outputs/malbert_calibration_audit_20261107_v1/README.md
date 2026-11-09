# Calibration DEV-only trên model v7 cố định

Hoàn thành kiểm chứng. Checkpoint 3731173, 11.941.122 tham số FP32. Không train, không thay weights, không TEST, không áp ngưỡng/default mới. Parent vẫn tạm dừng.

Ngưỡng cũ 0.1; ngưỡng chọn bằng calibration 0.03. Tiêu chí giữ nguyên: tối đa F1 của tiếng thấp hơn, rồi tổng F1; khi bằng nhau lấy ngưỡng đầu theo grid tăng dần. Chỉ một ngưỡng mới được chấm trên monitor; baseline monitor lấy đúng report trước của cùng weights.

Kết luận của lần kiểm này: operating point cũ góp phần gây bỏ sót. Trên ASR VI, F1 tăng10,43→22,88 và recall6,32→20,07 chỉ bằng ngưỡng; nhưng precision29,82→26,60, FP40→149 và FN vẫn215/269. Vì vậy chưa thể coi đây là sửa xong hoặc chứng minh model đủ dùng. Phép kiểm không áp dụng cho weights của nhánh ASR-adapted đã bị từ chối; không suy ra nó chữa được sự sụt điểm của nhánh đó.

Các ngưỡng mới từ 0,00001 tới 0,1 chạy actual predicted-reset riêng từng mức, không threshold lại trace cũ. Grid cũ >0,1 dùng lại calibration summaries cùng checkpoint. Mốc 0,1 chạy lại trên calibration và khớp confusion/metrics gốc. Không tối ưu theo monitor.

| Nguồn | Tiếng | F1 trước → sau | Precision trước → sau | Recall trước → sau | FP trước → sau | FN trước → sau |
|---|---|---|---|---|---|---|
| ALL | vi | 29.93 → 34.88 | 39.12 → 30.66 | 24.23 → 40.44 | 221 → 536 | 444 → 349 |
| ALL | en | 47.61 → 48.24 | 47.24 → 38.42 | 47.99 → 64.80 | 373 → 723 | 362 → 245 |
| asr | en | 39.05 → 44.69 | 52.33 → 40.74 | 31.14 → 49.48 | 82 → 208 | 199 → 146 |
| asr | vi | 10.43 → 22.88 | 29.82 → 26.60 | 6.32 → 20.07 | 40 → 149 | 252 → 215 |
| daily | en | 57.44 → 54.78 | 48.46 → 41.45 | 70.51 → 80.77 | 117 → 178 | 46 → 30 |
| daily | vi | 44.85 → 46.20 | 44.20 → 36.32 | 45.52 → 63.43 | 77 → 149 | 73 → 49 |
| factory | en | 54.81 → 54.68 | 54.68 → 43.44 | 54.95 → 73.76 | 92 → 194 | 91 → 53 |
| factory | vi | 45.24 → 48.00 | 50.89 → 40.00 | 40.71 → 60.00 | 55 → 126 | 83 → 56 |
| numeric/dialogue | en | 29.87 → 29.33 | 21.90 → 18.75 | 46.94 → 67.35 | 82 → 143 | 26 → 16 |
| numeric/dialogue | vi | 14.14 → 16.57 | 12.50 → 11.11 | 16.28 → 32.56 | 49 → 112 | 36 → 29 |

Tỷ lệ ở bảng là phần trăm. Nguồn ASR riêng nhỏ, không được coi là kiểm chứng độc lập đủ rộng. Nhãn punctuation silver; không human/audio gold, không đo độ trễ endpoint hay chất lượng dịch. Đây là chẩn đoán operating point trên một checkpoint, không chứng minh mô hình mới hoặc cách train nào tốt hơn. Không khẳng định ngưỡng tối ưu ngoài grid hữu hạn.

Kiểm chứng 160,776 inference calibration mới và 35,270 inference monitor; 158 canonical batch đơn, maxdelta 1.07288361e-06. Toàn bộ saved new predictions đã audit population/causalinput/future/reset/tokenhash/labels/confusion. Summary calibration cũ >0,1 được dùng lại, không có tuyên bố audit mới toàn bộ trace của các mức cũ. Parent full checkpoint và weights SHA không đổi. Report SHA 4330c69784d51e522ff70fb76a0ddab0d60cd9120a3871f13c1c46094cc784ca.

Full weights và traces ở SSH, không tải Windows/backup Drive. Lượt này không mở nhánh training mới.
