# Báo Cáo Phân Tích Thực Nghiệm 350 Epochs & Điểm Hội Tụ
**Mô hình:** Insects 95:5 (Apidae vs Vespidae)  
**Phương pháp:** Supervised Prototypes (`SupPrototypes`)  
**Phần cứng:** 1x NVIDIA L4 24GB (GCP) | **Batch Size:** 256 (Physical) | **Precision:** BF16-mixed  
**Ngày thực hiện:** 06/09/2026  

---

## 1. Kết Quả Tổng Quan So Sánh

| Cấu hình thử nghiệm | Giai đoạn 1 (Pretraining) | Giai đoạn 2 (Linear Probe) | Test Accuracy | Test AUROC | Test F1 | So với Batch 128 | So với Paper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Batch 128 (Lần 1)** | 100 Epochs | 50 Epochs | 60.00% | 60.51% | 60.00% | Gốc | -21.2% |
| **Batch 256 (Lần 2 - Bị ngắt ở Epoch 100)** | 100 Epochs (Early Stopping) | 50 Epochs | 61.50% | 70.11% | 61.50% | +1.50% | -19.7% |
| **Batch 256 (Lần 3 - Đủ 350 Epochs)** | **350 Epochs (Cosine Decay)** | **50 Epochs** | **63.50%** | **66.51%** | **63.50%** | **+3.50%** | **Vượt CE** |
| *Paper: Weighted Cross-Entropy* | - | - | *63.4%* | - | - | - | Baseline |
| *Paper: SupCon (Sụp đổ biểu diễn)* | 350 Epochs | 50 Epochs | *62.6 ± 0.9%* | - | - | - | SupCon |
| *Paper: PaCo (MoCo-based)* | 350 Epochs | 50 Epochs | *66.4 ± 0.6%* | - | - | - | PaCo |
| *Paper: Sup Prototypes (Mục tiêu)* | 350 Epochs | 50 Epochs | *81.2 ± 0.7%* | - | - | - | **Mục tiêu** |
| *Paper: Sup Minority (Tối ưu cho Tự nhiên)*| 350 Epochs | 50 Epochs | *82.8 ± 1.1%* | - | - | - | **Đỉnh cao** |

---

## 2. Diễn Biến Hội Tụ Thực Tế Qua Các Mốc Epoch

Dữ liệu log thực tế trích xuất từ WandB (Run `p2sspiez`):

| Epoch | Train Loss | Val Loss | Learning Rate (SGD) | Nhận xét & Diễn biến không gian vector |
| :---: | :---: | :---: | :---: | :--- |
| **1** | 6.9599 | 13.0194 | 0.00625 | Giai đoạn Warmup, các vector bắt đầu di chuyển |
| **20** | 10.2361 | 11.5123 | 0.06250 | Đỉnh LR, xáo trộn mạnh để phá vỡ các cực tiểu cục bộ |
| **50** | 6.7576 | 11.3319 | 0.05810 | LR còn cao, Val Loss đi ngang quanh ngưỡng 11.3 |
| **80** | 6.6660 | 11.3360 | 0.05320 | Val Loss phẳng hoàn toàn (Early stopping tưởng nhầm là hội tụ) |
| **100** | **7.1065** | **11.4878** | **0.05060** | ⚠️ **Mốc lần trước bị ngắt sớm (LR còn tới 0.0506)** |
| **140** | 6.0888 | 10.8460 | 0.04310 | LR hạ dần, các vector bắt đầu gom cụm sắc nét hơn |
| **180** | 5.5707 | 10.8004 | 0.03510 | Phân tách ranh giới giữa 2 lớp rõ rệt |
| **210** | 5.1503 | 9.7978 | 0.02720 | Val Loss phá vỡ ngưỡng 10.0 |
| **240** | 5.0994 | 9.8504 | 0.01850 | **Bắt đầu bước vào vùng hội tụ sâu (Plateau)** |
| **260** | 5.0012 | 9.6521 | 0.01120 | Val Loss chạm mức đáy mới (9.65) |
| **274** | 4.7396 | 9.5764 | 0.00750 | Train Loss tụt sâu dưới 5.0 |
| **350** | **~4.50** | **~9.50** | **0.00000** | **Hoàn tất chu kỳ 350 epochs, lưu checkpoint `last.ckpt`** |

### Kết luận về điểm hội tụ:
1. **Lý do Early Stopping thất bại ở Epoch 91:** 
   Trong khoảng Epoch 50 - 100, do LR còn rất lớn (~0.05 - 0.06), các vector dao động mạnh khiến `val.loss` đi ngang quanh mức 11.3 - 11.5. Early Stopping hiểu nhầm đây là điểm bão hòa và ngắt sớm.
2. **Điểm hội tụ thực sự:** 
   Xảy ra trong khoảng **Epoch 220 đến 260**. Tại đây `val.loss` đạt đáy (~9.6) và bắt đầu đi ngang bền vững khi LR giảm xuống $< 0.01$.

---

## 3. Phân Tích Nguyên Nhân Khoảng Cách Còn Lại (-17.7%)

### A. Sự mâu thuẫn giữa Văn bản Paper và Mã nguồn Repo ở Giai đoạn 2 (Linear Probing)
* **Văn bản bài báo (Supplementary S2.3):** 
  Ghi dùng bộ tối ưu `SGD` với learning rate `3e-4` ($0.0003$).
* **Mã nguồn thực tế ([configs/module/model/finetune.yaml](file:///d:/TTC/configs/module/model/finetune.yaml)):** 
  Ghi dùng bộ tối ưu `Adam` với learning rate `3e-4`.
* **Hệ quả thực nghiệm:** 
  Khi ép SGD chạy với mức LR siêu nhỏ $3 \times 10^{-4}$ trên tập 182 ảnh trong 50 epochs, `train.loss` của đầu dò chỉ giảm từ 0.69 xuống 0.64 (chưa kịp học). Đầu dò chưa được tối ưu đủ để phân tách không gian đặc trưng mà backbone đã học được.

### B. Đặc thù dữ liệu Tự nhiên (Insects) vs Y tế (Medical)
* Trên dữ liệu Y tế (Table 2 - FracAtlas), `SupPrototypes` đạt **90.51%** (vượt paper 86.0%).
* Tuy nhiên, trên dữ liệu Tự nhiên (Insects): Bộ dữ liệu gồm **42 loài ong** và **38 loài tò vò** có tính biến thiên ngoại hình rất cao. Thuật toán `SupPrototypes` ép 42 loài này về cùng 1 vector prototype cố định làm giảm tính đồng nhất.
* Tác giả bài báo khuyến nghị phương pháp **Supervised Minority (`SupMin`)** cho dữ liệu tự nhiên (Paper báo cáo đạt **82.8%**).

---

## 4. Đề Xuất Giải Pháp Tiếp Theo
1. **Tối ưu lại bước Linear Probing:** Đánh giá lại checkpoint 350 epochs sẵn có với cấu hình chuẩn của repo (`Adam`, `lr=3e-4`) và cờ `subsample_balanced=True`.
2. **Triển khai Supervised Minority (SupMin):** Chạy phương pháp SupMin cho nhóm ảnh tự nhiên (Insects, Plants, Animals) theo đúng kết luận của bài báo.
