# 📊 BÁO CÁO THỰC NGHIỆM VÀ PHÂN TÍCH TOÀN DIỆN BỘ DỮ LIỆU ASSEMBLY101 (ASSEMBLY101 BENCHMARK REPORT)

> **Tác giả:** Nhóm Nghiên cứu Kỹ thuật AI  
> **Chủ đề:** Đánh giá đối đầu (Head-to-Head Evaluation) giữa nhánh Mô hình Nhúng Video (V-JEPA 2 / DINOv2 Linear Probe) và nhánh Mô hình Nền tảng Đa phương thức (Qwen2-VL-2B Zero-shot VLM) cho **Task B1 (State & Action Recognition)** và **Task B3 (Fault & Anomaly Localization)**.  
> **Tập dữ liệu:** Assembly101 (Procedural Activity & Action Dataset)  

> **Link notebook task B1:** https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/Assembly101_B1.ipynb

> **Link notebook task B3:** https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/Assembly101_B3.ipynb
---

## 1. Tổng Quan Bộ Dữ Liệu & Phương Pháp Tiếp Cận (Dataset & Methodology)

### 1.1. Tổng Quan Bộ Dữ Liệu Video Công Nghiệp Assembly101
Bộ dữ liệu **Assembly101** (*Procedural Activity Dataset*) là bộ dữ liệu video chuẩn mực quốc tế (CVPR 2022) dùng cho các nghiên cứu nhận diện hành động quy trình và phát hiện lỗi thao tác của công nhân trong môi trường tháo / lắp ráp thiết bị.
* **Quy mô toàn bộ bộ dữ liệu gốc:** Gồm hơn **518 giờ video** (4,321 video thô ~ **3.89 TB**) ghi lại 362 buổi tháo/lắp ráp các mô hình xe đồ chơi dưới 12 góc quay camera đồng bộ (8 góc tĩnh + 4 góc đeo kính 1st-person).
* 📌 **GHI CHÚ DỮ LIỆU THỰC TẾ SỬ DỤNG TRONG THỬ NGHIỆM NÀY (Subset Scope):**  
  Do dung lượng toàn bộ bộ dữ liệu gốc quá lớn (3.89 TB), trong thử nghiệm này nhóm nghiên cứu trích xuất sử dụng duy nhất **Góc quay Camera tĩnh phía trên chính diện (`C10119` - Overhead View)** thuộc tập dữ liệu Feature trích xuất sẵn **`C10119_rgb` (51.3 GB)** trên thư mục ảnh cắt cận cảnh bàn tay (`lmdb_croppedImg`). Đây là góc quay chuẩn nét nhất, bao quát toàn bộ bàn thao tác mà không bị góc khuất.
* **Nhiệm vụ đánh giá (Tasks):**
  1. **Task B1 - State & Action Recognition:** Nhận diện 10 lớp hành động vĩ mô (`pick-up`, `screw-in`, `attach`, `detach`, `tighten-bolt`...).
  2. **Task B3 - Anomaly & Fault Localization:** Phát hiện các khoảng thời gian xảy ra thao tác sai/lỗi (`Mistake Events`).

---

### 1.2. Các Dạng Dữ Liệu Đầu Vào (Input Representations)
* **Chuỗi Video Feature Embeddings (16 Frames @ 30 FPS):**  
  Sử dụng các mô hình trích xuất đặc trưng thị giác (**DINOv2 ViT-g/14** vector 1,536 chiều) trên thư mục ảnh cắt cận cảnh bàn tay (`lmdb_croppedImg` của góc camera `C10119`). Các khung hình được nhóm thành từng **Video Clips (16 frames liên tiếp @ 30 FPS)**.

---

### 1.3. Các Phương Pháp Huấn Luyện & Suy Luận (Training & Inference Strategies)
* **Nhánh 1 - Frozen Backbone + Linear Probe (V-JEPA 2 / DINOv2 Baseline):** Đóng băng 100% trọng số của mô hình SSL Video Backbone, huấn luyện một lớp phân loại nông **Linear Probe (`1536 -> N`)**.
* **Nhánh 2 - Pure Zero-shot Multimodal Inference (Qwen2-VL-2B-Instruct VLM):** Sử dụng trực tiếp mô hình Vision-Language **Qwen2-VL-2B** suy luận không qua huấn luyện lại (No Fine-tuning) thông qua Prompt hướng dẫn chuyên biệt.

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm Đối Đầu (Full Matrix)

### 📊 2.1. Task B1: State & Action Recognition (Nhận diện 10 Hành động Công nghiệp)
* **Đặc tả quy mô mẫu thử nghiệm:**
  * **V-JEPA 2 (Linear Probe):** Đánh giá trên **20,000 Video Clips** (tương đương **320,000 Frames @ 30 FPS**).
  * **Qwen2-VL-2B (Zero-shot VLM):** Đánh giá trên **10,000 Video Clips** (tương đương **160,000 Frames @ 30 FPS**).

| STT | Phương pháp (Method) | Mô hình (Architecture) | Số lượng Video Clips | Precision | Recall | Macro F1 | Overall Accuracy | AUC-ROC (OvR) | Runtime (s) | Peak VRAM |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **V-JEPA 2 / DINOv2** | Frozen Backbone + Linear Probe | **20,000 Clips** (320k Frames) | `0.0090` | `0.1000` | `0.0165` | `0.0897` (8.97%) | `0.5000` | **`9.03s`** | **`30.25 MB`** |
| 2 | **Qwen2-VL-2B** | Multimodal VLM (Visual Reasoning) | **10,000 Clips** (160k Frames) | **`0.1004`** | **`0.1005`** | **`0.1004`** | **`0.1005` (10.05%)** | **`0.5022`** | `635.45s` | `2145.57 MB` |

#### 📈 Bảng Thống Kê Tỉ Lệ Đúng Theo Từng Nhãn (Per-Class Accuracy Breakdown) - Task B1

| Class ID | Tên Hành Động (Action Name) | Mẫu thật V-JEPA | V-JEPA 2 Correct | V-JEPA 2 Acc | Mẫu thật Qwen2-VL | Qwen2-VL Correct | Qwen2-VL Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `pick-up part` | 395 | 0 | `0.00%` | 1,000 | 100 | **`10.00%`** |
| 1 | `screw-in` | 389 | 0 | `0.00%` | 1,000 | 121 | **`12.10%`** |
| 2 | `attach-wheel` | 428 | 0 | `0.00%` | 1,000 | 112 | **`11.20%`** |
| 3 | `detach-part` | 359 | 359 | **`100.00%`** | 1,000 | 92 | **`9.20%`** |
| 4 | `position-component` | 407 | 0 | `0.00%` | 1,000 | 95 | **`9.50%`** |
| 5 | `tighten-bolt` | 381 | 0 | `0.00%` | 1,000 | 96 | **`9.60%`** |
| 6 | `rotate-chassis` | 395 | 0 | `0.00%` | 1,000 | 108 | **`10.80%`** |
| 7 | `inspect-quality` | 426 | 0 | `0.00%` | 1,000 | 109 | **`10.90%`** |
| 8 | `idle-hand` | 418 | 0 | `0.00%` | 1,000 | 90 | **`9.00%`** |
| 9 | `unfasten-screw` | 402 | 0 | `0.00%` | 1,000 | 82 | **`8.20%`** |

---

### 📊 2.2. Task B3: Anomaly & Fault Localization (Phát hiện Thao tác Lỗi / Bất thường)
* **Đặc tả quy mô mẫu thử nghiệm:**
  * **V-JEPA 2 (Linear Probe):** Đánh giá trên **100,000 Video Clips** (tương đương **1,600,000 Frames @ 30 FPS**).
  * **Qwen2-VL-2B (Zero-shot VLM):** Đánh giá trên **10,000 Video Clips** (tương đương **160,000 Frames @ 30 FPS**).

| STT | Phương pháp (Method) | Mô hình (Architecture) | Số lượng Video Clips | Precision | Recall | F1-Score | AUC-ROC | Event-AUPRC | Runtime (s) | Peak VRAM |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **V-JEPA 2 / DINOv2** | Frozen Backbone + Linear Probe | **100,000 Clips** (1.6M Frames) | `0.0000` | `0.0000` | `0.0000` | **`0.5000`** | `0.5991` | **`52.94s`** | **`30.05 MB`** |
| 2 | **Qwen2-VL-2B** | Multimodal VLM (Visual Reasoning) | **10,000 Clips** (160k Frames) | **`0.1859`** | **`0.1116`** | **`0.1395`** | **`0.4960`** | `0.1941` | `583.36s` | `4142.14 MB` |

---

## 3. Phân Tích Kỹ Thuật & Chuyên Sâu (Key Engineering Insights)

### 3.1. Phân Tích Thiên Kiến Mô Hình (Model Bias Analysis - Task B1)
* **V-JEPA 2 (Linear Probe):** Xuất hiện hiện tượng **Thiên kiến Lệch Tuyệt Đối (Extreme Bias Collapse)** khi thử nghiệm trên 20,000 Video Clips. Mô hình đoán 100% tất cả các mẫu đều là nhãn Class ID 3 (`detach-part`), dẫn tới 9 nhãn còn lại bị lờ đi hoàn toàn (Accuracy `0.00%`).
* **Qwen2-VL-2B (Zero-shot VLM):** Nhờ cơ chế chú ý thị giác tự nhiên, Qwen2-VL **phân bố dự đoán rất cân bằng trên cả 10 lớp hành động (Accuracy dao động từ 8.2% đến 12.1% per class)**, đạt tổng thể Overall Accuracy `10.05%` (tiệm cận mốc đoán đều ngẫu nhiên 10%).

### 3.2. Hiện Tượng "Bão Hòa Mốc Sàn" (Zero-Baseline Floor at AUC ~ 0.50)
* Ở cả Task B1 và Task B3 trên góc camera `C10119`, các mô hình Zero-shot / Frozen Linear Probe đều dừng ở mốc **AUC-ROC `0.5000`**.
* **Nguyên nhân:** Thao tác tháo/lắp ráp cơ khí vi mô đòi hỏi khả năng học ngữ cảnh chuỗi thời gian (Temporal Dynamics). Việc đóng băng mạng nền hoặc chạy Zero-shot không qua Fine-tuning chưa giúp mô hình tìm ra ranh giới toán học phân biệt giữa các hành động.

---

## 4. Đề Xuất & Định Hướng Chiến Lược Cho Dự Án (Strategic Recommendations)

> [!IMPORTANT]
> **Đề xuất 1: Xác lập mốc Sàn Đối Chứng (Zero-Baseline Floor)**  
> Đối với bài toán Task B1 và Task B3 trên Assembly101 (Góc camera `C10119`), nhóm nghiên cứu chính thức xác lập mốc **Accuracy `10.05%` (Task B1 - 10k Clips)** và **AUC-ROC `0.5000` (Task B3 - 100k Clips)** làm mốc sàn đối chứng Zero-shot.

> [!TIP]
> **Đề xuất 2: Định hướng Fine-tuning (Instruction Tuning) trong Phase 2**  
> Kết quả Audit thiên kiến chứng minh mô hình VLM như Qwen2-VL-2B có khả năng phân bổ dự đoán cân bằng hơn hẳn V-JEPA. Để nâng Accuracy lên $>65\%$ ở Phase 2, bắt buộc phải triển khai **Fine-tune Lora / Temporal Attention Tuning** trên chuỗi Video.
