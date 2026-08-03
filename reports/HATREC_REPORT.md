
Báo cáo tổng hợp và phân tích chuyên sâu các chỉ số thực nghiệm thu được từ quá trình đánh giá độc lập các mô hình Vision-Language Models (VLMs) trên Notebook Kaggle (`test-models-with-hatrec.ipynb`) đối với bộ dữ liệu **HATRec Industrial Assembly Video Dataset** (546 video clips, 7 lớp thao tác lắp ráp công nghiệp).

---
- Môi trường test: Kaggle GPU T4 x 2 (16GB VRAM x 2)
- Người test: Đức
- Dataset: HATREC (546 videos)
- Bài toán: Xác định hành động của công nhân trong video, phát triển từ báo cáo chạy trên Cosmos3-nano của anh Sơn
- Link notebook: https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/HATREC_VLMs.ipynb
## 1. BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM CHÍNH THỨC


| Mô Hình (Model) | Chế Độ Đánh Giá (Test Mode) | Tổng Video | Dự Đoán Đúng | Độ Chính Xác (%) | Tổng Thời Gian | Latency Trung Bình | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2-VL-2B-Instruct** | 🎬 Dynamic Native Video | 546 | 98 | **17.95%** | 493.13s (~8.22 phút) | 0.90s / video | **4.23 GB** |
| **Qwen2-VL-2B-Instruct** | 🛑 Static-Frame (Đóng Băng 16x) | 546 | 105 | **19.23%** ⬆️ | 1403.31s (~23.39 phút) | 2.57s / video | **8.25 GB** |
| **Qwen2-VL-7B-Instruct (4-bit)** | 🎬 Dynamic Native Video | 546 | 82 | **15.02%** | 1009.82s (~16.83 phút) | 1.85s / video | **13.46 GB** |
| **Qwen2-VL-7B-Instruct (4-bit)** | 🛑 Static-Frame (Đóng Băng 16x) | 546 | 61 | **11.17%** ⬇️ | 2246.84s (~37.45 phút) | 4.11s / video | **6.68 GB** |
| **LLaVA-1.5-7B (4-bit)** | 🎬 Dynamic Native Video | 546 | 78 | **14.29%** | 896.76s (~14.95 phút) | 1.64s / video | **12.35 GB** |
| *Mốc Đoán Mò Toán Học (Pure Random)* | *Random Chance Baseline (1/7)* | *546* | *78* | ***14.28%*** | *-* | *-* | *-* |

---

## 2. PHÂN TÍCH KẾT QUẢ

### 2.1. Qwen2-VL-2B-Instruct: Static-frame cho kết quả nhỉnh hơn Dynamic Native Video

Ở phiên bản **Qwen2-VL-2B-Instruct**, chế độ **Static-Frame (đóng băng 16 khung hình)** đạt **19.23%**, cao hơn nhẹ so với **Dynamic Native Video** (**17.95%**), chênh lệch **+1.28 điểm phần trăm**.

Kết quả này cho thấy, trong thiết lập thử nghiệm hiện tại, mô hình 2B chưa tận dụng tốt tín hiệu chuyển động của video. Việc “đóng băng” đầu vào có thể đã làm giảm phần nhiễu đến từ biến thiên thời gian, từ đó giúp mô hình dựa nhiều hơn vào các dấu hiệu tĩnh trong khung hình. Tuy nhiên, mức chênh lệch còn nhỏ, nên nên diễn giải đây là **một xu hướng đáng chú ý**, chưa đủ để khẳng định chắc chắn về cơ chế suy luận của mô hình.

### 2.2. Qwen2-VL-7B-Instruct (4-bit): Dynamic Native Video tốt hơn Static-frame

Với **Qwen2-VL-7B-Instruct (4-bit)**, chế độ **Dynamic Native Video** đạt **15.02%**, trong khi **Static-Frame** chỉ đạt **11.17%**, giảm **3.85 điểm phần trăm**.

Điều này cho thấy ở quy mô 7B, mô hình có vẻ **phụ thuộc nhiều hơn vào thông tin động** trong video so với bản 2B. Khi đầu vào bị làm đứng hình, hiệu năng giảm rõ rệt, cho thấy tín hiệu thời gian có thể đang đóng vai trò quan trọng hơn trong quyết định dự đoán. Dù vậy, kết luận này vẫn cần được xem là **suy luận từ một bộ thí nghiệm đơn lẻ**, và sẽ thuyết phục hơn nếu có thêm nhiều lần chạy, nhiều seed, hoặc thêm mô hình trung gian để đối chiếu.

### 2.3. LLaVA-1.5-7B (4-bit): Hiệu năng xấp xỉ mức đoán ngẫu nhiên

Mô hình **LLaVA-1.5-7B (4-bit)** đạt **14.29% (78/546)**, gần như trùng với mức **1/7 = 14.28%** của bài toán 7 lớp.

Kết quả này cho thấy mô hình zero-shot này chưa nắm được các đặc trưng đủ mạnh để phân biệt chính xác các thao tác lắp ráp trong HATRec. Nói cách khác, ở cấu hình hiện tại, mô hình hoạt động **gần như tương đương đoán ngẫu nhiên**, và chưa tạo ra giá trị rõ ràng cho bài toán phân loại hành động chuyên biệt trong môi trường công nghiệp.

---

## 3. KẾT LUẬN & ĐỀ XUẤT

Từ các kết quả trên, có thể rút ra rằng các mô hình VLM zero-shot tổng quát hiện tại **chưa phù hợp để triển khai trực tiếp** cho bài toán nhận diện hành động lắp ráp công nghiệp trên HATRec. Dù một số mô hình có thể đạt mức trên ngẫu nhiên nhẹ, hiệu năng vẫn còn thấp và chưa đủ ổn định để đáp ứng yêu cầu thực tế.

Vì vậy, hướng đi khả thi hơn là:

* **Fine-tune** mô hình trên dữ liệu đúng miền công nghiệp;
* hoặc sử dụng một pipeline chuyên biệt hơn như **Hybrid YOLOv8 + LSTM** để khai thác tốt hơn cả thông tin không gian lẫn chuỗi thời gian.

Nếu mục tiêu là ứng dụng thực tế trong nhà máy, giải pháp chuyên biệt sau khi được huấn luyện phù hợp vẫn là lựa chọn đáng tin cậy hơn so với việc dùng các mô hình zero-shot VLM đa năng.
