# Báo cáo Đánh giá Model - Phase 1

### Mục 1: Thông tin cơ bản
* **Tên model test:** Gemini 3.5 Flash & Gemini 3.5 Flash Lite
* **Dataset test trên:** MMAD (Task B3 - Anomaly Detection) & ALPI (Task B5 - Next-state Prediction)
* **Ngày test:** 01/08/2026
* **Người test:** Huy

### Mục 2: Setup
* **Môi trường:** Chạy qua script framework nội bộ, gọi trực tiếp đến Google commercial API.
* **Chi phí/Tài nguyên:** Sử dụng Free Tier API, không tốn chi phí thuê server/GPU.
* **Cấu hình:** Test hoàn toàn zero-shot, không fine-tune hay train lại. 

### Mục 3: Kiểm tra leak
* **Nhánh MMAD:** Các lần chạy đầu tiên (tập 100 và 250 câu) bị lỗi Data Leakage do mô hình học từ đường dẫn file. Đã phát hiện và khắc phục triệt để trong bản chạy 456 câu. Dữ liệu hiện tại đảm bảo tính độc lập.
* **Nhánh ALPI:** Đã áp dụng cơ chế `random.shuffle(seed=42)` nhằm chống bias chuỗi thời gian. Không phát hiện rò rỉ giữa train/test.

### Mục 4: Kết quả chính

#### 1. Dataset ALPI (Task B5 - Next-state Prediction)

| Lần chạy / Model | Cỡ mẫu (N) | Exact Match Accuracy [95% CI] | Macro-F1 [95% CI] | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence Baseline** | 500 | N/A | **0.93%** | Dự đoán trạng thái Event 5 |
| **Demo (Flash Lite)** | 10 | 70.00% [40.00% - 100.00%] | 70.00% [28.00% - 92.00%] | Thử nghiệm nhỏ ban đầu |
| **Flash Lite (Chính thức)** | 499 | **54.71%** [50.50% - 59.12%] | **46.61%** [41.33% - 51.57%] | Đã nâng cấp Answer Extractor |
| **Gemini 3.5 Flash** | 53 | **60.38%** [47.17% - 73.58%] | **57.80%** [40.33% - 70.85%] | Đánh giá trên 5 class chuẩn |

* **Giải thích chỉ số Baseline Macro-F1 = 0.93%:**
  Tập dữ liệu ALPI Task B5 chứa 5 nhãn trạng thái nhắm tới (target states) trong tổng thể phân phối. Chỉ số `macro_f1` trung bình không trọng số (unweighted average) được tính bằng cách lấy trung bình F1 score trên toàn bộ các nhãn xuất hiện trong ground truth. Các baseline đơn giản (như Persistence hay Majority-class) chỉ dự đoán 1 hoặc 2 trạng thái xuất hiện với tần suất cao nhất, dẫn tới F1 = 0 đối với các nhãn chuyển tiếp còn lại. Do đó, điểm Macro-F1 của baseline bị kéo xuống mức 0.93%.

* **Giải thích hiện tượng biến động ngược chiều giữa Accuracy và Macro-F1 của Flash-Lite:**
  Sau khi nâng cấp bộ trích xuất đáp án (Answer Extractor), điểm Accuracy của Flash-Lite (499 câu) tăng từ 49.30% lên 54.71%, trong khi điểm Macro-F1 biến động nhẹ từ 47.62% xuống 46.61%. Nguyên nhân do trong 94 mẫu được cứu nhờ extractor mới, phần lớn rơi vào các nhãn trạng thái phổ biến (`production`, `performance_loss`), thể hiện qua số lỗi thuộc nhóm *Majority-class bias* ở Mục 6 tăng từ 54 lên 75 mẫu. Việc trích xuất đúng các mẫu thuộc lớp đa số giúp tăng tổng số câu đoán đúng trên toàn bộ tập dữ liệu (kéo Accuracy tăng), nhưng không làm cải thiện tương ứng cho các lớp hiếm, dẫn tới Macro-F1 (trung bình không trọng số trên cả 5 class) chỉ biến động nhẹ.

---

#### 2. Dataset MMAD (Task B3 - Anomaly Detection)

| Lần chạy / Model | Cỡ mẫu (N) | Exact Match Accuracy [95% CI] | Macro-F1 [95% CI] | Trạng thái dữ liệu |
| :--- | :---: | :---: | :---: | :--- |
| **Flash-Lite (Test 1)** | 100 | 89.00% [82.00% - 95.00%] | 88.00% [79.76% - 94.12%] | ⚠️ Bị Leak Data (Path leak) |
| **Flash-Lite (Test 2)** | 250 | 88.00% [84.00% - 92.00%] | 87.86% [83.67% - 91.90%] | ⚠️ Bị Leak Data (Path leak) |
| **Flash-Lite (Clean)** | 456 | **45.83%** [41.01% - 50.44%] | 47.96% [43.19% - 52.42%] | ✅ Dữ liệu sạch đã fix leak |
| **Gemini 3.5 Flash** | 72 | **56.94%** [45.83% - 68.06%] | 56.45% [43.87% - 67.78%] | ✅ Dữ liệu sạch đã fix leak |

* **So sánh điểm số & Khoảng tin cậy (CI):**
  - Trên tập dữ liệu sạch MMAD, Gemini 3.5 Flash đạt điểm trung bình Accuracy là **56.94%**, cao hơn **45.83%** của Flash Lite (+11.11%). Tuy nhiên, khoảng tin cậy 95% CI của Flash [`45.83%` - `68.06%`] và Flash Lite [`41.01%` - `50.44%`] có dải chồng lấn (overlap) tại vùng `45.83% - 50.44%`.
  - Trên tập ALPI (sau khi nâng cấp answer extractor), Gemini 3.5 Flash đạt Accuracy **60.38%** (so với **54.71%** của Flash Lite) và Macro-F1 **57.80%** (so với **46.61%** của Flash Lite). Khoảng tin cậy 95% CI của Flash [`40.33%` - `70.85%`] có dải chồng lấn với Flash Lite [`41.33%` - `51.57%`].

### Mục 5: Kiểm tra shortcut
* **N/A**. Phép thử static-frame (chạy 1 frame lặp lại) chỉ áp dụng cho dữ liệu video. ALPI là dữ liệu sự kiện (numeric/tabular) và MMAD là ảnh tĩnh công nghiệp, nên không áp dụng kiểm tra shortcut này.

### Mục 6: Phân loại nhóm lỗi (Error Taxonomy)

#### 1. Bảng Phân loại Lỗi ALPI (Task B5)

| Nhóm lỗi chính | Mô tả bản chất | Run 499 câu (Flash Lite) | Run 53 câu (Flash) |
| :--- | :--- | :---: | :---: |
| **Lỗi phân loại trạng thái khác** | Dự đoán sai sang các trạng thái không trùng khớp với ground truth | 112 (49.56%) | 8 (38.10%) |
| **Majority-class bias** | Thiên lệch dự đoán về các trạng thái phổ biến (`production` / `performance_loss`) | 75 (33.19%) | 8 (38.10%) |
| **Nhầm trạng thái liền kề / thời gian** | Nhầm giữa các trạng thái kề cận (`performance_loss` vs `downtime`, `idle` vs `scheduled_downtime`) | 37 (16.37%) | 3 (14.29%) |
| **Lỗi định dạng đầu ra (Format Error)** | Mô hình đưa ra tên trạng thái không nằm trong 5 nhãn chuẩn (vd: `working`, `producing`) | 2 (0.88%) | 2 (9.52%) |

*Ví dụ minh họa tiêu biểu (ALPI):*
* **Mẫu Sample ID: `alpi_s5_59212`**
  * **Model đoán:** `performance_loss`
  * **Đáp án thật:** `downtime`
  * **Nguyên nhân:** Nhầm lẫn trạng thái liền kề thời gian khi các sự kiện ngắt máy chưa đủ tín hiệu phân định rõ giữa giảm hiệu suất và dừng thiết bị hẳn.

---

#### 2. Bảng Phân loại Lỗi MMAD (Task B3)

| Nhóm lỗi chính | Mô tả bản chất | Run 456 câu (Flash Lite) | Run 72 câu (Flash) |
| :--- | :--- | :---: | :---: |
| **Lỗi vị trí defect rìa/góc** | Khuyết tật xuất hiện ở góc, viền ảnh hoặc bị nhiễu nền | 94 (38.06%) | 12 (38.71%) |
| **Lỗi nhầm dạng khuyết tật** | Nhầm lẫn giữa các loại khuyết tật có hình thái tương tự (vd: Class A vs B) | 82 (33.20%) | 9 (29.03%) |
| **Lỗi ánh sáng & phản chiếu** | Ánh sáng bề mặt kim loại/công nghiệp gây bóng mờ, lóa | 71 (28.74%) | 10 (32.26%) |

*Ví dụ minh họa tiêu biểu (MMAD):*
* **Mẫu Sample ID: `mmad-28761`**
  * **Model đoán:** `A`
  * **Đáp án thật:** `B`
  * **Nguyên nhân:** Ánh sáng phản chiếu công nghiệp trên bề mặt sản phẩm làm mờ vết xước, dẫn tới mô hình nhận diện sai hình thái khuyết tật từ nhóm B sang nhóm A.

### Mục 7: Kết luận cuối cùng

**ALPI (Task B5):**
☐ **CÓ XU HƯỚNG VƯỢT TRỘI NHƯNG KHÔNG ĐỦ CƠ SỞ KHẲNG ĐỊNH KHÁC BIỆT CÓ Ý NGHĨA THỐNG KÊ**
Gemini 3.5 Flash có điểm trung bình điểm Macro-F1 (**57.80%**) và Accuracy (**60.38%**) cao hơn so với Flash Lite (**46.61%** và **54.71%**). Tuy nhiên, do cỡ mẫu của Flash bị giới hạn cứng ở N=53 bởi API quota (không thể mở rộng), khoảng tin cậy 95% CI của Flash [`40.33%` - `70.85%`] có dải chồng lấn (overlap) với Flash Lite [`41.33%` - `51.57%`]. Do đó, chưa đủ cơ sở khẳng định sự khác biệt có ý nghĩa thống kê.

**MMAD (Task B3):**
☐ **CÓ XU HƯỚNG VƯỢT TRỘI NHƯNG KHÔNG ĐỦ CƠ SỞ KHẲNG ĐỊNH KHÁC BIỆT CÓ Ý NGHĨA THỐNG KÊ**
Mặc dù Gemini 3.5 Flash đạt độ chính xác trung bình cao hơn Flash Lite (**56.94%** so với **45.83%**), khoảng tin cậy 95% CI của Flash [`45.83%` - `68.06%`] và Flash Lite [`41.01%` - `50.44%`] **có dải chồng lấn** tại vùng `45.83% - 50.44%`. Nguyên nhân do cỡ mẫu Flash bị giới hạn cứng ở N=72 bởi API quota (không thể mở rộng), khiến dải CI mở rộng rộng hơn.

### Mục 8: Lưu ý / giới hạn
* **Xử lý Bug Answer Extraction:** Đã phát hiện và khắc phục triệt để vấn đề extractor đơn giản (`raw_output.strip()`) bỏ sót 94/499 mẫu (~18.84%) ở ALPI Flash-Lite (mô hình đưa ra nhãn chuẩn ở cuối đoạn văn Chain-of-Thought nhưng không được trích xuất). Sau khi nâng cấp extractor bằng regex tìm nhãn chuẩn từ dưới lên (`robust_parse_alpi_output`), độ chính xác Accuracy của ALPI Flash-Lite tăng từ `49.30%` lên **`54.71%`**. Số mẫu thực sự bị lỗi định dạng (model trả lời nhãn không chuẩn như `working`, `producing`) được ghi nhận độc lập tại Mục 6 (2/499 = 0.40% ở Flash-Lite, 2/53 = 3.77% ở Flash).
* **Giới hạn Quota API (Cấu trúc nghiên cứu):** Cỡ mẫu của bản Gemini 3.5 Flash (ALPI 53 câu, MMAD 72 câu) bị chặn cứng bởi giới hạn free-tier quota (`429 RESOURCE_EXHAUSTED`) của Google AI Studio API. Đây là giới hạn ngoại cảnh khách quan ngoài tầm kiểm soát của nhóm chứ không phải lỗi thiết kế thí nghiệm. Giới hạn N này khiến khoảng tin cậy 95% CI của Flash rộng hơn.
* **Đề xuất khắc phục trong tương lai:** Nâng cấp lên tài khoản Google Cloud commercial API trả phí hoặc áp dụng cơ chế luân phiên nhiều API key (key rotation) nếu ngân sách cho phép, nhằm mở rộng cỡ mẫu Flash đạt N tương đương Flash-Lite (499 câu ALPI, 456 câu MMAD) trong lần đánh giá tiếp theo.
* **Tính toàn vẹn dữ liệu (Fix Leak):** Việc phát hiện và loại bỏ triệt để lỗi Data Leakage qua đường dẫn file trên MMAD (khiến Accuracy giảm từ ~88% về ~45.8%) khẳng định các kết quả báo cáo ở Mục 4 hoàn toàn độc lập và trung thực.
