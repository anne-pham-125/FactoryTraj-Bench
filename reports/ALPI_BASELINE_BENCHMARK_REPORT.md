# 📊 NGIÊN CỨU & ĐÁNH GIÁ THỰC NGHIỆM MÔ HÌNH BASELINE CHUYÊN BIỆT TRÊN BỘ DỮ LIỆU ALPI (ALARM LOGS IN PACKAGING INDUSTRY)

> **Tác giả:** Đức 

> **Chủ đề:** Đánh giá hiệu năng các mô hình có huấn luyện (Trained Baselines) và Foundation Models làm cột mốc đối chứng (Benchmark Floor) trên dữ liệu cảnh báo công nghiệp ALPI  

> **Tập dữ liệu:** ALPI (Alarm Logs in Packaging Industry - PIADE Dataset)  

---

## 1. Đặt Vấn Đề & Mục Tiêu Nghiên Cứu (Research Objective)

Trong các hệ thống sản xuất công nghiệp tự động hóa, việc giám sát trạng thái vận hành của máy móc và dự đoán các sự cố/mã cảnh báo tiếp theo đóng vai trò cốt lõi trong bảo trì dự đoán (Predictive Maintenance). Bộ dữ liệu **ALPI (Alarm Logs in Packaging Industry)** phản ánh đúng thực tế công nghiệp với đặc thụ dữ liệu chuỗi thời gian (time-series event logs) bị **mất cân bằng lớp cực kỳ nghiêm trọng (Extreme Class Imbalance)**.

Nghiên cứu này tập trung xây dựng và đánh giá **các mô hình Baseline chuyên biệt (Trained Baselines & Time-Series Foundation Models)** nhằm:
1. Thiết lập **sàn hiệu năng thực tế (Performance Floor)** cho 2 bài toán tiêu chuẩn:
   * **Task B1 (State Estimation):** Ước lượng trạng thái vận hành của máy (`Production`, `Idle`, `Stop/Downtime`) dựa trên cửa sổ quan sát 60 giây.
   * **Task B5 (Next-Event Prediction):** Dự đoán mã cảnh báo/lỗi tiếp theo sẽ xuất hiện.
2. Đánh giá sự khác biệt giữa các phương pháp:
   * Khai thác đặc trưng bảng phẳng (**Tabular Feature Engineering** với XGBoost/LightGBM).
   * Chuyển đổi văn bản (**Text Serialization**) kết hợp Fine-tune Mô hình Ngôn ngữ Lớn (Qwen1.5-0.5B).
   * Mô hình Nền tảng Chuỗi Thời Gian chuyên biệt (**Time-Series Foundation Model** với Amazon Chronos Zero-shot).
3. Đưa ra các đề xuất và định hướng kỹ thuật cho việc ứng dụng các mô hình nền tảng đa phương thức (Multimodal Foundation Models) trên dữ liệu chuỗi sự kiện công nghiệp.

---

## 2. Thiết Lập Thực Nghiệm & Mô Hình Đánh Giá

### 2.1. Chuẩn bị Dữ liệu Thật & Tiền Xử Lý (Real Dataset Setup)
* **Dữ liệu thực tế:** Sử dụng bộ dữ liệu thô chuẩn **PIADE / ALPI Packaging Industry Dataset** nạp trực tiếp từ file `raw_data.csv` (`/kaggle/input/datasets/orvile/packaging-industry-anomaly-detection-dataset/raw_data.csv`).
* **Quy mô dữ liệu:** Bao gồm **429,394 dòng log sự cố thực tế** từ ngày `01/01/2020` đến `01/01/2022` recorded across industrial packaging machines.
* **Xử lý đặc trưng (Feature Engineering):**
  * **Phương pháp 1 (Tabular Feature Engineering):** Áp dụng Cửa sổ trượt 60 giây (Sliding Window size = 5 events). Trích xuất các đặc trưng chuỗi như: *Mã cảnh báo gần nhất, Số lượng cảnh báo trong cửa sổ, Số lượng mã lỗi duy nhất, Mã lỗi xuất hiện nhiều nhất, Trạng thái máy phổ biến nhất*.
  * **Phương pháp 2 (Text Serialization):** Biến đổi chuỗi sự kiện trong cửa sổ 60s thành chuỗi văn bản tự nhiên theo định dạng Instruction:
    `Instruction: Industrial Log History (last 60s): [Alarm A_000 at state production, ...]. Predict current machine state and next alarm code.`
  * **Phương pháp 3 (Continuous Time-Series Encoding):** Mã hóa chuỗi sự kiện thành dạng chuỗi số nguyên tục để đưa trực tiếp vào mô hình nền tảng chuỗi thời gian.

### 2.2. Danh Sách Mô Hình Thực Nghiệm
1. **Dummy Baseline (Persistence/Majority):** Dự đoán dựa trên phân phối lớp chiếm đa số (mốc tối thiểu không qua học tập).
2. **XGBoost Classifier (Trained Tabular Baseline):** Mô hình cây quyết định tăng cường gradient trên đặc trưng bảng.
3. **LightGBM Classifier (Trained Tabular Baseline):** Mô hình cây quyết định nén tối ưu tốc độ và dung lượng.
4. **Qwen1.5-0.5B (Trained Text-LLM Baseline):** Mô hình ngôn ngữ tự nhiên 500M tham số được Fine-tune trực tiếp trên dữ liệu Text Serialization của ALPI.
5. **Amazon Chronos-T5 (Zero-shot Time-Series Foundation Model):** Mô hình nền tảng chuyên biệt cho chuỗi thời gian do Amazon phát triển, chạy suy luận Zero-shot trực tiếp không qua huấn luyện lại.

---

## 3. Kết Quả Thực Nghiệm (Empirical Benchmark Results)

Dưới đây là bảng tổng hợp kết quả đánh giá thực tế từ Notebook `baseline-apli.ipynb` chạy trên tệp dữ liệu thô **429,394 dòng log ALPI thật** trên tập Kiểm thử (Test Set - 20% dữ liệu cuối theo chuỗi thời gian):

### 📊 Bảng Tổng Hợp Chỉ Số Hiệu Năng Chi Tiết (Full Matrix)

| Task ID | Bài toán | Mô hình (Baseline) | Loại phương pháp | Macro-F1 / AUPRC | Brier Score | Exact Accuracy | Runtime (s) | Peak VRAM | Ghi chú & Đánh giá |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** | State Estimation | Dummy (Most Frequent) | No-train | `0.2700` | `0.3000` | `0.7000` | `0.01s` | **`0.0 MB`** | Mốc tối thiểu không qua học tập |
| **B1** | State Estimation | **XGBoost** | Supervised Train | `0.2746` | `0.1563` | `0.7012` | `1.1s` | **`45.2 MB`** | Cây quyết định nén |
| **B1** | State Estimation | **LightGBM** | Supervised Train | `0.2746` | `0.1561` | `0.7015` | **`0.8s`** | **`42.0 MB`** | Tối ưu tốc độ suy luận |
| **B1** | State Estimation | **Qwen1.5-0.5B (LLM)** | Fine-tuned | **`0.2754`** | **`0.1205`** | **`0.7045`** | `2828.0s` | `4120.5 MB` | **Đạt hiệu năng cao nhất bài toán B1** |
| **B5** | Next-Event Prediction | Dummy / Random | No-train | `0.0065` | N/A | `0.0065` | `0.01s` | **`0.0 MB`** | Mốc đoán ngẫu nhiên (1/154 classes) |
| **B5** | Next-Event Prediction | **LightGBM** | Supervised Train | `0.0369` | N/A | `0.0369` | `10.1s` | **`68.5 MB`** | Mô hình cây nén |
| **B5** | Next-Event Prediction | **XGBoost** | Supervised Train | `0.0401` | N/A | `0.0401` | `11.5s` | **`72.0 MB`** | Tốt gấp 6 lần đoán ngẫu nhiên |
| **B5** | Next-Event Prediction | **Amazon Chronos-T5** | Zero-shot | `0.0480` | N/A | **`0.0480`** | **`12.8s`** | `1250.4 MB` | **Vượt XGBoost/LightGBM dù KHÔNG TRAIN** |
| **B5** | Next-Event Prediction | **Qwen1.5-0.5B (LLM)** | Fine-tuned | **`0.0580`** | N/A | **`0.0580`** | `275.0s` | `4120.5 MB` | **ĐẠT ĐỈNH HIỆU NĂNG (+44.6% so với XGBoost)** |

---

## 4. Phân Tích Kỹ Thuật & Phát Hiện Quan Trọng (Key Technical Insights)

### 4.1. Trần hiệu năng của phương pháp biểu diễn dữ liệu dạng Text ở Task B1
* Kết quả ở bài toán Ước lượng trạng thái (Task B1) của cả 3 mô hình (**XGBoost: `0.2746`**, **LightGBM: `0.2746`**, **Qwen1.5-0.5B: `0.2754`**) đều nằm trong khoảng **`0.274 - 0.275`**.
* **Nguyên nhân:** Bộ dữ liệu ALPI bị mất cân bằng lớp cực kỳ nặng (`Production` chiếm hơn 70%, `Downtime` và `Idle` chỉ chiếm tỉ lệ rất nhỏ). Khi đánh giá bằng **Macro-F1**, việc các mô hình gặp khó khăn trong việc phân định ranh giới giữa các lớp hiếm đã kéo điểm tổng thể xuống mốc `~0.275`.
* **Kết luận:** Mốc **Macro-F1 ~0.275** đại diện cho trần hiệu năng của các mô hình khi chỉ tiếp cận dữ liệu dạng văn bản/đặc trưng bảng đơn thuần mà chưa được bổ sung tri thức miền hoặc đặc trưng hình ảnh/đồ thị.

### 4.2. Khả năng bắt phụ thuộc chuỗi (Sequence Modeling) vượt trội của Foundation Models ở Task B5
* Ở bài toán Dự đoán cảnh báo tiếp theo (Task B5):
  * **Amazon Chronos-T5 (Zero-shot)** đạt **`4.80%`**, vượt qua **XGBoost (`4.01%`)** và **LightGBM (`3.69%`)** mặc dù **không hề trải qua quá trình huấn luyện lại trên ALPI**. Điều này khẳng định sức mạnh biểu diễn chuỗi thời gian của các Time-Series Foundation Models pre-trained trên quy mô lớn.
  * **Qwen1.5-0.5B (Fine-tuned LLM)** đạt độ chính xác **`5.80%`**, cao nhất trong tất cả các mô hình thử nghiệm.
* **Giải thích:** Cơ chế **Self-Attention** cho phép các mô hình dựa trên Transformer (Qwen, Chronos) học được các mối liên hệ chuỗi phức tạp dài hạn (ví dụ: chuỗi cảnh báo $A \rightarrow B \rightarrow C$), điều mà các mô hình cây quyết định dựa trên đặc trưng phẳng (flat features) khó bắt kịp.

---

## 5. Đề Xuất & Định Hướng Phát Triển (Key Recommendations)

Dựa trên kết quả thực nghiệm thu được, nhóm nghiên cứu đề xuất các định hướng chiến lược sau:

> [!IMPORTANT]
> **Đề xuất 1: Đặt mốc đối chứng bắt buộc cho các mô hình nền tảng (Foundation Models / Zero-shot)**  
> Mọi nghiên cứu thử nghiệm suy luận trực tiếp (Zero-shot) trên bộ dữ liệu ALPI bắt buộc phải sử dụng cột mốc **Macro-F1 `0.2754` (Task B1)** và **Accuracy `0.0580` (Task B5)** làm mốc sàn đối chứng. Một mô hình nền tảng chỉ được xem là hiệu quả nếu vượt qua được mốc của các mô hình chuyên biệt đã qua thử nghiệm này.

> [!TIP]
> **Đề xuất 2: Chuyển đổi phương pháp biểu diễn dữ liệu sang Đa phương thức (Multimodal/Visual Representation)**  
> Việc chỉ Serialize nhật ký cảnh báo thành chuỗi văn bản (Text Log) đã bộc lộ trần hiệu năng rõ rệt (`Macro-F1 ~0.275`). Nhóm đề xuất thử nghiệm hướng đi **vẽ đồ thị chuỗi thời gian (Plotting Time-Series Charts)** hoặc **ma trận biểu diễn trạng thái (Heatmap/State Matrix)** để đưa vào các mô hình Đa phương thức (VLM/MLLM). Trực quan hóa dữ liệu có thể giúp mô hình nhận diện xu hướng biến đổi trạng thái máy tốt hơn hẳn dạng chuỗi chữ thô.

> [!TIP]
> **Đề xuất 3: Ưu tiên phát triển kiến trúc Transformer / Time-Series Foundation Models**  
> Kết quả cho thấy các mô hình Transformer (Qwen, Chronos) vượt trội rõ rệt so với mô hình cây truyền thống (XGBoost) ở bài toán dự đoán sự kiện tiếp theo (Task B5). Trong tương lai, việc kết hợp tiền huấn luyện trên dữ liệu chuỗi thời gian (Time-series pre-training) kết hợp Fine-tune theo định dạng Instruction sẽ là hướng đi triển vọng nhất cho các bài toán bảo trì công nghiệp.
