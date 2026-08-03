# 📊 BÁO CÁO THỰC NGHIỆM VÀ PHÂN TÍCH TOÀN DIỆN BỘ DỮ LIỆU ÂM THANH MIMII (MIMII AUDIO BENCHMARK REPORT)

> **Tác giả:** Đức

> **Chủ đề:** Đánh giá đa phương pháp (Ablation Study) trên bộ dữ liệu âm thanh công nghiệp MIMII làm mốc sàn đối chứng (Benchmark Floor) cho bài toán phát hiện bất thường (Task B3).
  
> **Tập dữ liệu:** MIMII (Malfunctioning Industrial Machine Investigation and Inspection)  

> **Link notebook**

- https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/MIMII_fan.ipynb
- https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/MIMII_pump.ipynb
- https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/MIMII_slide.ipynb
- https://github.com/anne-pham-125/FactoryTraj-Bench/blob/main/notebooks/MIMII_valve.ipynb
---

## 1. Tổng Quan Bộ Dữ Liệu & Phương Pháp Tiếp Cận (Dataset & Methodology)

### 1.1. Tổng Quan Bộ Dữ Liệu Âm Thanh Công Nghiệp MIMII
Bộ dữ liệu **MIMII** (*Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*) là bộ dữ liệu chuẩn mực quốc tế dùng cho các nghiên cứu phát hiện lỗi và bất thường trong môi trường sản xuất tự động hóa.
* **Quy mô dữ liệu:** Bao gồm hơn **26,000 tệp âm thanh `.wav`** (thời lượng 10 giây/file, tần số lấy mẫu 16kHz/48kHz) được ghi âm trực tiếp từ mảng micrô đa hướng đặt xung quanh các thiết bị.
* **4 Loại máy móc công nghiệp (Machine Types):**
  1. `FAN` (Quạt hút/quạt làm mát công nghiệp): Lỗi mất cân bằng cánh quạt, bám bẩn.
  2. `PUMP` (Bơm nước/bơm áp lực): Lỗi rò rỉ van, tụt áp, hỏng phớt.
  3. `SLIDER` (Thanh trượt/băng tải tự động): Lỗi kẹt ma sát, lệch đường ray.
  4. `VALVE` (Van đóng mở công nghiệp): Lỗi rò rỉ khí/chất lỏng, kẹt ti van.
* **Đặc điểm mất cân bằng lớp (Class Imbalance):** Phản ánh đúng thực tế nhà máy với số lượng mẫu chạy bình thường (`Normal - Label 0`) chiếm đa số tuyệt đối (75–80%) và mẫu bị lỗi/sự cố (`Abnormal - Label 1`) chỉ chiếm tỉ lệ nhỏ (20–25%).

---

### 1.2. Các Dạng Dữ Liệu Đầu Vào (Input Representations)
Trong nghiên cứu này, tín hiệu âm thanh thô (`.wav`) được chuyển đổi qua 2 dạng biểu diễn chính để đưa vào mô hình:
1. **Vector Đặc Trưng Âm Thanh 1D (Audio Feature Vector):**
   * Sử dụng các mô hình trích xuất đặc trưng âm thanh tiền huấn luyện (**Whisper Audio Encoder** tạo vector 384 chiều, **CLAP Encoder** tạo vector 512 chiều) bằng phương pháp nén trung bình theo thời gian (Mean-pooling).
2. **Ảnh Phổ Tần Số 2D (Mel-Spectrogram Image):**
   * Biến đổi tín hiệu sóng âm 1D thời gian thành hình ảnh ma trận phổ tần số **Log-Mel Spectrogram** kích thước $128 \times 128$ pixel. Dạng này biến bài toán xử lý âm thanh thành bài toán nhận dạng mẫu thị giác (Visual Pattern Recognition).

---

### 1.3. Các Phương Pháp Huấn Luyện (Training Strategies)
Để đánh giá toàn diện, 5 phương pháp được chia thành 3 nhóm chế độ huấn luyện:
* **Chế độ Giám sát (Supervised Training - 80/20 Split):** Mô hình được học trên cả mẫu Normal và Abnormal với tỉ lệ 80% Train - 20% Test. (Đại diện: *Whisper + XGBoost*, *Mel-Spectrogram + ResNet18*).
* **Chế độ Không giám sát (Unsupervised Anomaly Detection - Normal Only):** Mô hình chỉ được học trên duy nhất tập mẫu Normal để nắm bắt đặc đặc tính "âm thanh chuẩn". Khi gặp âm thanh lạ ở tập Test, mô hình sẽ tính toán mức độ sai lệch/tái tạo lỗi (Reconstruction Error) để cảnh báo. (Đại diện: *CLAP + One-Class SVM*, *Audio Autoencoder*).
* **Chế độ Suy luận Trực tiếp (Zero-shot Inference):** Mô hình Nền tảng Đa phương thức (Multimodal VLM) nhận trực tiếp ảnh phổ Mel-Spectrogram và suy luận nhãn không qua quá trình huấn luyện lại. (Đại diện: *Mel-Spectrogram + Qwen2-VL*).

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm Trên 4 Loại Máy (Full Matrix)

Tất cả các thử nghiệm được tiến hành trên 4 loại máy móc công nghiệp độc lập (`FAN`, `PUMP`, `SLIDER`, `VALVE`) với mẫu dữ liệu 1/4 (Subsample 25% chống OOM RAM) trên môi trường GPU T4.

### 📊 Bảng Tổng Hợp Chỉ Số AUC-ROC Toàn Diện

| STT | Phương pháp (Method) | Dạng dữ liệu (Input Type) | Chế độ huấn luyện (Training Mode) | FAN (Quạt) | PUMP (Bơm) | SLIDER (Băng tải) | VALVE (Van) | Trung bình (Mean) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Whisper + XGBoost** | Audio Feature Vector | Supervised (80/20) | `0.8719` | `0.9467` | `0.9823` | `0.9254` | **`0.9316`** |
| 2 | **CLAP + One-Class SVM** | CLAP Audio Vector | Unsupervised (Normal) | `0.5169` | `0.5079` | `0.5332` | `0.4961` | **`0.5135`** |
| 3 | **Spectrogram + ResNet18** | Mel-Spectrogram Image | Supervised Vision | **`0.9908`** | **`0.9704`** | **`1.0000`** | **`0.9853`** | **`0.9866`** |
| 4 | **Audio Autoencoder** | Audio Feature Vector | Unsupervised Anomaly | `0.5938` | `0.7131` | `0.8235` | `0.5765` | **`0.6767`** |
| 5 | **Spectrogram + Qwen2-VL** | Mel-Spectrogram Image | Zero-shot Multimodal | `0.4950` | `0.5029` | `0.4944` | `0.4917` | **`0.4960`** |

---

## 3. Phân Tích Kỹ Thuật & Chuyên Sâu (Key Engineering Insights)

### 3.1. Sự Thống Trị Tuyệt Đối Của Phương Pháp Thị Giác (Spectrogram + ResNet18 - Mean AUC: 0.9866)
* **Kế thừa đặc trưng thị giác (Spatial Visual Shortcut):** Âm thanh bị lỗi (Abnormal) của các thiết bị công nghiệp khi chuyển đổi sang dạng **Mel-Spectrogram 2D** tạo ra các dải tần số bị đứt đoạn, sọc vạch hoặc dải nhiễu nổ rất rõ ràng.
* Các mạng nếp gấp thị giác như **ResNet18** nhạy cảm gấp nhiều lần với các nét thị giác này so với việc nén âm thanh thành vector 1D phẳng. Kết quả đạt điểm tuyệt đối **`1.0000` trên máy SLIDER** và trung bình **`0.9866`** trên cả 4 máy, khẳng định đây là **phương pháp Trained Baseline mạnh nhất bài toán âm thanh**.

### 3.2. Tính Hiệu Quả Của Mô Hình Cây Trên Vector Đặc Trưng Whisper (Whisper + XGBoost - Mean AUC: 0.9316)
* Mô hình **Whisper Audio Encoder** (dù được pre-train chính cho bài toán nhận dạng giọng nói) vẫn trích xuất được vector đặc trưng đại diện âm thanh công nghiệp rất tốt.
* Kết hợp với **XGBoost (Supervised 80/20)** tạo nên một **Trained Audio Baseline** rất vững chắc với điểm trung bình **`0.9316`** (đạt cao nhất `0.9823` ở máy SLIDER).

### 3.3. Khó Khăn Của Bài Toán Anomaly Detection Không Giám Sát (Autoencoder & One-Class SVM)
* **Autoencoder (Unsupervised - Mean AUC: 0.6767):** Khi chỉ được huấn luyện trên mẫu âm thanh bình thường (`Normal`), Autoencoder đạt kết quả khả quan nhất trên máy `SLIDER` (`0.8235`) và `PUMP` (`0.7131`), nhưng giảm ở `VALVE` (`0.5765`) do tiếng van đóng mở có độ biến động ngẫu nhiên cao.
* **CLAP + One-Class SVM (Mean AUC: 0.5135):** Điểm tiệm cận mốc ngẫu nhiên (`0.51`), cho thấy vector CLAP trích xuất dạng tĩnh chưa đủ độ nhạy ranh giới khi dùng với thuật toán One-Class Boundary không giám sát.

### 3.4. Phát Hiện Đắt Giá Về Mô Hình Đa Phương Thức Zero-Shot (Qwen2-VL - Mean AUC: 0.4960)
* **Hiện tượng "Mù chữ" Ảnh Phổ (Spectrogram Blindness):** Mặc dù mô hình VLM như Qwen2-VL nhận diện đối tượng ảnh tự nhiên rất giỏi, nhưng khi đối mặt với **bức ảnh phổ âm thanh Mel-Spectrogram**, mô hình suy luận Zero-shot chỉ đạt AUC-ROC **`0.4960` (ngang với đoán ngẫu nhiên pure random `0.50`)**.
* **Ý nghĩa:** Kết quả này khẳng định các mô hình VLM tổng quát **không thể tự động hiểu được ảnh phổ âm thanh nếu không trải qua quá trình Fine-tune hoặc Prompt Instruction chuyên biệt**.

---

## 4. Đề Xuất & Định Hướng Chiến Lược Cho Dự Án (Strategic Recommendations)

> [!IMPORTANT]
> **Đề xuất 1: Đặt mốc sàn đối chứng (Performance Floor) cho nhánh Âm thanh**  
> Đối với bài toán chẩn đoán âm thanh công nghiệp (Task B3 - MIMII), nhóm nghiên cứu đề xuất sử dụng mốc **AUC-ROC `0.9866` (Họ thị giác - ResNet18)** và **`0.9316` (Họ vector - XGBoost)** làm mốc sàn có huấn luyện (Trained Baselines). Mọi mô hình nền tảng thế hệ mới bắt buộc phải tiệm cận mốc này.

> [!TIP]
> **Đề xuất 2: Chuyển đổi tín hiệu Âm thanh sang dạng Đa phương thức (Audio-to-Visual Pipeline)**  
> Thực nghiệm chứng minh việc chuyển đổi âm thanh `.wav` ➔ ảnh **Mel-Spectrogram** mang lại hiệu quả vượt trội hơn hẳn dạng vector 1D phẳng. Nhóm đề xuất tiếp tục giữ nguyên đường ống (pipeline) biểu diễn dạng ảnh phổ này cho tất cả các mô hình Đa phương thức tiếp theo.

> [!TIP]
> **Đề xuất 3: Định hướng Fine-tune cho Mô hình World Model / VLM**  
> Do kết quả Zero-shot của VLM chỉ đạt `0.4960`, để mô hình World Model (như Cosmos3 hay Qwen2-VL) giải được bài toán âm thanh công nghiệp, nghiên cứu trong Phase tiếp theo bắt buộc phải triển khai **Fine-tune Lora / Instruction Tuning** trên bộ dữ liệu ảnh phổ Mel-Spectrogram thay vì chạy suy luận Zero-shot đơn thuần.
