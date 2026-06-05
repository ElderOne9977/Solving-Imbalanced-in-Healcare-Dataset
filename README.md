# Solving Imbalanced in Healthcare Dataset: Dự đoán nguy cơ suy tim với dữ liệu y tế NHANES

Dự án này tập trung vào việc xử lý **dữ liệu y tế mất cân bằng nghiêm trọng** nhằm xây dựng một mô hình Machine Learning dự đoán nguy cơ suy tim (Heart Failure) từ khảo sát sức khỏe quốc gia Hoa Kỳ **NHANES** (National Health and Nutrition Examination Survey) qua 7 chu kỳ từ năm 2005 đến năm 2018.

---

## Bối Cảnh & Đặt Vấn Đề

Trong dữ liệu y tế công cộng, tỷ lệ người mắc các bệnh lý nguy hiểm (như suy tim) thường rất nhỏ so với nhóm người khỏe mạnh. Trong bộ dữ liệu NHANES sau khi làm sạch và tiền xử lý:
* **Tổng số mẫu:** 38,570 mẫu.
* **Không suy tim (Class 0):** 37,235 mẫu (96.54%).
* **Bị suy tim (Class 1):** 1,335 mẫu (3.46%).

Tỷ lệ mất cân bằng xấp xỉ **28:1**. Nếu huấn luyện mô hình trực tiếp trên dữ liệu gốc, mô hình sẽ bị thiên lệch (bias) về nhóm đa số (không bị bệnh), đạt độ chính xác (Accuracy) tổng thể rất cao nhưng hoàn toàn bỏ sót các ca bệnh thực tế. Trong y tế, việc bỏ sót ca bệnh (False Negative) là cực kỳ nguy hiểm, do đó dự án này ưu tiên tối ưu hóa chỉ số **Recall** (độ nhạy/khả năng phát hiện bệnh).

---

## Quy Trình Thực Hiện (Pipeline)

### 1. Thu thập & Làm sạch dữ liệu (Data Crawling & Cleaning)
* Tự động tải 63 file dữ liệu định dạng SAS (`.XPT`) từ CDC Hoa Kỳ thuộc 7 chu kỳ khảo sát khác nhau.
* Các nhóm biến khảo sát bao gồm:
  * **DEMO:** Dữ liệu nhân khẩu học (Tuổi, Giới tính, Chủng tộc...)
  * **BMX:** Chỉ số đo lường cơ thể (BMI, Cân nặng, Chiều cao, Vòng eo...)
  * **BPX:** Huyết áp tâm thu và tâm trương.
  * **TCHOL:** Chỉ số Cholesterol toàn phần.
  * **GLU:** Chỉ số đường huyết lúc đói.
  * **DIQ:** Tiền sử tiểu đường.
  * **SMQ:** Thói quen hút thuốc.
  * **PAQ:** Hoạt động thể chất cường độ trung bình.
  * **MCQ:** Thông tin tiền sử bệnh suy tim (Biến mục tiêu `heart_failure`).
* Dữ liệu thô sau khi đọc được chuẩn hóa và đẩy trực tiếp lên cơ sở dữ liệu **SQL Server** để thực hiện các câu lệnh truy vấn ghép bảng và lọc nâng cao nhằm tăng hiệu năng xử lý.
* Xử lý ngoại lai (Outliers) bằng phương pháp **IQR (Interquartile Range)** và loại bỏ các dòng thiếu thông tin nhãn.

### 2. Kỹ nghệ đặc trưng (Feature Engineering)
Từ các đặc trưng ban đầu, hai đặc trưng dẫn xuất quan trọng được tạo ra:
1. **Pulse Pressure (Áp lực xung):** Hiệu số giữa huyết áp tâm thu và huyết áp tâm trương (`systolic_bp` - `diastolic_bp`). Đây là chỉ số quan trọng phản ánh độ cứng của thành mạch máu.
2. **Waist-Height Ratio (Tỷ lệ vòng eo/chiều cao):** `waist_cm` / `height_cm`, giúp đánh giá béo phì trung tâm chính xác hơn chỉ số BMI thông thường đối với các bệnh tim mạch.

Danh sách 14 đặc trưng đưa vào huấn luyện mô hình:
`age`, `sex_Male`, `bmi`, `height_cm`, `waist_cm`, `systolic_bp`, `diastolic_bp`, `total_cholesterol`, `fasting_glucose`, `diabetes`, `smoking_100_cigs_life`, `moderate_activity`, `pulse_pressure`, `waist_height_ratio`.

### 3. Xử lý mất cân bằng dữ liệu (Imbalanced Data Handling)
Dự án áp dụng thuật toán **NearMiss (Version 2)** để giảm số lượng mẫu của lớp đa số (Undersampling) trên tập huấn luyện (train set).
* **NearMiss-2** lựa chọn giữ lại các mẫu lớp đa số có khoảng cách trung bình ngắn nhất tới tất cả các mẫu lớp thiểu số. Điều này giúp mô hình tập trung tìm kiếm ranh giới phân loại chính xác tại những khu vực nhập nhằng nhất.
* Tỷ lệ Undersampling được thiết lập để giữ tỷ lệ **đa số : thiểu số là 5 : 1** (majority_ratio = 5), giúp cân bằng vừa đủ để thuật toán học được đặc trưng lớp thiểu số mà không bị mất đi quá nhiều thông tin của lớp đa số.

### 4. Huấn luyện mô hình (Model Training)
Ba mô hình học máy được so sánh và đánh giá chéo:
* **Logistic Regression:** Mô hình baseline tuyến tính, có sử dụng trọng số lớp (`class_weight={0: 1, 1: 4}`) và chuẩn hóa dữ liệu đầu vào (`StandardScaler`).
* **Random Forest:** Mô hình học máy dạng ensemble (Bagging) phi tuyến, xử lý tốt dữ liệu dạng bảng có tương quan phức tạp.
* **XGBoost Classifier:** Mô hình Boosting mạnh mẽ, tối ưu hóa hàm mất mát và hạn chế Overfitting tốt.

---

## Kết Quả Đánh Giá Mô Hình

Do bài toán y tế ưu tiên phát hiện tối đa bệnh nhân suy tim, chúng ta tập trung vào **Recall** trên tập kiểm thử độc lập (Test Set).

### Kết quả trên tập Validation:
| Mô hình | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.1100** | **0.6567** | **0.1884** | **0.8035** |
| **Random Forest** | 0.0329 | 0.7090 | 0.0629 | 0.2658 |
| **XGBoost** | 0.0323 | 0.7164 | 0.0619 | 0.2455 |

### Kết quả trên tập Test độc lập:
| Mô hình | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.1015** | **0.6090** | **0.1740** | **0.8006** |
| **XGBoost** | 0.0316 | 0.6992 | 0.0605 | 0.2515 |
| **Random Forest** | 0.0302 | 0.6541 | 0.0578 | 0.2647 |

> [!IMPORTANT]
> **Nhận xét quan trọng:** 
> Mặc dù XGBoost và Random Forest đạt chỉ số Recall cao hơn một chút (~65% - 70%), nhưng độ chính xác tổng thể (Accuracy) của chúng cực kỳ thấp (~25%), đồng thời Precision chỉ khoảng ~3%. Điều này nghĩa là hai mô hình này dự đoán hầu hết mọi người đều bị suy tim (gây ra tỷ lệ báo động giả khổng lồ).
> 
> **Logistic Regression** được lựa chọn là mô hình tối ưu nhất nhờ giữ được mức cân bằng hợp lý: **Recall đạt 60.90%** trên tập test, trong khi duy trì **Accuracy vượt trội (80.06%)** và **Precision cao nhất (10.15%)**.

---

## Mô hình Ensemble & Dự đoán tương tác (Interactive Prediction)

Để tận dụng ưu điểm của cả ba thuật toán, notebook xây dựng một mô hình **Weighted Ensemble** (Kết hợp có trọng số) dựa trên xác suất dự đoán của các mô hình thành phần:
* **Logistic Regression:** Trọng số **0.6** (Mô hình chính, cân bằng tốt nhất).
* **Random Forest:** Trọng số **0.2**.
* **XGBoost:** Trọng số **0.2**.

Ngưỡng phân loại (Decision Threshold) của Ensemble được thiết lập ở mức **0.65** nhằm đảm bảo chỉ những trường hợp có nguy cơ cao mới được đưa ra kết luận cuối cùng là suy tim, hạn chế tối đa các ca dương tính giả không cần thiết. Cuối notebook cung cấp một giao diện nhập liệu trực tiếp (console prompt) cho phép nhân viên y tế nhập chỉ số của bệnh nhân mới và nhận dự đoán xác suất tức thời từ hệ thống.

---

## Cấu Trúc Thư Mục Dự Án

```
├── Data/
│   └── data.csv         # Dữ liệu cuối cùng sau khi làm sạch và ghép bảng (dùng để chạy trực tiếp)
├── docs/
│   └── jupyter-notebook/
│       └── Final_Project.docx  # Báo cáo tài liệu chi tiết của dự án
├── imbalanced.ipynb     # Jupyter Notebook chứa toàn bộ pipeline từ tải dữ liệu đến đánh giá mô hình
├── .gitignore           # File loại trừ các tệp tin không cần thiết (nhập nhằng, dữ liệu thô XPT)
└── requirements.txt     # Danh sách các thư viện cần thiết để chạy dự án
```

---

## Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python (phiên bản khuyến nghị `>= 3.10`). Di chuyển vào thư mục dự án và cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

### 2. Chạy Notebook
Mở và chạy file `imbalanced.ipynb` bằng Jupyter Notebook hoặc VS Code:

```bash
jupyter notebook imbalanced.ipynb
```

### Lưu ý khi chạy dữ liệu:
* **Tiền xử lý SQL Server:** Đoạn mã tiền xử lý dữ liệu thô đầu notebook yêu cầu kết nối cơ sở dữ liệu SQL Server cục bộ (được cấu hình với server `DESKTOP-B1NE7EL`).
* **Bỏ qua bước SQL:** Nếu bạn không cài đặt SQL Server hoặc không muốn tải lại dữ liệu thô từ đầu, bạn có thể **bắt đầu chạy thẳng từ phần "EDA & Visualization" (Cell 36 trở đi)**, vì bộ dữ liệu sạch đã được lưu sẵn tại `Data/data.csv` và sẽ được tải trực tiếp bằng Pandas.
