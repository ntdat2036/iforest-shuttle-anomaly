# Phát Hiện Bất Thường Cảm Biến Tàu Con Thoi NASA (Statlog Shuttle Telemetry)
## Mô Hình Học Máy Không Giám Sát Isolation Forest (Triển Khai Thuần NumPy - Zero Scikit-Learn)

Dự án phát triển hệ thống phát hiện dị thường trên dữ liệu đo đạc cảm biến thời gian thực (telemetry) của tàu con thoi không gian NASA (Statlog Shuttle Dataset). Toàn bộ thuật toán **Isolation Forest** (theo công bố gốc của Liu, Ting & Zhou, 2008) được tự xây dựng từ số 0 (**From Scratch - Pure NumPy**) mà không sử dụng bất kỳ thư viện học máy bậc cao nào (Zero Scikit-Learn).

Quy trình vận hành theo chuẩn mực **100% Học Không Giám Sát (Pure Unsupervised Anomaly Detection)** trên **9 kênh cảm biến đo đạc vật lý thực tế** (`feat_1` đến `feat_9`). Cột thứ 10 gốc (`feat_10`) là nhãn lớp UCI Statlog ({1..7}), được bóc tách hoàn toàn khỏi ma trận đặc trưng $X$ để loại trừ triệt để hiện tượng rò rỉ nhãn (Label Leakage) và chỉ được sử dụng cho việc đối chuẩn ngoại vi (External Supervised Validation). Toàn bộ tiến trình nghiên cứu được tổ chức theo khung phương pháp luận **21 bước chuẩn hóa của vòng đời phát triển mô hình Machine Learning**.

---

## 1. Cấu Trúc Dự Án

Thư mục dự án được tinh giản tối đa, chỉ lưu trữ mã nguồn cốt lõi, bộ kiểm thử, tài liệu báo cáo và dữ liệu quan trắc gốc:

```
iforest/
├── config.py                                # Hằng số cấu hình & lưới siêu tham số (Đổi tên từ weights.py)
├── weights_io.py                            # Quản lý I/O mã hóa/giải mã trọng số (save/load weights)
├── model.py                                 # Thuật toán thuần Isolation Forest (to_dict/from_dict)
├── run_pipeline.py                          # Pipeline điều phối không giám sát & quản lý cache trọng số
│
├── shuttle.csv                              # Dữ liệu telemetry cảm biến NASA gốc (58,000 mẫu x 10 cột, gồm 9 cảm biến + 1 nhãn UCI)
├── requirements.txt                         # Danh mục thư viện phụ thuộc (numpy, pandas, matplotlib, seaborn, pytest)
├── README.md                                # Báo cáo khoa học kỹ thuật và hướng dẫn vận hành toàn diện
│
├── weights/                                 # THƯ MỤC TRỌNG SỐ ĐÃ SINH RA (4 file)
│   ├── baseline_weights.json                # Trọng số baseline Distance-to-Centroid trên TRAIN (source of truth)
│   ├── baseline_weights.npz                 # Trọng số baseline định dạng nén NumPy (np.savez_compressed, allow_pickle=False)
│   ├── baseline_weights.txt                 # Trọng số baseline dạng văn bản UTF-8 đọc bằng mắt (format .17g)
│   └── best_model_weights.json              # Trọng số & cấu hình mô hình Isolation Forest tốt nhất
│
├── shuttle_anomaly_detection_iforest.ipynb  # Jupyter Notebook báo cáo trực quan
├── shuttle_anomaly_detection_iforest.html   # Bản xuất HTML tương tác đầy đủ hình ảnh và báo cáo thực nghiệm
│
└── tests/
    ├── test_pipeline.py                     # Bộ 21 bài kiểm thử tự động toàn diện
    └── test_weights.py                      # Bộ kiểm thử serialization & cache trọng số
```

---

## 2. Đặc Tả Dữ Liệu Cảm Biến NASA Shuttle (Statlog Telemetry)

- **Quy mô mẫu:** 58,000 quan sát từ hệ thống cảm biến tàu con thoi không gian NASA.
- **Không gian thuộc tính:** 9 biến số nguyên định lượng (`feat_1` đến `feat_9`) đo đạc áp suất van thủy lực, lưu lượng chất tải nhiệt, nhiệt độ buồng trao đổi và chu kỳ bơm.
- **Khắc phục triệt để Rò rỉ Nhãn (Label Leakage):**
  - Cột thứ 10 trong file gốc `shuttle.csv` (`col[9]`) có phân phối: Class 1 (Rad Flow - 45,586 mẫu / 78.6%), Class 2 (50), Class 3 (171), Class 4 (8,903), Class 5 (3,267), Class 6 (10), Class 7 (13) — trùng khớp 100% với nhãn lớp benchmark UCI Statlog Shuttle gốc.
  - Cột này được bóc tách độc lập hoàn toàn khỏi ma trận đặc trưng $X \in \mathbb{R}^{58,000 \times 9}$. Mô hình hoàn toàn không nhìn thấy nhãn này khi huấn luyện và suy luận. Nhãn này chỉ dùng để đối soát năng lực phát hiện thực tế sau cùng (External Validation).
- **Hiện tượng Zero-Inflated & Kurtosis Cực Cao:**
  - `feat_2` (kurtosis = 2,647, 61.9% zero), `feat_4` (kurtosis = 7,698, 65.6% zero), `feat_6` (kurtosis = 5,979, 31.8% zero) có độ lệch cực hạn.
  - Do đó, phương pháp đo lệch Z-score tham số truyền thống ($\frac{x - \mu}{\sigma}$) bị phá vỡ hoàn toàn và được thay thế bằng phương pháp phân vị phi tham số (Percentile Rank Deviation).
- **Tính toàn vẹn dữ liệu:** 0 giá trị khuyết thiếu (NaN), 0 giá trị vô hạn (Inf) trên toàn bộ 522,000 điểm đo cảm biến.
- **Tính bất biến tỷ lệ (Scale Invariance):** Cây cô lập chỉ thực hiện phép so sánh thứ tự nhị phân $X_j < v$ với $v \sim \text{Uniform}(\min(X_j), \max(X_j))$, bảo toàn tuyệt đối thứ tự phân hoạch dưới mọi phép biến đổi đơn điệu tăng ngặt. Mô hình được huấn luyện trực tiếp trên thang đo thô của cảm biến mà không cần chuẩn hóa (Feature Scaling).
- **Phân tách kiểm soát rò rỉ (Leakage Control):** Phân chia tập Huấn luyện (Train - 46,400 mẫu / 80%) và tập Kiểm thử (Test - 11,600 mẫu / 20%) với tập chỉ mục rời rạc tuyệt đối:
  $$\text{Index}_{\text{train}} \cap \text{Index}_{\text{test}} = \emptyset$$

---

## 3. Cơ Sở Lý Thuyết & Kiến Trúc Giải Thuật

### 3.1. Thuật toán Isolation Forest (Liu et al., 2008)
Khác với các phương pháp dựa trên mật độ hoặc khoảng cách (DBSCAN, LOF, k-Means) có độ phức tạp $O(N^2)$, Isolation Forest tận dụng hai đặc tính tự nhiên của dị thường: **số lượng thiểu số** và **tọa độ tách biệt**. Bằng cách phân hoạch đệ quy ngẫu nhiên không gian, các điểm bất thường bị cô lập ở độ sâu rất nông của cây.

### 3.2. Hàm Tính Điểm Bất Thường (Anomaly Score)
Độ dài đường đi kỳ vọng $\mathbb{E}(h(\mathbf{x}))$ qua $t$ cây cô lập được chuẩn hóa bằng hệ số $c(\psi)$ (độ dài tìm kiếm trung bình không thành công trong cây tìm kiếm nhị phân BST với kích thước mẫu con $\psi$):

$$c(\psi) = 2 \ln(\psi - 1) + 2\gamma - \frac{2(\psi - 1)}{\psi}$$

với $\gamma \approx 0.5772156649$ là hằng số Euler-Mascheroni. Điểm bất thường $s(\mathbf{x}, \psi) \in [0, 1]$ được xác định bởi:

$$s(\mathbf{x}, \psi) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(\psi)}}$$

- Khi $\mathbb{E}(h(\mathbf{x})) \to 0 \implies s \to 1$: Mẫu có tính dị biệt cực cao (cô lập rất sớm).
- Khi $\mathbb{E}(h(\mathbf{x})) \to c(\psi) \implies s \to 0.5$: Mẫu có trạng thái bình thường danh định.
- Khi $\mathbb{E}(h(\mathbf{x})) \to \psi - 1 \implies s \to 0$: Mẫu nằm sâu trong vùng cụm mật độ dày đặc.

### 3.3. Hệ Thống Các Hệ Số Phạt Phù Hợp (Penalty Factors Formulation)
Để mô hình hoạt động khách quan, không thiên vị và đáp ứng tiêu chuẩn an toàn hàng không vũ trụ khắt khe, hệ thống tích hợp 3 cấp độ hệ số phạt (Penalty Factors):
1. **Hệ số phạt phần dư độ sâu tại nút lá ($c(n)$ as Leaf Adjustment Penalty):**
   Khi cây cô lập đạt giới hạn độ sâu tối đa $max\_depth = \lceil \log_2(\psi) \rceil = 7$ hoặc các điểm còn lại hoàn toàn đồng nhất về giá trị, quá trình đệ quy kết thúc tại nút lá chứa $n > 1$ mẫu. Theo định lý Liu et al. (2008), nếu chỉ lấy độ sâu duyệt $e$, đường đi sẽ bị đánh giá thấp nghiêm trọng. Độ dài đường đi thực tế được cộng thêm hệ số phạt chính xác tương đương kỳ vọng độ dài cây tìm kiếm nhị phân còn lại:
   $$h(\mathbf{x}) = e + c(n)$$
   với $c(n) = 2\ln(n-1) + 2\gamma - \frac{2(n-1)}{n}$. Hệ số phạt $c(n)$ giúp ước lượng độ dài đường đi bảo toàn tính tiệm cận không thiên vị (unbiased estimator) ngay cả khi cây bị cắt tỉa sớm.
2. **Hệ số phạt điều chuẩn cấu trúc (Structural Regularization Penalty):**
   Trần độ sâu tối đa $max\_depth = \lceil \log_2(\psi) \rceil = 7$ đóng vai trò như một cơ chế phạt độ phức tạp mô hình (Structural Regularization), ngăn chặn cây phân nhánh quá sâu vào vùng vi cấu trúc nhiễu, kiểm soát phương sai và triệt tiêu nguy cơ quá khớp (overfitting).
3. **Hệ số phạt chi phí rủi ro bất đối xứng (Asymmetric Risk Cost Penalty $\omega$):**
   Trong hệ thống giám sát an toàn bay NASA, chi phí tổn thất do bỏ sót sự cố bất thường (False Negative - sự cố phần cứng, áp suất đột biến, rò rỉ nhiệt) đe dọa thảm họa nghiệm trọng hơn rất nhiều so với chi phí kiểm tra một cảnh báo giả (False Positive - quy trình rà soát telemetry định kỳ). Do đó, tỷ số chi phí phạt được xác lập:
   $$\omega = \frac{C_{FN}}{C_{FP}} \ge 10$$
   $$\mathcal{L}_{\text{cost}}(\tau; \omega) = \omega \cdot \mathbb{P}(s < \tau \mid \text{Anomaly}) + 1.0 \cdot \mathbb{P}(s \ge \tau \mid \text{Nominal})$$
   Hệ số phạt $\omega \ge 10$ định hình chiến lược lựa chọn ngưỡng quyết định: Ưu tiên các ngưỡng có độ nhạy cao ($P_{95}$ hoặc ngưỡng lý thuyết $0.50$) nhằm triệt tiêu tối đa rủi ro tổn thất từ các sự cố tiềm ẩn.

---

## 4. Hệ Thống 5 Ngưỡng Quyết Định Không Giám Sát

Hệ thống thiết lập cơ chế đánh giá đa tầng dựa trên phân phối xác suất và độ dài đường đi trên tập kiểm thử Test ($N = 11,600$ mẫu x 9 cảm biến):

| Cấp Độ Ngưỡng | Căn Cứ Khoa Học | Giá Trị Ngưỡng | Số Mẫu Phát Hiện | Tỷ Lệ Phát Hiện (%) | Mức Độ Can Thiệp Kỹ Thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Ngưỡng Lý Thuyết (Liu et al., 2008)** | $\mathbb{E}(h(\mathbf{x})) < c(\psi) \iff s \ge 0.50$ | `0.5000` | 1,967 | **16.96%** | Cảnh báo giám sát thường quy |
| **2. Phân Vị Ô Nhiễm Ước Lượng (Top 5%)** | Phân vị $P_{95}$ từ phân phối điểm tập Train | `0.5937` | 589 | **5.08%** | Cảnh báo bất thường telemetry |
| **3. Phân Vị Nguy Hiểm Cao (Top 1%)** | Phân vị $P_{99}$ từ phân phối điểm tập Train | `0.6554` | 134 | **1.16%** | Báo động nguy cơ sự cố cấp 2 |
| **4. Phân Vị Cực Đoan Thảm Họa (Top 0.1%)** | Phân vị $P_{99.9}$ từ phân phối điểm tập Train | `0.6875` | 13 | **0.11%** | Báo động đỏ / Ngắt hệ thống khẩn cấp |
| **5. Giới Hạn Thống Kê ($\mu + 2\sigma$)** | Độ lệch chuẩn phân phối Gaussian | `0.5706` | 751 | **6.47%** | Theo dõi trôi dạt dữ liệu (Data Drift) |

---

## 5. Chẩn Đoán Căn Nguyên & Tính Ổn Định (RCA & Feature Importance)

Hệ thống ứng dụng phương pháp phi tham số (Non-parametric) kết hợp cơ chế phá vỡ thế hòa (Tie-Breaking) để đảm bảo độ bền vững tuyệt đối trước hiện tượng zero-inflated và bão hòa phân vị:

1. **Đo lệch chuẩn phi tham số (Percentile Rank Deviation):**
   - Thay thế $Z$-score bằng độ lệch phân vị so với trung vị:
     $$\text{Dev}_j(x) = \left| \text{PercentileRank}(x, X_j) - 50.0 \right|$$
   - Miền giá trị $[0.0, 50.0]$: 0.0 nghĩa là nằm tại trung vị, 50.0 nghĩa là nằm ở điểm biên cực đoan của phân phối. Hoàn toàn miễn nhiễm trước kurtosis cực hạn ($> 2,000$).
2. **Cơ chế Phá Vỡ Thế Hòa (Robust IQR Tie-Breaking) & Báo Cáo Đồng Cực Đoan (Co-dominant Anomalies):**
   - *Hạn chế đã giải quyết:* Khi nhiều cảm biến cùng chạm giá trị biên (phân vị tiệm cận $50.0\%$), phương pháp phân vị thông thường dễ bị bão hòa, dẫn đến việc `np.argmax` chỉ chọn ngẫu nhiên 1 biến đầu tiên và bỏ sót các kênh cảm biến khác cũng đang gặp sự cố nghiêm trọng (ví dụ mẫu `idx=3228` có cả `feat_1`, `feat_2`, `feat_7` cùng ở mức $49.95\% - 49.99\%$).
   - *Giải pháp triệt để:*
     - Tích hợp độ lệch chuẩn hóa theo khoảng tứ phân vị Robust IQR: $\text{Dev}_{\text{IQR}, j} = \frac{|x_j - \text{Median}_j|}{\text{IQR}_j}$ làm tiêu chí phụ (Tie-Breaker), xác định chính xác và tất định kênh cảm biến lệch xa nhất trong không gian đuôi.
     - Bổ sung trường thông tin **`Đặc trưng đồng cực đoan` (Co-dominant Features)**: Tự động phát hiện và báo cáo toàn bộ các cảm biến đồng hạng lệch sát cực đại ($\ge \max - 0.15\%$ và $\ge 49.0\%$), đảm bảo kỹ sư buồng lái nắm bắt trọn vẹn mọi sự cố đa biến mà không bị che khuất.
3. **Độ quan trọng đặc trưng không giám sát (Percentile-Based UFI):**
   - Kết hợp tương quan tuyến tính Pearson $|r(X_j, s)|$ với trung vị độ lệch phân vị của nhóm bất thường ($s \ge 0.5$).
   - **Top cảm biến chủ đạo:** `feat_8`, `feat_9`, `feat_5`.
   - **Kiểm định tính ổn định xếp hạng (Spearman Stability Test):** Hệ số Spearman rho giữa 2 nửa nhóm bất thường đạt $\rho = 0.83$ (vượt ngưỡng yêu cầu $0.60$), chứng minh thứ hạng cảm biến hoàn toàn ổn định và không bị phụ thuộc vào phân chia mẫu ngẫu nhiên.

---

## 6. Kết Quả Kiểm Định Ngoại Vi (External Supervised Validation)

Sau khi mô hình được huấn luyện hoàn toàn không giám sát trên 9 cảm biến, nhãn lớp UCI gốc (`y_eval = y_test != 1`, tỷ lệ dị thường thực tế 21.48%) được đưa vào để đo lường năng lực phát hiện thực tế:

| Chỉ Số Đánh Giá Ngoại Vi | Giá Trị Thực Nghiệm (Leak-Free, 9 Cảm Biến) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :--- |
| **ROC-AUC (Không Rò Rỉ Nhãn)** | **0.8269** | Khả năng phân biệt thực tế của mô hình không giám sát (trung thực, loại bỏ con số bị thổi phồng do leakage) |
| **Average Precision (PR-AUC / AP)** | **0.6447** | Năng lực phân định trên phân phối mất cân bằng (gấp 3.0 lần baseline ngẫu nhiên 0.2148) |
| **Tại Ngưỡng Lý Thuyết ($s \ge 0.5000$)** | Prec = 0.5638 \| Rec = 0.4450 \| **F1 = 0.4974** | Bắt được 1,109 mẫu sự cố thực tế với độ chuẩn xác 56.4% |
| **Tại Ngưỡng Top 5% ($s \ge 0.5937$)** | **Prec = 0.9949** \| Rec = 0.2352 \| F1 = 0.3804 | Độ chính xác đạt tới 99.5%, cực kỳ tin cậy cho cảnh báo mức cao |

---

## 7. Khung 21 Bước Phát Triển Mô Hình Machine Learning

Toàn bộ quy trình trong dự án và Jupyter Notebook [shuttle_anomaly_detection_iforest.ipynb](file:///d:/May_Hoc/iforest/shuttle_anomaly_detection_iforest.ipynb) được liên kết chặt chẽ theo 21 bước kinh điển:

| STT | Bước Chuẩn Hóa | Mô Tả Thực Hiện Trong Dự Án |
| :---: | :--- | :--- |
| **1** | **Problem Definition** | Nhận diện sớm sai lệch telemetry cảm biến tàu con thoi NASA; 9 kênh đầu vào thô, bóc tách nhãn UCI lớp 1-7. |
| **2** | **Nature of ML Problem** | Bản chất Thuần Không Giám Sát (Unsupervised Learning); dị thường là thiểu số tách biệt trong không gian $\mathbb{R}^9$. |
| **3** | **Domain Understanding** | Khảo sát 9 kênh đo đạc áp suất, lưu lượng, nhiệt độ buồng trao đổi; bóc tách phân phối 7 lớp của nhãn UCI Statlog gốc. |
| **4** | **Data Cleaning** | Thẩm định 0 NaN, 0 Inf; nhận diện đặc tính zero-inflated (kurtosis > 2,000 ở feat_2, feat_4, feat_6). |
| **5** | **Feature Scaling** | Chứng minh tính bất biến tỷ lệ của $iTree$ trước phép biến đổi đơn điệu; giữ nguyên thang đo nguyên gốc `int64`. |
| **6** | **Categorical Encoding** | Thẩm định 9 kênh đều là biến định lượng liên tục/rời rạc, không phát sinh chi phí mã hóa One-Hot. |
| **7** | **Algorithm, Objective & Penalty Selection** | Isolation Forest (Liu et al., 2008); thiết lập hàm điểm bất thường và 3 cấp độ hệ số phạt ($c(n)$, $max\_depth$, và $\omega \ge 10$). |
| **8** | **Feature Engineering** | Khảo sát lời nguyền số chiều; tận dụng cơ chế phân hoạch 1D ngẫu nhiên ở mỗi nút để kháng đồng nhất khoảng cách. |
| **9** | **Data Splitting & Leakage** | Phân chia Train/Test 80/20 ($46,400$ / $11,600$) với chỉ số rời rạc tuyệt đối (`isdisjoint == True`), loại trừ hoàn toàn rò rỉ nhãn. |
| **10** | **Splitting Strategy** | Phân chia ngẫu nhiên đồng đều có cố định hạt giống (`random_state=42`) trên dữ liệu cảm biến không nhãn. |
| **11** | **Baseline Model** | Xây dựng mô hình chuẩn đối sánh thống kê đa biến (Distance-to-Centroid / Normalized Euclidean Baseline). |
| **12** | **No Free Lunch Theorem** | Phân tích không gian giả thuyết phân hoạch trực giao của $iTree$ so với giả định phân phối lồi của Baseline. |
| **13** | **Bias-Variance Tradeoff** | Khảo sát số lượng cây $t$ (kiểm soát Variance, hội tụ tại $t \ge 100$) và kích thước mẫu con $\psi=128$ (kiểm soát Bias). |
| **14** | **Hyperparameter Tuning** | Thiết lập lưới tìm kiếm không giám sát theo `config.py` `PARAM_GRID` gồm 4 cấu hình ($n\_estimators / max\_samples / max\_features$: 100/128/1.0, 100/256/1.0, 150/256/0.8, 200/256/1.0), tiêu chí chọn là Score Spread ($\sigma_s$) trung bình 3-fold (`KFOLD_N_SPLITS=3`). Nêu rõ: tiêu chí chọn KHÔNG dùng nhãn; nhãn chỉ dùng để đối chiếu sau. |
| **15** | **Evaluation Metrics & Cost Thresholds** | Xây dựng hàm chi phí rủi ro bất đối xứng $\mathcal{L}_{\text{cost}}(\tau; \omega=10)$ và hệ thống 5 cấp độ ngưỡng phân tầng. |
| **16** | **Cross-Validation** | Thực hiện Unsupervised 3-Fold Cross-Validation trên tập Train, chứng minh tính ổn định cao của phân phối điểm số. |
| **17** | **Model Training & Inference** | Huấn luyện mô hình sản xuất tối ưu ($n=100, \psi=128, \text{max\_features}=1.0$, $max\_depth=7, c(\psi)=8.8584$) trên $46,400$ mẫu Train và suy luận trên $11,600$ mẫu Test. |
| **18** | **Statistical Significance** | Tính khoảng tin cậy 95% ($\text{CI}_{95\%} = [0.4270, 0.4293]$) và kiểm định giả thuyết hai mẫu độc lập $Z$-test ($Z = -31.79, p < 10^{-10}$). |
| **19** | **Error Analysis** | Phân tích độ bất định tại vùng biên quyết định $s \in [0.48, 0.52]$; thiết lập vùng đệm cảnh báo vàng (Yellow Buffer Zone). |
| **20** | **Model Interpretability & RCA** | Trích xuất Top 5 mẫu bất thường nhất kèm Root Cause Analysis phi tham số (Percentile Rank) và đánh giá đối chuẩn ROC-AUC/AP. |
| **21** | **Iterative Deployment** | Thiết lập chu trình theo dõi trôi dạt dữ liệu (Drift Detection) và đóng gói lớp `TelemetryInferencePipeline` tối ưu dưới 1ms/mẫu. |

---

## 8. Cài Đặt & Hướng Dẫn Vận Hành

### 8.1. Cài Đặt Môi Trường
Yêu cầu Python $\ge$ 3.10 (khuyến nghị Python 3.13). Cài đặt các thư viện phụ thuộc:

```powershell
py -3.13 -m pip install -r requirements.txt
```

### 8.2. Thực Thi Pipeline Không Giám Sát (Single Run)
Chạy quy trình huấn luyện trên 9 cảm biến, phân tích 5 ngưỡng quyết định và trích xuất Root Cause Analysis:

```powershell
py -3.13 run_pipeline.py
```

### 8.3. Tinh Chỉnh Siêu Tham Số với Unsupervised 3-Fold CV
Chạy khảo sát lưới tham số và kiểm định chéo K-Fold không giám sát:

```powershell
py -3.13 run_pipeline.py --tune
```

### 8.4. Chạy Toàn Bộ Bộ Kiểm Thử Tự Động (Unit Tests)
Thực thi 21 bài kiểm thử nghiêm ngặt bao quát toán học, kiểm soát rò rỉ nhãn, độ ổn định RCA và tính tái lập:

```powershell
python -m pytest tests -v
```

*Kết quả kiểm thử thực tế:* **25 passed (< 1.5s, 100% Pass Rate)**.

### 8.5. Khởi Chạy Jupyter Notebook
Mở notebook phân tích trực quan toàn diện (18 hình vẽ khoa học tương tác):

```powershell
jupyter notebook shuttle_anomaly_detection_iforest.ipynb
```

---

## 9. Mô Phỏng Suy Luận Thời Gian Thực (Inference Simulation)

Đoạn mã mẫu thể hiện cách tích hợp mô hình vào luồng dữ liệu telemetry thời gian thực với 9 kênh cảm biến:

```python
import numpy as np
import pandas as pd
from model import IsolationForest
from run_pipeline import Pipeline

# 1. Nạp dữ liệu và huấn luyện mô hình sản xuất trên 9 cảm biến
df = pd.read_csv("shuttle.csv", header=None)
X_train = df.iloc[:46400, :9].values  # Đúng 9 cảm biến, không lấy cột nhãn 10

iforest = IsolationForest(n_estimators=100, max_samples=128, random_state=42)
pipe = Pipeline(model=iforest)
pipe.fit(X_train)

# 2. Gói tin telemetry cảm biến mới từ tàu con thoi (9 thuộc tính vật lý)
test_packet = np.array([[50, 21, 77, 0, 28, 0, 27, 48, 22]])

# 3. Tính Anomaly Score trực tiếp
score = float(pipe.anomaly_score(test_packet)[0])

# 4. Phân loại theo hệ thống ngưỡng quyết định
if score >= 0.6554:
    status = "NGUY HIỂM CAO (BÁO ĐỘNG ĐỎ)"
elif score >= 0.5000:
    status = "CẢNH BÁO BẤT THƯỜNG (MỨC VÀNG)"
else:
    status = "DANH ĐỊNH (BÌNH THƯỜNG)"

print(f"Anomaly Score: {score:.6f} -> Trạng thái: {status}")
```

---

## 10. Trọng Số & Tái Sử Dụng Mô Hình (Weights & Model Caching)

Hệ thống cung cấp cơ chế mã hóa/giải mã và nạp trọng số đã huấn luyện (Model Persistence) nhằm giảm thời gian tính toán từ **~12 giây (lần đầu train/tune)** xuống **< 1.5 giây (các lần chạy tiếp theo)** mà không cần huấn luyện lại.

### 10.1. Cấu Trúc Thư Mục `weights/`
Thư mục `weights/` chứa đầy đủ **4 file trọng số** theo đúng quy định:
1. `weights/baseline_weights.json`: Chứa các thống kê chuẩn hóa `train_mean`, `train_std`, và khoảng cách min/max `dist_min`, `dist_max` được tính **thuần túy trên tập TRAIN** (nguồn chính - source of truth).
2. `weights/baseline_weights.npz`: Định dạng nén nhị phân NumPy (`np.savez_compressed`), cho phép nạp siêu tốc mà không dùng pickle (`allow_pickle=False`).
3. `weights/baseline_weights.txt`: File văn bản UTF-8 trình bày trực quan các chỉ số baseline, định dạng số thực `.17g` bảo toàn độ chính xác tuyệt đối.
4. `weights/best_model_weights.json`: Chứa toàn bộ siêu tham số thắng cuộc (`best_params`), nhật ký tinh chỉnh (`tuning`), các chỉ số hiệu năng thực tế (`metrics`: ROC-AUC, AP, Score Spread), thông tin kiểm soát mã băm dữ liệu (`meta`: SHA-256, seed, test size) và cây cô lập được flatten dạng preorder traversal (`trees`).

### 10.2. Cơ Chế Vận Hành Cache & Tùy Chọn CLI
- **Lần chạy đầu tiên:** Chạy `python run_pipeline.py --tune` để tìm kiếm siêu tham số và lưu trọng số tối ưu vào `weights/` (sinh đủ 4 file).
- **Các lần chạy sau:** Chạy `python run_pipeline.py` mô hình sẽ tự động phát hiện 4 file trọng số hợp lệ và **LOAD ngay lập tức, bỏ qua bước huấn luyện**.
- **Tự động đồng bộ file baseline:** Nếu thư mục `weights/` thiếu file `.npz` hoặc `.txt` (hoặc dữ liệu trong `.npz` bị lệch so với JSON), hàm `sync_baseline_files()` sẽ tự động tái tạo lại các file này từ `baseline_weights.json` mà **KHÔNG huấn luyện lại** và **KHÔNG sửa đè file JSON**.
- **Ép train lại (`--retrain`):** Chạy `python run_pipeline.py --retrain` để huấn luyện lại từ đầu và ghi đè trọng số mới.
- **Thay đổi siêu tham số CLI:** Khi truyền tường minh bất kỳ tham số mô hình nào (`--n_estimators`, `--max_samples`, `--max_features`, `--contamination`), hệ thống tự nhận diện tham số thay đổi và kích hoạt huấn luyện lại.

---

## 11. Cam Kết Tiêu Chuẩn Kỹ Thuật & Học Thuật

1. **Zero Scikit-Learn Dependency:** 100% cấu trúc cây nhị phân, thuật toán phân hoạch không gian và tính toán đường đi được hiện thực thuần túy bằng Python và NumPy.
2. **Loại Trừ Hoàn Toàn Rò Rỉ Nhãn (100% Leak-Free):** Cột 10 được tách biệt hoàn toàn khỏi không gian huấn luyện; ROC-AUC trung thực đạt **0.8269** trên 9 cảm biến thật.
3. **RCA Phi Tham Số Bền Vững (Percentile Rank RCA):** Khắc phục triệt để sự thất bại của Z-score trên dữ liệu zero-inflated và kurtosis > 2,000; kiểm định ổn định Spearman đạt $\rho = 0.83 \ge 0.60$.
4. **Phòng Chống Rò Rỉ Dữ Liệu Tuyệt Đối:** Mọi tham số phân vị và thống kê đều được ước lượng độc quyền trên tập Train trước khi áp dụng cho tập Test.
5. **Độ Tin Cậy & Tinh Gọn Mã Nguồn:** 0 lỗi/cảnh báo linter, 25/25 unit tests tự động (21 pipeline tests + 4 weights tests) vượt qua hoàn hảo.

