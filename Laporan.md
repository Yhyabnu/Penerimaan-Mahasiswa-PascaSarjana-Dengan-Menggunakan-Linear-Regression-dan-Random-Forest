# Laporan Proyek Machine Learning - Yahya Ibnu Fajar

# Judul: Prediski Penerimaan Mahasiwa PascaSarjana

## Domain Proyek
Proyek ini berfokus pada domain pendidikan tinggi, khususnya proses seleksi penerimaan mahasiswa pascasarjana. Dalam era digital dan kompetitif saat ini, universitas dan institusi pendidikan memerlukan pendekatan berbasis data untuk memperkirakan kemungkinan diterimanya calon mahasiswa berdasarkan faktor-faktor seperti skor GRE, TOEFL, IPK, dan kualitas universitas asal. Dengan adanya model prediksi ini, pihak universitas dapat memperoleh gambaran awal tentang calon mahasiswa dan mengoptimalkan proses seleksi. Bagi calon mahasiswa, model ini dapat membantu mengevaluasi peluang mereka diterima sebelum mendaftar.
Proyek ini tidak hanya menyederhanakan proses administratif tetapi juga mendukung keadilan pendidikan dengan meminimalkan disparitas informasi. Ke depannya, pengembangan dapat diperluas dengan memasukkan faktor non-akademik (pengalaman riset, rekomendasi dosen) atau integrasi dengan platform seperti LinkedIn untuk analisis jejaring profesional.

sumber yang digunakan:
M. A. Wahab, M. U. Siddiqui, "Graduate Admission Prediction using Machine Learning," International Journal of Computer Applications, 2019.

Dataset diambil dari Kaggle: [https://www.kaggle.com/code/zohaib123/graduate-admission-prediction-linearregression?select=Admission_Predict_Ver1.1.csv]

# Business Understanding

Problem Statements 

1. Bagaimana memprediksi peluang diterimanya mahasiswa pada program pascasarjana berdasarkan atribut akademik dan latar belakangnya?
2. Fitur apa saja yang memiliki pengaruh besar terhadap kemungkinan diterimanya mahasiswa?n

**Goals**
1. Membuat model prediksi regresi yang dapat memprediksi persentase peluang diterima.
2. Mengidentifikasi fitur-fitur yang berkontribusi besar terhadap hasil prediksi.

**Solution statements**
1. Menerapkan dua model regresi yaitu Linear Regression dan Random Forest Regressor untuk melakukan prediksi.
2. Menggunakan metrik evaluasi MSE, RMSE, MAE, dan R² untuk menilai kedua performa model.


## Data Understanding

**KONDISI DATA AWAL**
Data awal yang dimuat dari file `Admission_Predict.csv` terdiri dari 500 baris dan 9 kolom.
1.  Kolom identifikasi unik yaitu **'Serial No.' dihapus** karena tidak relevan untuk pemodelan. Setelah penghapusan kolom ini, dataset memiliki **500 baris dan 8 kolom**.
2.  Pemeriksaan **nilai yang hilang (missing values)** menunjukkan bahwa **tidak ada nilai yang hilang** di seluruh kolom dataset.
3.  Pemeriksaan **data terduplikasi** juga menunjukkan bahwa **tidak ada baris data yang terduplikasi** dalam dataset. Oleh karena itu, jumlah baris tetap 500 setelah pemeriksaan ini.
4.  Outlier: pengecekan outlier dilakukan dengan metode Interquartile Range (IQR). setalah melakukan pengecekan outlier terdapat 3 data outlier pada kolom Chance of Admit Setelah penghapusan outlier, jumlah data berkurang dari 500 menjadi 497.
Dengan demikian, dataset yang siap untuk eksplorasi lebih lanjut dan pemodelan memiliki 497 sampel dan 8 fitur.

**SUMBER DATASET**
Dataset Admission_Predict.csv merupakan kumpulan data komprehensif yang merekam profil akademik dan non-akademik calon mahasiswa pascasarjana. dataset tersebut dapat diunduh pada kaggle [https://www.kaggle.com/code/zohaib123/graduate-admission-prediction-linearregression?select=Admission_Predict_Ver1.1.csv]. 

**VARIABEL/FITUR**
- GRE Score (310–340): Skor tes Graduate Record Examination dengan distribusi tertinggi di rentang 315–325.
- TOEFL Score (90–120): Skor bahasa Inggris dengan rata-rata 105 (±8 poin).
- University Rating (1–5): Skala reputasi universitas asal (1 = lokal, 5 = internasional bereputasi tinggi).
- SOP (Statement of Purpose, 1–5): Penilaian kualitas esai motivasi.
- LOR (Letter of Recommendation, 1–5): Tingkat kekuatan rekomendasi akademik.
- CGPA (6.8–9.9): IPK kumulatif dalam skala 10.
- esearch Experience (0/1): Indikator pengalaman riset sebelumnya.
- Chance of Admit (0.3–0.9): Probabilitas diterima (variabel target).


## Data Preparation
Proses persiapan data dilakukan untuk memastikan dataset siap digunakan dalam pemodelan. Tahapan-tahapan yang dilakukan sesuai urutan di notebook adalah sebagai berikut:

1. **menghapus Kolom yang Tidak Relevan (series no)**
Kolom Serial No. dihapus karena tidak memiliki pengaruh terhadap target dan hanya berfungsi sebagai penomoran urut.

2. **Pemeriksaan Struktur Data (df.info)**
Dataset diperiksa menggunakan df.info() untuk memastikan jumlah baris, tipe data, serta memastikan tidak terdapat nilai kosong (missing value).

3. **melihat deskripsi statistik data**
memberikan ringkasan cepat tentang distribusi data numerik, termasuk ukuran pemusatan, dispersi, dan bentuk distribusi. 
Untuk data numerik, output default mencakup:
count : Jumlah nilai non-null.
mean : Rata-rata.
std : Standar deviasi (ukuran dispersi).
min : Nilai minimum.
25% : Kuartil pertama (Q1).
50% : Median (kuartil kedua/Q2).
75% : Kuartil ketiga (Q3).
max : Nilai maksimum.

4. **mengecek missing value**
melakukan pengecekan menyeluruh terhadap keberadaan nilai kosong atau missing value di seluruh kolom dataset. Hasil pemeriksaan menunjukkan bahwa tidak ditemukan satupun nilai null atau missing value, sehingga tidak diperlukan proses imputasi atau pengisian data. Kondisi ini memastikan kualitas data yang baik dan mengurangi risiko bias pada model prediksi.

5. **mengecek duplikat data**
Dilakukan pengecekan terhadap data duplikat menggunakan .duplicated(). Hasilnya menunjukkan bahwa tidak ada baris yang terduplikasi.

6. **mengecek dan menangani outlier**
Outlier dideteksi menggunakan metode Interquartile Range (IQR). Data yang berada di luar rentang Q1 - 1.5 * IQR dan Q3 + 1.5 * IQR dihapus. Untuk memvisualisasikan keberadaan outlier, digunakan boxplot untuk setiap fitur numerik. kemudian divisualisaikan untuk memberikan gambaran letak adanya outlietr. hasilnya menunjukan bahwa outlier terdapat 3 data outlier pada kolom Chance of Admit Setelah penghapusan outlier, jumlah data berkurang dari 500 menjadi 497. 

7. **normalisasi**
Dataset kemudian dinormalisasi menggunakan StandardScaler agar setiap fitur numerik memiliki skala distribusi yang seragam (rata-rata 0 dan standar deviasi 1).

8. **pembagian data**
Dataset dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan train_test_split() dari Scikit-learn. Data latih digunakan untuk melatih model sebanyak 80%, sementara data uji digunakan untuk evaluasi performa adalah 20%.

## Modeling
**PEMODELAN**
Dalam proyek ini, digunakan dua algoritma regresi untuk memprediksi probabilitas penerimaan mahasiswa, yaitu:
# *1. Linear Regression:*
Linear Regression adalah salah satu metode supervised learning paling dasar yang digunakan untuk memodelkan hubungan antara satu atau lebih fitur (variabel independen) dan target (variabel dependen). Model ini bekerja dengan mencari garis lurus terbaik yang meminimalkan selisih kuadrat antara nilai prediksi dan nilai aktual (disebut least squares method).

**cara algoritma Linear regresion bekerja**
1. Menerima Data Input
2. Mencari Pola Hubungan
3. Menghitung Kesalahan Prediksi
4. Menyesuaikan Garis untuk Minimalkan Error
5. Menentukan Garis Terbaik
6. Memprediksi Nilai Baru
7. Evaluasi Model

Contoh Analogi
Bayangkan Anda mencoba menggambar garis lurus terbaik melalui titik-titik hasil lemparan dart di papan. Linear Regression bekerja seperti:
- Mengamati semua titik.
- Mencoba berbagai kemiringan garis.
- Memilih garis yang paling dekat dengan semua titik.
- Garis itu kemudian bisa digunakan untuk menebak di mana dart berikutnya akan mendarat.

*Kapan Linear Regression Cocok Digunakan?*
- Jika hubungan antara fitur dan target sekitar lurus (misalnya: semakin banyak jam belajar, semakin tinggi nilainya).
- Untuk prediksi nilai numerik (harga, suhu, jumlah penjualan).
- Ketika butuh model sederhana dan mudah dijelaskan.

**Parameter yang digunakan Oleh algoritma Linear Regression:**
1. fit_intercept	berfungsi untuk Menambahkan intercept (β0β0) ke model. Jika False, garis melewati origin. nilai default fit_intercept adalahTrue
2. normalize digunakan digunakan untuk melakukan Normalisasi fitur sebelum fitting (digantikan oleh StandardScaler di versi baru). default nilainya adalah False
3. copy_X berfungdi Menyalin data sebelum fitting untuk menghindari modifikasi data asli.default nilai adalah True
4. n_job adalah jumlah core CPU untuk paralelisasi (berguna untuk dataset besar).

**Alasan Linear Regression dipilih**
1. Interpretabilitas Tinggi
2. Efisiensi Komputasi
3. Baseline yang Andal
4. Asumsi Dasar Terpenuhi
5. Kemampuan Generalisasi

# *2. Random Forest*
Random Forest adalah algoritma ensemble yang membentuk banyak decision tree dan menggabungkan hasil prediksi mereka. Untuk regresi, prediksi akhir diambil sebagai rata-rata dari seluruh prediksi tree. Metode ini dikenal kuat terhadap overfitting dan bekerja baik pada data non-linear dan kompleks.

**Cara Kerja Algoritma Random Forest**
Berikut adalah langkah-langkah kerjanya:
1. Bootstrap Sampling: Dataset dilatih dengan teknik bagging, yaitu membuat beberapa subset data dari dataset asli dengan pengambilan sampel secara acak dengan pengembalian.
2. Training Multiple Trees: Setiap subset digunakan untuk melatih satu decision tree. Setiap tree hanya melihat sebagian fitur saat melakukan pemisahan (splitting) untuk mencegah overfitting.
3. Agregasi Hasil: Setelah semua pohon selesai dilatih, prediksi akhir diambil sebagai rata-rata (mean) dari semua prediksi yang diberikan oleh masing-masing pohon (untuk regresi).

*Keunggulan utama Random Forest:*
1. Dapat menangani non-linear relationship antar fitur.
2. Robust terhadap outlier dan noise.
3. Dapat digunakan untuk estimasi pentingnya fitur (feature importance).
4. Tidak terlalu overfitting karena prinsip randomness dan averaging.

**Parameter Yang digunakan oleh Algoritma Random Forest**
1. n_estimators=100
- Menentukan jumlah decision tree yang akan dibuat oleh model Random Forest. Semakin banyak tree, model menjadi lebih stabil dan akurat karena hasil prediksi merupakan rata-rata dari banyak pohon. Nilai 100 adalah angka umum yang cukup optimal untuk mendapatkan hasil yang baik tanpa waktu komputasi berlebihan.
2. random_state=42
Seed atau angka acak yang digunakan untuk mengatur proses randomisasi internal. Tujuannya adalah untuk memastikan hasil yang reproducible, artinya hasil training dan evaluasi akan konsisten jika dijalankan ulang. Angka 42 adalah nilai konvensional yang sering digunakan, tetapi bisa diubah sesuai kebutuhan.

**Alasan Random Forest dipilih**
1. Mampu Menangani Kompleksitas Data
Prediksi Chance of Admit dipengaruhi oleh banyak faktor numerik (GRE, TOEFL, CGPA, SOP, LOR, Research), dan hubungan antar fitur tidak selalu linier. Random Forest dapat menangkap hubungan kompleks ini lebih baik dibanding Linear Regression.
2. Tahan terhadap Outlier dan Noise
Meski data telah dibersihkan dari outlier menggunakan IQR, Random Forest tetap lebih tahan terhadap sisa noise dibanding algoritma sederhana.
3. Performa Tinggi
Dalam eksperimen pada notebook Anda, Random Forest menunjukkan hasil metrik evaluasi (R², RMSE, MAE) yang lebih baik dibanding Linear Regression.
4. Fleksibilitas dan Stabilitas
Random Forest mengurangi overfitting yang sering terjadi pada Decision Tree tunggal dengan mengombinasikan banyak pohon dan memberikan hasil prediksi yang lebih stabil.

**Tahapan Pelatihan Model secara general untuk kedua model**

1. Fit model pada data pelatihan.
Proses pemodelan dimulai dengan melakukan fit atau pelatihan model menggunakan data pelatihan yang telah dipersiapkan sebelumnya. Pada tahap ini, algoritma machine learning secara iteratif mempelajari pola dan hubungan antara fitur input (seperti GRE, TOEFL, IPK, dan lain-lain) dengan variabel target (peluang diterima). Proses ini melibatkan penyesuaian parameter internal model agar mampu menghasilkan prediksi yang akurat dan generalisasi yang baik terhadap data baru

2. Prediksi data uji.
Setelah model selesai dilatih, langkah berikutnya adalah melakukan prediksi menggunakan data pengujian yang belum pernah dilihat oleh model sebelumnya. Ini bertujuan untuk mengukur kemampuan model dalam memprediksi peluang diterima calon mahasiswa secara objektif dan menghindari overfitting. Hasil prediksi ini nantinya akan dibandingkan dengan nilai aktual untuk menilai performa model.

3. Visualisasi: grafik actual vs predicted.
Untuk memberikan gambaran yang jelas mengenai akurasi model, dilakukan visualisasi grafik yang memperlihatkan perbandingan antara nilai aktual (real chance of admit) dan nilai prediksi yang dihasilkan oleh model. Visualisasi ini biasanya berupa plot scatter atau line chart yang memudahkan identifikasi pola kesesuaian dan deviasi prediksi. Dengan demikian, pihak universitas dan pengembang dapat mengevaluasi efektivitas model secara visual dan menentukan langkah perbaikan jika diperlukan. 

## Evaluation
**Metrik evaluasi yang digunakan:**

1. Mean Absolute Error (MAE)
MAE mengukur rata-rata nilai absolut dari selisih antara nilai prediksi model dengan nilai aktual. Dengan kata lain, MAE menunjukkan seberapa besar kesalahan prediksi secara rata-rata tanpa memperhatikan arah kesalahan (positif atau negatif). Nilai MAE yang lebih rendah mengindikasikan model yang lebih akurat dan konsisten dalam memprediksi peluang diterima calon mahasiswa.
2. Mean Squared Error (MSE)
MSE merupakan rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual. Karena kesalahan dikuadratkan, MSE memberikan penalti yang lebih besar terhadap kesalahan prediksi yang besar (outlier). Metrik ini sangat berguna untuk mendeteksi dan mengurangi dampak prediksi yang jauh meleset, sehingga model dapat dioptimalkan agar lebih stabil.
3. Root Mean Squared Error (RMSE)
RMSE adalah akar kuadrat dari MSE, yang mengembalikan skala kesalahan ke satuan asli variabel target sehingga lebih mudah diinterpretasikan. RMSE juga sensitif terhadap outlier, sehingga memberikan gambaran yang lebih realistis mengenai performa model ketika menghadapi data ekstrem. Nilai RMSE yang kecil menunjukkan bahwa model mampu memprediksi dengan kesalahan minimal secara keseluruhan.
4. R² Score (Koefisien Determinasi)
R² Score mengukur proporsi variasi dalam data target yang berhasil dijelaskan oleh model prediksi. Nilai R² berkisar antara 0 hingga 1, di mana nilai mendekati 1 menandakan model mampu menjelaskan hampir seluruh variabilitas data dengan baik. Metrik ini sangat penting untuk menilai seberapa efektif model dalam menangkap pola dan hubungan dalam dataset.

Berikut adalah nilai-nilai evaluasi dati kedua model tersebut:
**Linear Regression Evaluation:**
R² Score: 0.7959
MAE: 0.0436
MSE: 0.0037
RMSE: 0.0606

**Random Forest Regressor Evaluation:**
R² Score: 0.7727
MAE  : 0.0463
MSE  : 0.0041
RMSE : 0.064

**Berdasarkan nilai metrik evaluasi di atas, dapat disimpulkan perbandingan performa kedua model sebagai berikut:**
# *R² Score (Koefisien Determinasi):*
Linear Regression memiliki R² Score sebesar 0.7959, yang berarti model ini dapat menjelaskan sekitar 79.59% variabilitas dalam variabel target ('Chance of Admit').
Random Forest memiliki R² Score sebesar 0.7727, yang berarti model ini dapat menjelaskan sekitar 77.27% variabilitas dalam variabel target.
Perbandingan: Dalam hal kemampuan untuk menjelaskan variabilitas data, Linear Regression sedikit lebih unggul dibandingkan Random Forest, karena R² Score-nya lebih tinggi. Ini menunjukkan bahwa Linear Regression dapat menangkap proporsi variasi yang lebih besar dalam peluang penerimaan mahasiswa.

# *Mean Squared Error (MSE) & Root Mean Squared Error (RMSE):*
Linear Regression memiliki MSE sebesar 0.0037 dan RMSE sebesar 0.0606.
Random Forest memiliki MSE sebesar 0.0041 dan RMSE sebesar 0.0640.
Perbandingan: Nilai MSE dan RMSE yang lebih rendah menunjukkan bahwa model memiliki rata-rata kesalahan kuadrat yang lebih kecil. Dalam hal ini, Linear Regression memiliki nilai MSE dan RMSE yang lebih rendah daripada Random Forest. Ini mengindikasikan bahwa prediksi Linear Regression, secara rata-rata, lebih dekat ke nilai aktual dan memiliki kesalahan yang lebih kecil secara keseluruhan, terutama dalam menanggapi outlier.

# *Mean Absolute Error (MAE):*
Linear Regression memiliki MAE sebesar 0.0436.
Random Forest memiliki MAE sebesar 0.0463.
Perbandingan: MAE yang lebih rendah menunjukkan rata-rata kesalahan absolut yang lebih kecil. Linear Regression juga memiliki MAE yang lebih rendah dibandingkan Random Forest. Ini berarti, secara rata-rata, selisih absolut antara prediksi Linear Regression dan nilai sebenarnya lebih kecil.

# *Hubungan Antara Evaluasi Model dan Business Understanding*
Berdasarkan hasil evaluasi model dan Business Understanding yang telah dijelaskan, berikut adalah analisis keterkaitannya:

# Problem Statement
**1. Problem Statements 1: Bagaimana memprediksi peluang diterimanya mahasiswa pada program pascasarjana berdasarkan atribut akademik dan latar belakangnya?**
 Kedua model (Linear Regression dan Random Forest Regressor) telah berhasil memprediksi peluang diterima mahasiswa. Metrik evaluasi menunjukkan bahwa kedua model mampu memberikan prediksi dengan tingkat kesalahan yang relatif rendah. Linear Regression sedikit lebih unggul dalam hal akurasi (R², MAE, MSE, RMSE) dibandingkan Random Forest. Ini berarti model-model ini telah memenuhi kebutuhan untuk menghasilkan prediksi persentase peluang diterima, yang merupakan inti dari problem statement ini.

**2. Problem Statement 2: Fitur apa saja yang memiliki pengaruh besar terhadap kemungkinan diterimanya mahasiswa?**
 Meskipun evaluasi model secara langsung tidak menampilkan fitur-fitur yang paling berpengaruh (ini biasanya dilakukan melalui analisis fitur importance setelah model dilatih), keberhasilan model dalam memprediksi peluang diterima menunjukkan bahwa model telah belajar dari fitur-fitur yang diberikan. Untuk menjawab problem statement ini sepenuhnya, diperlukan langkah tambahan setelah evaluasi, yaitu menganalisis feature importance dari model Random Forest atau koefisien dari Linear Regression. Namun, dari konteks "Goals" yang menyatakan "Mengidentifikasi fitur-fitur yang berkontribusi besar terhadap hasil prediksi", dapat diasumsikan bahwa analisis ini akan dilakukan sebagai bagian dari proyek secara keseluruhan.

# Goals
**Goal 1: Membuat model prediksi regresi yang dapat memprediksi persentase peluang diterima.**
goal ini telah berhasil dicapai. Kedua model, Linear Regression dan Random Forest Regressor, telah diimplementasikan dan dievaluasi. Hasilnya menunjukkan bahwa Linear Regression mampu memprediksi persentase peluang diterima dengan R² Score 0.7959 dan MAE 0.0436, sementara Random Forest Regressor dengan R² Score 0.7727 dan MAE 0.0463. Keduanya memberikan nilai prediksi dalam bentuk persentase, sesuai dengan tujuan.

**Goal 2: Mengidentifikasi fitur-fitur yang berkontribusi besar terhadap hasil prediksi.**
evaluasi metrik secara langsung tidak menunjukkan fitur yang paling berpengaruh. Namun, keberadaan metrik evaluasi yang baik untuk kedua model menunjukkan bahwa model-model ini bekerja dengan efektif dalam memprediksi. Asumsi ini berarti bahwa goal ini juga akan tercapai setelah analisis feature importance dilakukan.

# Dampak Solusi Statements:
**Solution Statement 1: Menerapkan dua model regresi yaitu Linear Regression dan Random Forest Regressor untuk melakukan prediksi.**
Solusi ini berdampak positif dan sangat krusial. Dengan menerapkan kedua model, memungkinkan adanya perbandingan performa. Dari evaluasi, diketahui bahwa Linear Regression sedikit lebih unggul dibandingkan Random Forest Regressor dalam memprediksi peluang diterima. Ini memberikan pilihan model yang paling optimal untuk digunakan dalam implementasi selanjutnya.

**Solution Statement 2: Menggunakan metrik evaluasi MSE, RMSE, MAE, dan R² untuk menilai kedua performa model.**
Solusi ini memiliki dampak yang sangat signifikan dalam memahami dan mengukur kualitas model. Metrik-metrik ini memberikan gambaran yang komprehensif mengenai seberapa baik model bekerja:
**R² Score**: Menunjukkan kemampuan model dalam menjelaskan variabilitas data, di mana Linear Regression lebih baik dalam hal ini. Ini berarti model Linear Regression lebih mampu menangkap pola-pola dalam data peluang penerimaan mahasiswa.
**MAE**: Memberikan gambaran rata-rata kesalahan absolut, menunjukkan bahwa Linear Regression memiliki kesalahan prediksi rata-rata yang lebih kecil. Ini penting dari sisi bisnis karena artinya prediksi Linear Regression lebih dekat ke nilai sebenarnya.
**MSE & RMSE**: Memberikan penalti lebih besar pada kesalahan besar, dan Linear Regression menunjukkan nilai yang lebih rendah. Hal ini mengindikasikan bahwa Linear Regression lebih stabil dan memiliki lebih sedikit "outlier" dalam prediksinya, yang sangat berharga untuk keputusan kritis.

# **Kesimpulan**
Berdasarkan semua metrik evaluasi yang digunakan (R² Score, MSE, RMSE, dan MAE), model Linear Regression menunjukkan performa yang lebih baik dalam memprediksi 'Chance of Admit' dibandingkan dengan model Random Forest Regressor pada dataset ini. Linear Regression memiliki R² Score yang lebih tinggi, serta nilai MSE, RMSE, dan MAE yang lebih rendah, mengindikasikan akurasi prediksi yang lebih baik dan kesalahan yang lebih kecil secara keseluruhan.
Meskipun Random Forest dikenal mampu menangani non-linearitas dan kompleksitas data, pada kasus ini, hubungan antara fitur dan target mungkin cukup linear atau data memiliki pola yang lebih cocok dijelaskan oleh model Linear Regression.
Dengan menggunakan kombinasi metrik MAE, MSE, RMSE, dan R² Score, evaluasi model menjadi lebih komprehensif, memungkinkan pengembang untuk mengidentifikasi kekuatan dan kelemahan model serta melakukan perbaikan yang tepat guna meningkatkan akurasi prediksi dalam proses seleksi mahasiswa pascasarjana.

