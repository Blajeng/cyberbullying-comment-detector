# Aplikasi Deteksi Cyberbullying Pada Komentar Media Sosial (Instagram & TikTok)

Proyek ini adalah sistem cerdas berbasis Web yang dirancang untuk mendeteksi dan mengklasifikasikan komentar perundungan siber (*cyberbullying*) secara otomatis menggunakan pendekatan Machine Learning dan Natural Language Processing (NLP). Aplikasi ini mempermudah proses moderasi konten dengan memisahkan komentar negatif (toksik) dan komentar aman (positif) secara *real-time*.

---

## Tautan Aplikasi Publik (Cloud Deployment)
Aplikasi ini telah di-deploy ke cloud publik dan dapat diuji coba secara langsung melalui tautan daring berikut:
**[cyberbullying-comment-detector.streamlit.app](https://cyberbullying-comment-detector.streamlit.app/)**

---

## 1. Problem Identification

Pertumbuhan pengguna media sosial seperti Instagram dan TikTok memicu lonjakan volume komentar teks yang masif setiap harinya. Fenomena ini diiringi oleh tingginya angka perundungan siber (*cyberbullying*) di kolom komentar. Jika proses penyaringan konten negatif ini hanya mengandalkan tenaga manusia (*human moderator*), hal tersebut tidak lagi efektif karena keterbatasan waktu dan besarnya volume data teks yang masuk secara cepat. Keterlambatan penanganan komentar toksik ini berdampak buruk bagi psikologis pengguna dan kenyamanan ekosistem digital.

Metode konvensional yang mengandalkan pencocokan kata kunci kaku (*keyword filtering*) sangat mudah dikelabuhi oleh pengguna. Pelaku perundungan sering kali memanipulasi ketikan menggunakan singkatan, bahasa gaul (slang), atau salah ketik yang disengaja (*typo*).

Oleh karena itu, solusi berbasis **Artificial Intelligence (Natural Language Processing)** sangat relevan karena mampu:
1. **Memahami Konteks Kalimat:** Algoritma NLP mempelajari pola kemunculan kata dan kombinasi kata yang berkorelasi dengan komentar cyberbullying berdasarkan data pelatihan, sehingga mampu melakukan klasifikasi secara lebih fleksibel dibanding pendekatan keyword filtering sederhana.
2. **Skalabilitas Tinggi:** Model mampu melakukan prediksi data baru dalam hitungan milidetik, sehingga penyaringan konten dapat dilakukan secara otomatis sebelum dibaca oleh pengguna lain.

---

## 2. Data Processing & AI Pipeline

### Sumber Dataset
Master data yang digunakan dalam proyek ini berjumlah **3.226 baris data komentar** yang diintegrasikan dari 4 sumber riset lintas platform secara transparan:
1. **Kaggle** (Dataset Cita Tiara Hani)
2. **HuggingFace** (Dataset aditdwi123)
3. **GitHub** (Dataset rizalespe)
4. **TikTok Dataset** (Dataset Prameswari et al.)

### Pembersihan Data (Preprocessing)
Eksperimen awal didokumentasikan lengkap pada file `notebook_eksperimen.ipynb`. Tahapan pembersihan meliputi:
* **Case Folding:** Menyamaratakan seluruh huruf menjadi huruf kecil.
* **Text Cleansing:** Menghapus angka, tanda baca, emoji, tautan URL, serta tag khusus seperti `<username>` atau `@mention`.
* **Normalisasi Kata Gaul:** Memperbaiki singkatan populer media sosial (`bgt`, `gk`, `yg`) dan menyeragamkan berbagai variasi kata makian kasar ke dalam rumpun kata dasar tertentu agar meningkatkan sensitivitas model dalam mengenali pola teks toksik.

### Analisis Data Eksploratif (EDA)
Karakteristik sebaran data ditampilkan langsung pada antarmuka aplikasi dalam bentuk diagram lingkaran (*pie chart*) untuk memantau keseimbangan kelas data, serta pemetaan kata-kata yang paling dominan muncul pada masing-masing sentimen menggunakan grafik *WordCloud*.

---

## 3. Model & Hasil Pengujian Kuantitatif

Ekstraksi fitur teks diubah ke dalam bentuk numerik menggunakan metode **TF-IDF Vectorizer** (maksimal 3.000 fitur). Proyek ini melakukan eksperimen komparatif antara dua algoritma klasifikasi yang berbeda untuk mencari performa terbaik:

Metode Evaluasi:
- Train-Test Split : 80% Training, 20% Testing
- Stratified Sampling digunakan untuk menjaga distribusi kelas.
- Random State : 42

* **Logistic Regression (Model Utama):** Mendapatkan akurasi sebesar **75.54%** (Dipilih sebagai penggerak utama sistem karena performanya yang paling optimal pada dataset lintas platform).
* **Multinomial Naïve Bayes:** Mendapatkan akurasi sebesar **72.91%**.


### A. Classification Report (Model Utama: Logistic Regression)

Berikut adalah metrik evaluasi kuantitatif terukur untuk menguji kemampuan model dalam mengklasifikasikan teks:

| Kategori (Sentimen) | Precision | Recall | F1-Score | Jumlah Data Uji (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Cyberbullying (Negative)** | 78% | 73% | 76% | 332 komentar |
| **Aman / Netral (Positive)** | 73% | 78% | 76% | 314 komentar |
| **Rata-rata / Total (Accuracy)** | | | **75,54%** | **646 komentar** |

---

## 4. Antarmuka Aplikasi (Streamlit UI)

Aplikasi ini dibungkus menggunakan framework Streamlit dengan arsitektur **2-Tab Utama**:
* **Tab 1 (Analisis Komparatif Data):** Memuat visualisasi statistik sebaran data master, sampel acak isi database, serta grafik awan kata (*WordCloud*) dari komentar cyberbullying (merah) dan komentar aman (hijau).
* **Tab 2 (Uji Coba Deteksi Komentar):** Menyediakan kolom input bagi pengguna untuk mengetik teks bebas secara acak. Sistem akan memproses teks tersebut melalui pipa preprocessing dan langsung mengeluarkan hasil keputusan sistem (Aman/Cyberbullying) beserta skor kepastiannya (*Confidence Score*).

---

## 5. Struktur Repositori
```text
├── app.py
├── notebook_eksperimen.ipynb
├── dataset_gabungan.csv
├── best_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
```
---

## 6. Panduan Menjalankan Aplikasi Secara Lokal

### 1. Klon Repositori
Buka terminal atau command prompt komputer Anda, lalu jalankan perintah:
```bash
git clone https://github.com/Blajeng/cyberbullying-comment-detector.git https://github.com/Blajeng/cyberbullying-comment-detector.git
cd cyberbullying-comment-detector
```

2. Instalasi Library / Dependencies
Pastikan perangkat komputer Anda sudah terpasang lingkungan Python. Jalankan perintah berikut untuk mengunduh pustaka pendukung:
```bash
pip install -r requirements.txt
```
3. Menjalankan Server Aplikasi
Luncurkan aplikasi web Streamlit di komputer lokal Anda dengan perintah:

```Bash
streamlit run app.py
```
Aplikasi akan secara otomatis dimuat dan dapat diakses langsung pada peramban web browser melalui alamat lokal http://localhost:8501.



