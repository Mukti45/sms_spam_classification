# 📘 Klasifikasi SMS Spam dengan Deep Learning

**Judul Proyek**
Klasifikasi SMS Spam dengan Deep Learning

**👤 Informasi**
* **Nama:** Mukti Ali, Mudlofar
* **Repo:** [https://github.com/Mukti45/sms_spam_classification.git](https://github.com/Mukti45/sms_spam_classification.git)
* **Video:** [...]

---

## 1. 🎯 Ringkasan Proyek

Proyek ini mengimplementasikan sistem **klasifikasi SMS spam otomatis** menggunakan teknik *Natural Language Processing* (NLP) dan *Deep Learning*. Sistem dapat membedakan pesan spam dari pesan normal (ham) dengan akurasi tinggi.

**Highlights:**
* ✅ Dataset: 5,574 SMS dari UCI Machine Learning Repository
* ✅ 3 Model: Naive Bayes, Random Forest, LSTM Neural Network
* ✅ Akurasi Terbaik: 98.93% (LSTM)
* ✅ 4 Visualisasi EDA komprehensif
* ✅ Complete reproducible pipeline

---

## 2. 📄 Problem & Goals

### Problem Statements
1. Deteksi Otomatis: Bagaimana membuat sistem yang dapat mendeteksi pesan spam secara otomatis dengan akurasi tinggi tanpa intervensi manual?
2. Kompleksitas Bahasa: Pesan spam sering menggunakan variasi kata, singkatan, dan pola bahasa yang kompleks yang sulit diidentifikasi dengan rule-based system.
3. Imbalanced Data: Dataset SMS spam umumnya memiliki ketidakseimbangan kelas (lebih banyak ham daripada spam), yang dapat mempengaruhi performa model.
4. Performa Model: Diperlukan perbandingan antara model tradisional machine learning dan deep learning untuk menentukan pendekatan terbaik.

### Goals
1. ✅ Membangun model ML untuk mengklasifikasikan SMS spam dengan akurasi minimal 95%
2. ✅ Mengukur dan membandingkan performa 3 pendekatan model (Baseline, Advanced, Deep Learning)
3. ✅ Menentukan model terbaik berdasarkan metrik evaluasi (Accuracy, Precision, Recall, F1-Score)
4. ✅ Menghasilkan sistem yang reproducible dengan dokumentasi lengkap
5. ✅ Mengidentifikasi pola dan karakteristik pesan spam vs ham melalui EDA

---

## 📁 Struktur Folder

```text
sms-spam-classification/
│
├── data/                           # Dataset (gitignore)
│   ├── raw/                        # Data mentah
│   │   └── SMSSpamCollection       # Dataset dari UCI (download otomatis)
│   └── processed/                  # Data terproses
│       └── processed_data.csv      # Hasil preprocessing
│
├── notebooks/                      # Jupyter notebooks
│   └── SMS_Spam_Classification_Complete.ipynb  # Full pipeline
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning & preprocessing
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── train_naive_bayes.py        # Model 1: Naive Bayes
│   ├── train_random_forest.py      # Model 2: Random Forest
│   └── train_deep_learning.py      # Model 3: LSTM
│
├── models/                         # Saved models (gitignore)
│   ├── naive_bayes_model.pkl       # Trained Naive Bayes
│   ├── random_forest_model.pkl     # Trained Random Forest
│   ├── lstm_model.h5               # Trained LSTM
│   └── lstm_tokenizer.pkl          # LSTM Tokenizer
│
├── results/                        # Hasil evaluasi
│   ├── figures/                    # Visualisasi
│   │   ├── 1_class_distribution.png
│   │   ├── 2_message_length.png
│   │   ├── 3_wordclouds.png
│   │   ├── 4_top_words.png
│   │   ├── cm_naive_bayes.png
│   │   ├── cm_random_forest.png
│   │   ├── cm_lstm.png
│   │   ├── feature_importance_rf.png
│   │   ├── lstm_training_history.png
│   │   └── model_comparison.png
│   └── model_comparison.csv        # Tabel perbandingan
│
├── .gitignore                      # Git ignore file
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi ini
└── main.py                         # Script utama (run all)

3. 📊 Dataset

### Informasi Umum
* Sumber: UCI Machine Learning Repository
* Jumlah Data: 5,574 pesan SMS
* Distribusi:
    * Ham (bukan spam): 4,827 (86.6%)
    * Spam: 747 (13.4%)
* Tipe: Text Data (Natural Language)
* Format: TSV (Tab-Separated Values)
* Size: 198.6 KB

### Fitur Dataset

| Fitur | Tipe Data | Deskripsi | Contoh |
| :--- | :--- | :--- | :--- |
| `label` | Binary | Label kelas (0=ham, 1=spam) | 0, 1 |
| `message` | Text | Isi pesan SMS asli | "How are you?" |
| `message_length` | Integer | Panjang karakter pesan | 20, 150 |
| `word_count` | Integer | Jumlah kata | 5, 20 |
| `cleaned_message` | Text | Teks setelah cleaning | "how are you" |
| `processed_message` | Text | Teks setelah preprocessing | "todai" (stemmed) |

### Karakteristik Data
* Imbalanced: Rasio Ham:Spam ≈ 6.5:1
* No Missing Values: Dataset lengkap
* No Duplicates: Tidak ada data duplikat
* Language: English (UK/US)
* Spam Patterns: Kata-kata seperti "free", "win", "call", "claim" dominan di spam
