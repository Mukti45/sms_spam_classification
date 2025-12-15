# 📘 Klasifikasi SMS Spam dengan Deep Learning

**Authors:**
* Mukti Ali
* Mudlofar

[Link Repository](https://github.com/Mukti45/sms_spam_classification.git)

---

## 🎯 Ringkasan Proyek

Proyek ini mengimplementasikan sistem **klasifikasi SMS spam otomatis** menggunakan teknik *Natural Language Processing* (NLP) dan *Deep Learning*. Sistem dapat membedakan pesan spam dari pesan normal (ham) dengan akurasi tinggi.

**Highlights:**
* ✅ **Dataset:** 5,574 SMS dari UCI Machine Learning Repository
* ✅ **3 Model:** Naive Bayes, Random Forest, LSTM Neural Network
* ✅ **Akurasi Terbaik:** 98.93% (LSTM)
* ✅ **EDA:** 4 Visualisasi komprehensif
* ✅ **Pipeline:** Complete reproducible pipeline

---

## 📄 Problem & Goals

### Problem Statements
1.  **Deteksi Otomatis:** Bagaimana membuat sistem deteksi spam otomatis akurasi tinggi tanpa intervensi manual?
2.  **Kompleksitas Bahasa:** Variasi kata, singkatan, dan pola bahasa kompleks pada spam sulit diidentifikasi rule-based system.
3.  **Imbalanced Data:** Ketidakseimbangan kelas (Ham > Spam) yang mempengaruhi performa model.
4.  **Performa Model:** Membandingkan model tradisional ML vs Deep Learning.

### Goals
1.  ✅ Membangun model ML dengan akurasi minimal **95%**.
2.  ✅ Membandingkan 3 pendekatan model (Baseline, Advanced, Deep Learning).
3.  ✅ Menentukan model terbaik berdasarkan Accuracy, Precision, Recall, dan F1-Score.
4.  ✅ Menghasilkan sistem yang reproducible.
5.  ✅ Mengidentifikasi pola karakteristik pesan spam vs ham melalui EDA.

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

📊 Dataset
Sumber: UCI Machine Learning Repository

Jumlah Data: 5,574 pesan SMS

Distribusi:

Ham (normal): 4,827 (86.6%)

Spam: 747 (13.4%)

Fitur Dataset
Fitur	Tipe Data	Deskripsi	Contoh
label	Binary	Label kelas (0=ham, 1=spam)	0, 1
message	Text	Isi pesan SMS asli	"How are you?"
message_length	Integer	Panjang karakter pesan	20, 150
word_count	Integer	Jumlah kata	5, 20
cleaned_message	Text	Teks setelah cleaning	"how are you"
processed_message	Text	Teks setelah preprocessing	"todai" (stemmed)

Data Preparation
Data Cleaning: Handling missing values, remove duplicates, label encoding.

Text Preprocessing: Lowercase, remove URLs/Special Chars/Numbers, Stopwords removal (NLTK), Stemming (Porter Stemmer).

Feature Engineering: Menambah fitur message_length dan word_count.

🔧 Data Transformation:

Traditional ML: TF-IDF Vectorization (Max features: 2000-3000, N-grams: 1,2).

Deep Learning: Keras Tokenizer (Vocab: 5000), Padding (Max len: 100), Word Embedding (Dim: 128).

Data Splitting: Train (70%), Validation (10%), Test (20%) dengan Stratified Split.

🤖 Modeling
Model 1: Naive Bayes (Baseline)
Algoritma: Multinomial Naive Bayes

Features: TF-IDF (3000 features)

Hasil: Accuracy 98.20%

Model 2: Random Forest (Advanced ML)
Algoritma: Random Forest Classifier

Params: n_estimators=100, max_depth=20

Hasil: Accuracy 97.57%

Model 3: LSTM (Deep Learning)
Arsitektur: Bidirectional LSTM

Structure: Embedding -> Bi-LSTM (64) -> Bi-LSTM (32) -> Dense layers

Params: Optimizer Adam, Binary Crossentropy

Hasil: Accuracy 98.93% 🏆

🧪 Evaluation

Model	Accuracy	Precision	Recall	F1-Score	Training Time
Naive Bayes	0.9820	0.9683	0.9333	0.9505	~2s
Random Forest	0.9757	0.9651	0.9150	0.9394	~30s
LSTM 🏆	0.9893	0.9862	0.9477	0.9666	~10min

🏁 Kesimpulan
Model Terbaik: LSTM Deep Learning

Alasan: Mampu memahami konteks urutan kata (sequential context) dan menangkap semantic similarity melalui Word Embeddings.

Insight:

Pesan Spam rata-rata 2x lebih panjang dari Ham.

Kata kunci kuat: "free", "call", "win", "prize".

Precision lebih diprioritaskan untuk menghindari False Positive (pesan penting masuk spam).

🔮 Future Work
Data: Tambah dataset (10k+) dan support Multi-language (Indonesia).

Model: Implementasi Transformers (BERT/DistilBERT) dan Ensemble Methods.

Deployment: REST API (FastAPI), Web App (Streamlit), Dockerize.

Optimasi: Model quantization untuk mobile deployment.

🔁 Reproducibility
Requirements
Python 3.10+

TensorFlow, Scikit-learn, Pandas, NLTK (lihat requirements.txt)

nstallation
Bash

# Clone repository
git clone [https://github.com/muktialimu/sms-spam-classification.git](https://github.com/muktialimu/sms-spam-classification.git)
cd sms-spam-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

Running the Project
Opsi 1: Run Full Pipeline

Bash

python main.py
Opsi 2: Run via Google Colab

Buka notebooks/SMS_Spam_Classification_Complete.ipynb

Upload ke Google Colab

Runtime > Run All
