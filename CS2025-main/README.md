📘 Judul Proyek
Klasifikasi SMS Spam dengan Deep Learning
👤 Informasi
•	Nama: Mukti Ali Mudlofar
•	Repo: https://github.com/Mukti45/sms_spam_classification.git 
•	Video: [...]
________________________________________
1. 🎯 Ringkasan Proyek
Proyek ini mengimplementasikan sistem klasifikasi SMS spam otomatis menggunakan teknik Natural Language Processing (NLP) dan Deep Learning. Sistem dapat membedakan pesan spam dari pesan normal (ham) dengan akurasi tinggi.
Highlights:
•	✅ Dataset: 5,574 SMS dari UCI Machine Learning Repository 
•	✅ 3 Model: Naive Bayes, Random Forest, LSTM Neural Network 
•	✅ Akurasi Terbaik: 98.93% (LSTM) 
•	✅ 4 Visualisasi EDA komprehensif 
•	✅ Complete reproducible pipeline
________________________________________
2. 📄 Problem & Goals
Problem Statements:
1.	Deteksi Otomatis: Bagaimana membuat sistem yang dapat mendeteksi pesan spam secara otomatis dengan akurasi tinggi tanpa intervensi manual? 
2.	Kompleksitas Bahasa: Pesan spam sering menggunakan variasi kata, singkatan, dan pola bahasa yang kompleks yang sulit diidentifikasi dengan rule-based system. 
3.	Imbalanced Data: Dataset SMS spam umumnya memiliki ketidakseimbangan kelas (lebih banyak ham daripada spam), yang dapat mempengaruhi performa model. 
4.	Performa Model: Diperlukan perbandingan antara model tradisional machine learning dan deep learning untuk menentukan pendekatan terbaik.
Goals:
1.	✅ Membangun model ML untuk mengklasifikasikan SMS spam dengan akurasi minimal 95% 
2.	✅ Mengukur dan membandingkan performa 3 pendekatan model (Baseline, Advanced, Deep Learning) 
3.	✅ Menentukan model terbaik berdasarkan metrik evaluasi (Accuracy, Precision, Recall, F1-Score) 
4.	✅ Menghasilkan sistem yang reproducible dengan dokumentasi lengkap 
5.	✅ Mengidentifikasi pola dan karakteristik pesan spam vs ham melalui EDA
________________________________________
📁 Struktur Folder
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
│   ├── train_naive_bayes.py       # Model 1: Naive Bayes
│   ├── train_random_forest.py     # Model 2: Random Forest
│   └── train_deep_learning.py     # Model 3: LSTM
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
└── main.py                         # Script utama (run all) ________________________________________
3. 📊 Dataset
Informasi Umum
•	Sumber: UCI Machine Learning Repository 
•	Jumlah Data: 5,574 pesan SMS 
•	Distribusi: 
•	Ham (bukan spam): 4,827 (86.6%)
•	Spam: 747 (13.4%)
•	Tipe: Text Data (Natural Language) 
•	Format: TSV (Tab-Separated Values) 
•	Size: 198.6 KB
Fitur Dataset
Fitur	Tipe Data	Deskripsi	Contoh
label	Binary	Label kelas (0=ham, 1=spam)	0, 1
message	Text	Isi pesan SMS asli	"How are you?"
message_length	Integer	Panjang karakter pesan	20, 150
word_count	Integer	Jumlah kata	5, 20
cleaned_message	Text	Teks setelah cleaning	"how are you"
processed_message	Text	Teks setelah preprocessing	"todai" (stemmed)
Karakteristik Data
•	Imbalanced: Rasio Ham:Spam ≈ 6.5:1 
•	No Missing Values: Dataset lengkap 
•	No Duplicates: Tidak ada data duplikat 
•	Language: English (UK/US) 
•	Spam Patterns: Kata-kata seperti "free", "win", "call", "claim" dominan di spam
________________________________________
4. 🔧 Data Preparation
a)	Data Cleaning
•	✅ Handling missing values (tidak ada) 
•	✅ Remove duplicates (tidak ada) 
•	✅ Label encoding (ham=0, spam=1)
b)	Text Preprocessing
•	✅ Lowercase: Standardisasi teks 
•	✅ Remove URLs: Hapus link 
•	✅ Remove special characters: Hanya alfabet 
•	✅ Remove numbers: Hapus angka 
•	✅ Stopwords removal: Hapus kata umum (NLTK) 
•	✅ Stemming: Porter Stemmer (reduce to root form)
c)	Feature Engineering
•	✅ message_length: Panjang karakter 
•	✅ word_count: Jumlah kata 
•	✅ processed_message: Hasil final preprocessing
d)	Data Transformation
Untuk Traditional ML (Naive Bayes & Random Forest):
•	TF-IDF Vectorization 
•	Max features: 2000-3000 
•	N-grams: (1, 2) - unigram + bigram
Untuk Deep Learning (LSTM):
•	Keras Tokenizer (vocab size: 5000) 
•	Sequence padding (max length: 100) 
•	Word embedding (dimension: 128)
e)	Data Splitting
Training set:   70% (3,902 samples)
Validation set: 10% (557 samples)
Test set:       20% (1,115 samples)
•	Stratified split (mempertahankan proporsi kelas) 
•	Random state: 42 (reproducibility)
________________________________________
5. 🤖 Modeling
•	Model 1 – Baseline: Naive Bayes
Algoritma: Multinomial Naive Bayes
Features:
•	TF-IDF vectorization (max_features=3000) 
•	Bigram + unigram
Hyperparameters:
MultinomialNB(alpha=1.0)
Hasil:
•	Accuracy: 98.20% 
•	Training time: ~2 seconds 
•	Model size: 500 KB
•	Model 2 – Advanced ML: Random Forest 
Algoritma: Random Forest Classifier
Features:
•	TF-IDF vectorization (max_features=2000) 
•	Bigram + unigram
Hyperparameters:
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42
)
Hasil:
•	Accuracy: 97.57% 
•	Training time: ~30 seconds 
•	Model size: 15 MB
•	Model 3 – Deep Learning: LSTM Deep Learning
Arsitektur: Bidirectional LSTM Neural Network
Layer Structure:
1. Embedding Layer (vocab_size=5000, dim=128)
2. Bidirectional LSTM (64 units) + Dropout(0.3)
3. Bidirectional LSTM (32 units) + Dropout(0.3)
4. Dense(64, relu) + Dropout(0.5)
5. Dense(32, relu) + Dropout(0.3)
6. Dense(1, sigmoid)
Total parameters: ~786,000
Hyperparameters:
optimizer='adam'
loss='binary_crossentropy'
epochs=20
batch_size=32
early_stopping (patience=5)
Hasil:
•	Accuracy: 98.93% 🏆 
•	Training time: ~10 minutes 
•	Model size: 9.5 MB
________________________________________
6. 🧪 Evaluation
Metrik: Accuracy
Hasil Singkat
Model	Accuracy	Precision	Recall	F1-Score	Training Time
Naive Bayes	0.9820	0.9683	0.9333	0.9505	2s
Random Forest	0.9757	0.9651	0.9150	0.9394	30s
LSTM 🏆	0.9893	0.9862	0.9477	0.9666	10min

________________________________________
7. 🏁 Kesimpulan
•	Model terbaik: LSTM Deep Learning
•	Alasan: 
•	Sequential Context: Memahami urutan kata dan konteks 
•	Automatic Feature Learning: Belajar representasi optimal sendiri 
•	Bidirectional Processing: Membaca dari kedua arah 
•	Word Embeddings: Menangkap semantic similarity
•	Insight penting: 
Dari Data:
•	Spam rata-rata 2x lebih panjang dari ham (138 vs 71 karakter) 
•	Kata "free", "call", "win", "prize" adalah strong spam indicators 
•	86.6% ham, 13.4% spam - imbalanced tapi manageable
Dari Modeling:
•	Baseline (Naive Bayes) sudah sangat kuat: 98.2% 
•	Deep Learning memberikan improvement signifikan meski marginal 
•	Random Forest underperform untuk text data 
•	Precision lebih penting dari recall (false positive mengganggu user)
________________________________________
8. 🔮 Future Work
•	 Tambah data
 Mengumpulkan lebih banyak data (10,000+ messages) 
 Multi-language support (Indonesia, dll) 
 Real-time data collection
•	 Tuning model
 Transformer models (BERT, DistilBERT) 
 Ensemble methods (NB + LSTM) 
 Hyperparameter tuning extensive 
 Active learning untuk continuous improvement
•	 Deployment
 REST API dengan FastAPI/Flask 
 Web app dengan Streamlit 
 Mobile app integration 
 Docker containerization 
 Cloud deployment (AWS/GCP)
•	 Optimization:
 Model compression (TensorFlow Lite) 
 Quantization untuk mobile 
 Inference speed optimization
________________________________________
9. 🔁 Reproducibility
Requirements
Python Version: 3.10+
Main Dependencies:
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tensorflow==2.13.0
keras==2.13.1
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
wordcloud==1.9.2
joblib==1.3.2
ucimlrepo==0.0.3
Installation
# Clone repository
git clone https://github.com/muktialimu/sms-spam-classification.git
cd sms-spam-classification
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
# Install dependencies
pip install -r requirements.txt
Download Dataset
Dataset akan otomatis didownload saat menjalankan script:
python main.py
Atau download manual dari: UCI ML Repository
Running the Project
Option 1: Run Full Pipeline
python main.py
Option 2: Run Individual Modules
# Data preprocessing
python src/data_preprocessing.py
# Exploratory Data Analysis
python src/eda.py

# Train models
python src/train_naive_bayes.py
python src/train_random_forest.py
python src/train_deep_learning.py
Option 3: Google Colab (Recommended)
•	Buka notebook: notebooks/SMS_Spam_Classification_Complete.ipynb 
•	Upload ke Google Colab 
•	Jalankan semua cell (Runtime > Run all) 
•	Estimated time: 20-25 menit
Hardware Specifications
Minimum:
•	CPU: 2 cores
•	RAM: 8 GB
•	Disk: 5 GB free space
Recommended:
•	CPU: 4+ cores
•	RAM: 12 GB
•	GPU: Optional (training lebih cepat)
Tested on:
•	Google Colab (Free Tier)
•	Local: Intel i5, 16GB RAM


