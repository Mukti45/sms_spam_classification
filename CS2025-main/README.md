# 📘 Klasifikasi SMS Spam dengan Deep Learning

👤 **Nama**: Mukti Ali Mudlofar  
📦 **Repository**: https://github.com/Mukti45/sms_spam_classification.git  
🎥 **Video**: [...]

---

## 🎯 Ringkasan Proyek

Proyek ini mengimplementasikan sistem klasifikasi SMS spam otomatis menggunakan teknik  
**Natural Language Processing (NLP)** dan **Deep Learning**.  
Sistem dapat membedakan pesan spam dari pesan normal (ham) dengan akurasi tinggi.

### 🔍 Highlights
- ✅ Dataset: **5.574 SMS** dari UCI Machine Learning Repository
- ✅ Model: Naive Bayes, Random Forest, LSTM Neural Network
- ✅ Akurasi Terbaik: **98.93% (LSTM)**
- ✅ Visualisasi EDA komprehensif
- ✅ Complete reproducible pipeline

---

## 📄 Problem & Goals

### ❓ Problem Statements
1. **Deteksi Otomatis**  
   Bagaimana membuat sistem yang dapat mendeteksi pesan spam secara otomatis dengan akurasi tinggi tanpa intervensi manual?
2. **Kompleksitas Bahasa**  
   Pesan spam menggunakan variasi kata, singkatan, dan pola bahasa kompleks yang sulit dideteksi rule-based system.
3. **Imbalanced Data**  
   Dataset memiliki lebih banyak ham dibanding spam.
4. **Performa Model**  
   Diperlukan perbandingan ML tradisional dan Deep Learning.

### 🎯 Goals
1. ✅ Akurasi minimal **95%**
2. ✅ Membandingkan 3 pendekatan model
3. ✅ Evaluasi dengan Accuracy, Precision, Recall, F1-Score
4. ✅ Sistem reproducible dengan dokumentasi lengkap
5. ✅ Identifikasi pola spam vs ham melalui EDA

---

## 📁 Struktur Folder

```text
sms-spam-classification/
├── data/
│   ├── raw/
│   │   └── SMSSpamCollection
│   └── processed/
│       └── processed_data.csv
├── notebooks/
│   └── SMS_Spam_Classification_Complete.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train_naive_bayes.py
│   ├── train_random_forest.py
│   └── train_deep_learning.py
├── models/
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── lstm_model.h5
│   └── lstm_tokenizer.pkl
├── results/
│   ├── figures/
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
│   └── model_comparison.csv
├── .gitignore
├── requirements.txt
├── README.md
└── main.py

==================================================

4. 📊 Dataset

Sumber        : UCI Machine Learning Repository
Jumlah Data  : 5.574 pesan SMS
Tipe         : Text (Natural Language)
Format       : TSV
Ukuran       : 198.6 KB

Distribusi:
- Ham  : 4.827 (86.6%)
- Spam :   747 (13.4%)

Fitur Dataset:
- label              : 0 = ham, 1 = spam
- message            : isi pesan SMS
- message_length     : panjang karakter pesan
- word_count         : jumlah kata
- cleaned_message    : teks setelah cleaning
- processed_message  : teks setelah preprocessing

Karakteristik:
- Imbalanced (Ham:Spam ≈ 6.5:1)
- No missing values
- No duplicates
- Bahasa: English (UK/US)

==================================================

5. 🔧 Data Preparation

- Data cleaning & encoding
- Text preprocessing (lowercase, stopwords, stemming)
- Feature engineering
- TF-IDF (ML) & Tokenizer (DL)
- Stratified split (70/10/20)

==================================================

6. 🤖 Modeling

Model 1: Naive Bayes
- Accuracy: 98.20%
- Training time: ~2s

Model 2: Random Forest
- Accuracy: 97.57%
- Training time: ~30s

Model 3: LSTM (Best Model)
- Accuracy: 98.93%
- Training time: ~10 menit

==================================================

7. 🧪 Evaluation

Model           Accuracy   Precision   Recall   F1
Naive Bayes     0.9820     0.9683      0.9333   0.9505
Random Forest   0.9757     0.9651      0.9150   0.9394
LSTM            0.9893     0.9862      0.9477   0.9666

==================================================

8. 🏁 Kesimpulan

Model terbaik adalah LSTM Deep Learning.
Alasan: konteks urutan kata, feature learning otomatis,
dan bidirectional processing.

==================================================

9. 🔁 Reproducibility

Clone repo:
git clone https://github.com/Mukti45/sms_spam_classification.git

Install:
pip install -r requirements.txt

Run:
python main.py
```
