
#@title CELL 2: DATA PREPROCESSING
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DATA PREPROCESSING")
print("="*60)

# ============================================================
# 1️⃣ SETUP NLTK
# ============================================================
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

print(f"✓ Stopwords loaded: {len(stop_words)} words")

# ============================================================
# 2️⃣ PREPROCESSING FUNCTIONS
# ============================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_stem(text):
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ============================================================
# 3️⃣ APPLY PREPROCESSING
# ============================================================
print("\n🔧 Applying preprocessing...")

df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))
df['cleaned_message'] = df['message'].apply(clean_text)
df['processed_message'] = df['cleaned_message'].apply(tokenize_stem)

print("✓ Preprocessing completed")

# ============================================================
# 4️⃣ STATISTICS
# ============================================================
print("\n📊 PREPROCESSING STATISTICS")

vocab_before = len(set(' '.join(df['cleaned_message']).split()))
vocab_after  = len(set(' '.join(df['processed_message']).split()))

print(f"Vocabulary before : {vocab_before:,}")
print(f"Vocabulary after  : {vocab_after:,}")
print(f"Reduction         : {(vocab_before - vocab_after)/vocab_before*100:.2f}%")

print(f"\nAvg message length (chars):")
print(f"  Original : {df['message_length'].mean():.2f}")
print(f"  Cleaned  : {df['cleaned_message'].apply(len).mean():.2f}")
print(f"  Processed: {df['processed_message'].apply(len).mean():.2f}")

# ============================================================
# 5️⃣ EXAMPLES (UNTUK LAPORAN)
# ============================================================
print("\n📝 PREPROCESSING EXAMPLES")

for i in [0, 10, 100]:
    print(f"\nExample {i+1}")
    print("Label    :", "SPAM" if df['label'].iloc[i] == 1 else "HAM")
    print("Original :", df['message'].iloc[i][:80])
    print("Cleaned  :", df['cleaned_message'].iloc[i][:80])
    print("Processed:", df['processed_message'].iloc[i][:80])

# ============================================================
# 6️⃣ DATA QUALITY CHECK
# ============================================================
empty_msg = (df['processed_message'].str.strip() == '').sum()
short_msg = (df['processed_message'].apply(lambda x: len(x.split()) < 2)).sum()

print("\n🔍 DATA QUALITY CHECK")
print(f"Empty processed messages : {empty_msg}")
print(f"Very short messages (<2 words): {short_msg}")

# ============================================================
# 7️⃣ SAVE PROCESSED DATA
# ============================================================
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/processed_data.csv'
df.to_csv(output_path, index=False)

print(f"\n💾 Processed data saved to: {output_path}")

# ============================================================
# 8️⃣ DOWNLOAD (COLAB)
# ============================================================
try:
    from google.colab import files
    files.download(output_path)
    print("✓ File downloaded")
except:
    print("✓ File saved locally")

print("\n✅ DATA PREPROCESSING COMPLETED")
print("="*60)
