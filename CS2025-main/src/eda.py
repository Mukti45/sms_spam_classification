# CELL 3: EXPLORATORY DATA ANALYSIS (EDA)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import os

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

os.makedirs("results/figures", exist_ok=True)

# ============================================================
# 1️⃣ CLASS DISTRIBUTION
# ============================================================
counts = df["label"].value_counts()

plt.figure()
sns.barplot(x=["Ham", "Spam"], y=counts.values)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/figures/class_distribution.png", dpi=300)
plt.show()

print("\n📊 Class Distribution:")
print(counts)

# ============================================================
# 2️⃣ MESSAGE LENGTH ANALYSIS
# ============================================================
plt.figure()
sns.histplot(df[df["label"]==0]["message_length"], label="Ham", color="green", bins=40)
sns.histplot(df[df["label"]==1]["message_length"], label="Spam", color="red", bins=40)
plt.legend()
plt.title("Message Length Distribution")
plt.xlabel("Characters")
plt.tight_layout()
plt.savefig("results/figures/message_length.png", dpi=300)
plt.show()

print("\n📊 Average Message Length:")
print(df.groupby("label")["message_length"].mean())

# ============================================================
# 3️⃣ WORD CLOUD
# ============================================================
def create_wordcloud(text, title, filename):
    wc = WordCloud(background_color="white", max_words=100).generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

create_wordcloud(
    " ".join(df[df["label"]==0]["cleaned_message"]),
    "Word Cloud - HAM",
    "results/figures/wordcloud_ham.png"
)

create_wordcloud(
    " ".join(df[df["label"]==1]["cleaned_message"]),
    "Word Cloud - SPAM",
    "results/figures/wordcloud_spam.png"
)

# ============================================================
# 4️⃣ TOP WORDS
# ============================================================
def top_words(text, title, filename):
    words = text.split()
    top = Counter(words).most_common(10)
    df_plot = pd.DataFrame(top, columns=["word", "count"])

    plt.figure()
    sns.barplot(y="word", x="count", data=df_plot)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

top_words(
    " ".join(df[df["label"]==0]["processed_message"]),
    "Top Words - HAM",
    "results/figures/top_words_ham.png"
)

top_words(
    " ".join(df[df["label"]==1]["processed_message"]),
    "Top Words - SPAM",
    "results/figures/top_words_spam.png"
)

# ============================================================
# SUMMARY
# ============================================================
print("\n📌 KEY INSIGHTS:")
print("- Dataset tidak seimbang (Ham dominan)")
print("- Spam cenderung lebih panjang")
print("- Spam mengandung kata promosi: free, call, win, claim")
print("- Perbedaan kosakata Ham vs Spam cukup jelas")

print("\n✅ EDA COMPLETED SUCCESSFULLY!")
