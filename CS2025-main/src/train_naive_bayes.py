
# train_naive_bayes.py
# Naive Bayes SMS Spam Classification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Asumsikan X_train, y_train, X_test, y_test sudah tersedia
# (misalnya dari pemrosesan dan pembagian data)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

joblib.dump({
    'vectorizer': vectorizer,
    'model': model
}, 'naive_bayes_model.pkl')

print("Model Naive Bayes disimpan!")
