import json
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. LOAD DATASET
# =========================
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

print("Total data:", len(texts))

# =========================
# 2. ENCODE LABEL
# =========================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# =========================
# 3. FEATURE EXTRACTION
# =========================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000
)
X = vectorizer.fit_transform(texts)

# =========================
# 4. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. TRAIN MODEL
# =========================
model = MultinomialNB()
model.fit(X_train, y_train)

# =========================
# 6. EVALUASI MODEL
# =========================
y_pred = model.predict(X_test)

print("\n=== EVALUASI MODEL ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# =========================
# 7. SIMPAN MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("\nModel berhasil disimpan!")
print("File:")
print("- model.pkl")
print("- vectorizer.pkl")
print("- label_encoder.pkl")
