# train_sentiment.py

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("Loading dataset...")

# Load tweet_eval sentiment dataset
ds = load_dataset("tweet_eval", "sentiment")

# Convert to a DataFrame
df = pd.DataFrame({
    "text": ds["train"]["text"],
    "label": ds["train"]["label"]
})

print("Dataset loaded. Total rows:", len(df))

# Prepare text & labels
X = df["text"].astype(str).tolist()
y = df["label"].tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, y_train)

# Predictions
preds = clf.predict(Xte)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Saving model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(vec, "model/tfidf_vectorizer.joblib")
joblib.dump(clf, "model/logreg_model.joblib")

print("\nModel saved successfully in folder: model/")
