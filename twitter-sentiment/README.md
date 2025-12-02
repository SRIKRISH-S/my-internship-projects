# Twitter Sentiment Analysis (TF-IDF + Logistic Regression)

This project classifies tweets into **Positive**, **Negative**, or **Neutral** sentiment using TF-IDF features and Logistic Regression.

---

## 📌 Overview
Dataset: **HuggingFace tweet_eval (sentiment subset)**  
Classes:
- `0` = Negative  
- `1` = Neutral  
- `2` = Positive  

The model extracts text features using **TF-IDF** and trains a **Logistic Regression** classifier.

---

## 🚀 Features
- Uses HuggingFace datasets  
- Fast and lightweight model  
- Demo script for real-time sentiment testing  
- Model saved using joblib (`.joblib` files)

---

## 📂 Project Files
```
twitter-sentiment/
│── train_sentiment.py         # Model training script
│── sentiment_demo.py          # Real-time sentiment prediction
│── model/
│     ├── tfidf_vectorizer.joblib
│     └── logreg_model.joblib
```

---

## ▶️ How to Train
```
python train_sentiment.py
```

## ▶️ How to Test (Real-time Demo)
```
python sentiment_demo.py
```

You can type:
```
I love this!
This is bad.
It's okay.
```

And the model will output:
- Positive 🙂
- Negative 😡
- Neutral 😐

---

## 📈 Example Output
- Training accuracy printed in terminal  
- Classification report shown after training  

---

## 👤 Author
**SRIKRISH S**  
GitHub: https://github.com/SRIKRISH-S
