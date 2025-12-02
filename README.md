# My Internship Projects

This repository contains two Machine Learning projects completed during my internship.

---

## 📌 1. MNIST Handwritten Digit Recognition (TensorFlow/Keras)

A Convolutional Neural Network (CNN) trained on the MNIST handwritten digit dataset.

### ⭐ Features
- Achieves ~99% test accuracy  
- Model saved in `.keras` format  
- Includes demo script to view predictions visually  
- Includes training loss graph  

### 📂 Important Files
- `mnist-cnn/train_mnist.py`
- `mnist-cnn/show_mnist_demo.py`
- `mnist-cnn/saved_model/mnist_cnn.keras`
- `mnist-cnn/training_loss.png`

---

## 📌 2. Twitter Sentiment Analysis (TF-IDF + Logistic Regression)

Classifies tweets into **Positive**, **Neutral**, or **Negative** sentiment.

### ⭐ Features
- Uses HuggingFace `tweet_eval` dataset  
- TF-IDF vectorizer for feature extraction  
- Logistic Regression classifier  
- Real-time sentiment prediction demo  

### 📂 Important Files
- `twitter-sentiment/train_sentiment.py`
- `twitter-sentiment/sentiment_demo.py`
- `twitter-sentiment/model/tfidf_vectorizer.joblib`
- `twitter-sentiment/model/logreg_model.joblib`

---

## 🚀 How to Use the Projects

### ▶️ MNIST Demo
```
cd mnist-cnn
python show_mnist_demo.py
```

### ▶️ Sentiment Demo
```
cd twitter-sentiment
python sentiment_demo.py
```

Both demos show real working output.

---

## 📬 Contact
Created by **SRIKRISH S**  
GitHub Profile: https://github.com/SRIKRISH-S
