import joblib

# Load model and vectorizer
vec = joblib.load("model/tfidf_vectorizer.joblib")
clf = joblib.load("model/logreg_model.joblib")

def predict_sentiment(text):
    x = vec.transform([text])
    pred = clf.predict(x)[0]

    if pred == 0:
        return "Negative 😡"
    elif pred == 1:
        return "Neutral 😐"
    else:
        return "Positive 🙂"

print("Sentiment Demo (type 'exit' to stop)")
while True:
    text = input("Enter a sentence or tweet: ")

    if text.lower() == "exit":
        break

    print("Sentiment:", predict_sentiment(text))
    print()
