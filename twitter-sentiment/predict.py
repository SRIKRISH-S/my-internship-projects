# predict.py
import joblib

vec = joblib.load("model/tfidf_vectorizer.joblib")
clf = joblib.load("model/logreg_model.joblib")

def predict(text):
    v = vec.transform([text])
    pred = clf.predict(v)[0]
    return pred

if __name__ == "__main__":
    print(predict("I love this! It's amazing."))
    print(predict("I hate this, very bad experience."))
