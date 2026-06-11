import joblib

model = joblib.load(
    "Models/TFIDF_LogisticRegression/logistic_regression_model.pkl"
)

vectorizer = joblib.load(
    "Models/TFIDF_LogisticRegression/tfidf_vectorizer.pkl"
)

LABELS = {
    0: "Original Review",
    1: "Computer Generated Review"
}

def predict_review(review):

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector)[0]

    confidence = max(probability)

    return prediction, confidence