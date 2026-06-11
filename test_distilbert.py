from src.predict_logistic import predict_review

pred, conf = predict_review(
    "This product is absolutely amazing and exceeded my expectations."
)

print("Prediction:", pred)
print("Confidence:", conf)