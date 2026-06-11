import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "adityadhawan/FAKE-REVIEW-DETECTION-DistilBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

LABELS = {
    0: "Original Review",
    1: "Computer Generated Review"
}
def predict_review(review):

    inputs = tokenizer(
        review,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )



    return (
        prediction.item(),
        confidence.item()
    )