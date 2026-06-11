from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "adityadhawan/FAKE-REVIEW-DETECTION-DistilBERT"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Success!")