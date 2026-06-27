# Fake Review Detection System

By Aditya Dhawan, DTU

A machine learning application that classifies reviews as:

- Original Human Review
- Computer Generated Review

## Demo

[[Streamlit App Link]](https://fake-review-detector-aditya-dhawan.streamlit.app/)


.
## Dataset Used 
The dataset used for training these models is from Kaggle.com from user Mexwell. Acknowlegement
Foto von Brett Jordan auf Unsplash
[[Dataset Link]](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

It has 50,000 reviews (50-50) fake and real reviews.

## Models: DistilBERT & TF-IDF Logistic Regression

DistilBERT weights hosted on Hugging Face:
https://huggingface.co/adityadhawan/FAKE-REVIEW-DETECTION-DistilBERT


DistilBERT - 96.0% Confidence

TF-IDF + Logistic Regression - 89.85% Confidence


## DistilBERT Confusion Matrix

![DistilBERT Confusion Matrix](Screenshots/distilbert_confusion_matrix.png)

---

## TF-IDF + Logistic Regression Confusion Matrix

![TF-IDF Confusion Matrix](Screenshots/logistic_confusion_matrix.png)


## Features
- Real-time review classification
- Multiple model support
- Confidence score display
- Hugging Face integration
- Streamlit web interface

