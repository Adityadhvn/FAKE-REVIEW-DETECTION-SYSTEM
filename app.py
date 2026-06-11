import streamlit as st
import time

from src.predict_distilbert import predict_review as predict_distilbert
from src.predict_logistic import predict_review as predict_logistic


# Page Config


st.set_page_config(
    page_title="Fake Review Detection",
    layout="centered"
)



# Label Mapping

LABELS = {
    0: "Original Human Review",
    1: "Fake Review"
}

SAMPLE_REVIEWS = [
    "The product arrived on time and works exactly as described. Build quality is good and I am satisfied with the purchase.",

    "Product is fine, wish it had an option to have a separate handle.",

    "Packaging was decent and delivery was fast. The item performs well and matches the description provided by the seller.",

    "Works as expected for connecting CCTV power supply to cameras.",

    "Easy to hold, and the material is nice and thick. I will keep my shelves in order and this product will help me do it!"
]

if "review_index" not in st.session_state:
    st.session_state.review_index = 0

if "review_text" not in st.session_state:
    st.session_state.review_text = ""



st.title("Fake Review Detection System")

st.markdown(
    "<p style='margin-top:-15px; color:black; font-size:20px'>By Aditya Dhawan, DTU</p>",
    unsafe_allow_html=True
)

st.markdown(
    """
Detect whether a review is an original human-written review
or a computer-generated review using machine learning models.
"""
)

st.divider()

# Review Input


if st.button("Test Random Review"):

    st.session_state.review_text = SAMPLE_REVIEWS[
        st.session_state.review_index
    ]

    st.session_state.review_index = (
        st.session_state.review_index + 1
    ) % len(SAMPLE_REVIEWS)

review = st.text_area(
    "Enter Review",
    value=st.session_state.review_text,
    height=200,
    placeholder="Paste a product review here..."
)


# Model Selection


model_choice = st.selectbox(
    "Select Model",
    [
        "DistilBERT (96.0% Confidence)",
        "TF-IDF + Logistic Regression (89.85% Confidence)"
    ]
)


if st.button("Analyze Review", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a review.")

    else:

        start_time = time.time()
        try:

            # DistilBERT
            if model_choice.startswith("DistilBERT"):

                prediction, confidence = predict_distilbert(
                    review
                )

                model_used = "DistilBERT"

            # Logistic Regression
            else:

                prediction, confidence = predict_logistic(
                    review
                )

                model_used = "TF-IDF + Logistic Regression"

            end_time = time.time()

            elapsed_time = round(
                end_time - start_time,
                3
            )

            prediction_text = LABELS[
                int(prediction)
            ]

            confidence_percent = (
                float(confidence) * 100
            )

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 0:

                st.success(
                    f"{prediction_text}"
                )

            else:

                st.error(
                    f"{prediction_text}"
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Confidence",
                    f"{confidence_percent:.2f}%"
                )

            with col2:

                st.metric(
                    "Model",
                    model_used
                )

            with col3:

                st.metric(
                    "Time",
                    f"{elapsed_time}s"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )


st.divider()

st.caption(
    "Models: DistilBERT (96.0% Confidence) | TF-IDF + Logistic Regression (89.85% Confidence)"
)