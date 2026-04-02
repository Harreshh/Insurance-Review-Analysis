import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Determine model directory relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load basic models
with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'nb_model.pkl'), 'rb') as f:
    nb_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'lr_model.pkl'), 'rb') as f:
    lr_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'svm_model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'cat_model.pkl'), 'rb') as f:
    cat_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'le_category.pkl'), 'rb') as f:
    le_category = pickle.load(f)

# Load neural network models if available
nn_model = None
lstm_model = None
tokenizer = None

# Check if TensorFlow models exist before importing
if os.path.exists('../models/nn_model.h5') or os.path.exists('../models/lstm_model.h5'):
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists('../models/nn_model.h5'):
            nn_model = load_model('../models/nn_model.h5')
        if os.path.exists('../models/lstm_model.h5'):
            lstm_model = load_model('../models/lstm_model.h5')
        if os.path.exists('../models/tokenizer.pkl'):
            with open('../models/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
    except ImportError:
        st.warning("TensorFlow not available. Neural network models will not be used.")
    except Exception as e:
        st.warning(f"Error loading neural network models: {e}")
else:
    st.info("Neural network models not found. Only basic ML models will be used.")

# Custom preprocessing
import sys
sys.path.insert(0, BASE_DIR)
from utils.preprocessing import preprocess_text

st.title("Insurance Review Prediction App")

st.write("Enter an insurance review to predict its sentiment and category, or upload a CSV file for batch processing.")

# File upload option
uploaded_file = st.file_uploader("Upload CSV file with reviews (optional)", type="csv")

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(df_upload.head())

    if st.button("Process All Reviews"):
        if 'review' in df_upload.columns:
            results = []
            for idx, row in df_upload.iterrows():
                review_text = str(row['review'])
                cleaned = preprocess_text(review_text)

                # TF-IDF for ML models
                tfidf_vec = tfidf.transform([cleaned])

                # Sentiment predictions
                nb_pred = nb_model.predict(tfidf_vec)[0]
                lr_pred = lr_model.predict(tfidf_vec)[0]
                svm_pred = svm_model.predict(tfidf_vec)[0]

                # Category prediction
                cat_pred = cat_model.predict(tfidf_vec)[0]
                category = le_category.inverse_transform([cat_pred])[0]

                results.append({
                    'review': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                    'nb_sentiment': 'positive' if nb_pred == 1 else 'negative',
                    'lr_sentiment': 'positive' if lr_pred == 1 else 'negative',
                    'svm_sentiment': 'positive' if svm_pred == 1 else 'negative',
                    'category': category
                })

            results_df = pd.DataFrame(results)
            st.subheader("Batch Prediction Results:")
            st.dataframe(results_df)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )
        else:
            st.error("CSV file must contain a 'review' column.")

st.write("---")
st.write("**Or predict individual reviews:**")

review = st.text_area("Review Text", height=100)

if st.button("Predict"):
    if review:
        # Preprocess
        cleaned = preprocess_text(review)

        # TF-IDF for ML models
        tfidf_vec = tfidf.transform([cleaned])

        # Sentiment predictions
        nb_pred = nb_model.predict(tfidf_vec)[0]
        lr_pred = lr_model.predict(tfidf_vec)[0]
        svm_pred = svm_model.predict(tfidf_vec)[0]

        # Neural network predictions if available
        nn_pred = None
        nn_conf = None
        lstm_pred = None
        lstm_conf = None

        if nn_model and tokenizer:
            try:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=100)
                nn_prob = nn_model.predict(pad)[0][0]
                nn_pred = 1 if nn_prob > 0.5 else 0
                nn_conf = nn_prob
            except Exception as e:
                st.warning(f"Error with neural network prediction: {e}")

        if lstm_model and tokenizer:
            try:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=100)
                lstm_prob = lstm_model.predict(pad)[0][0]
                lstm_pred = 1 if lstm_prob > 0.5 else 0
                lstm_conf = lstm_prob
            except Exception as e:
                st.warning(f"Error with LSTM prediction: {e}")

        # Category prediction
        cat_pred = cat_model.predict(tfidf_vec)[0]
        category = le_category.inverse_transform([cat_pred])[0]

        # Map back
        sentiment_map = {0: 'negative', 1: 'positive'}
        st.write("### Sentiment Predictions:")
        st.write(f"Naive Bayes: {sentiment_map[nb_pred]}")
        st.write(f"Logistic Regression: {sentiment_map[lr_pred]}")
        st.write(f"SVM: {sentiment_map[svm_pred]}")
        if nn_pred is not None:
            st.write(f"Neural Network: {sentiment_map[nn_pred]} (Confidence: {nn_conf:.4f})")
        else:
            st.write("Neural Network: Not available")
        if lstm_pred is not None:
            st.write(f"LSTM: {sentiment_map[lstm_pred]} (Confidence: {lstm_conf:.4f})")
        else:
            st.write("LSTM: Not available")

        st.write(f"### Predicted Category: {category}")
    else:
        st.write("Please enter a review.")