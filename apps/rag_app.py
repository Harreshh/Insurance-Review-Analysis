import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load basic models
with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load data
import pandas as pd

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-score {
        color: #dc3545;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .rating-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .rating-positive { background-color: #d4edda; color: #155724; }
    .rating-neutral { background-color: #fff3cd; color: #856404; }
    .rating-negative { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 Insurance Review Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask intelligent questions about insurance customer reviews</div>', unsafe_allow_html=True)

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Reviews", "913")
with col2:
    st.metric("Categories", "5")
with col3:
    st.metric("Avg Rating", "3.2")

# File upload option
st.subheader("📁 Data Source")
uploaded_file = st.file_uploader("Upload your own CSV file (optional)", type="csv", help="CSV must contain a 'review' column")

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        if 'review' in df_upload.columns:
            df = df_upload.copy()
            # Use cleaned_review if available, otherwise use review
            text_column = 'cleaned_review' if 'cleaned_review' in df_upload.columns else 'review'
            # Recompute TF-IDF for uploaded data
            with st.spinner("🔄 Processing your uploaded data..."):
                st.session_state.tfidf_matrix = tfidf.transform(df[text_column].tolist())
            st.success(f"✅ Loaded {len(df)} reviews from your file!")
        else:
            st.error("❌ CSV file must contain a 'review' column.")
            df = pd.read_csv('../data/processed/cleaned_reviews.csv')  # Fallback to default
            text_column = 'cleaned_review'
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {e}")
        df = pd.read_csv('../data/processed/cleaned_reviews.csv')  # Fallback to default
        text_column = 'cleaned_review'
else:
    # Load default data
    df = pd.read_csv('../data/processed/cleaned_reviews.csv')
    text_column = 'cleaned_review'
    if 'tfidf_matrix' not in st.session_state:
        with st.spinner("🔄 Loading default insurance reviews..."):
            st.session_state.tfidf_matrix = tfidf.transform(df[text_column].tolist())

st.write(f"📊 Currently analyzing {len(df)} insurance reviews")

# Question input
st.subheader("❓ Ask Your Question")
st.markdown('<div class="question-box">', unsafe_allow_html=True)
query = st.text_input("Enter your question about insurance reviews:", placeholder="e.g., What do customers say about customer service?")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 Search Reviews", type="primary", use_container_width=True):
    if query:
        # Transform query to TF-IDF
        query_tfidf = tfidf.transform([query])
        
        # Find similar reviews and ensure diversity
        similarities = cosine_similarity(query_tfidf, st.session_state.tfidf_matrix)[0]

        # Get top indices but ensure diverse results (no exact duplicate text)
        all_indices = np.argsort(similarities)[::-1]  # Sort by similarity descending
        top_indices = []
        seen_texts = set()

        for idx in all_indices:
            display_text = df.iloc[idx][text_column] if text_column in df.columns else df.iloc[idx]['review']
            if display_text not in seen_texts and len(top_indices) < 5:
                top_indices.append(idx)
                seen_texts.add(display_text)

        # If we don't have 5 diverse results, fill with remaining high-similarity ones
        if len(top_indices) < 5:
            for idx in all_indices:
                if idx not in top_indices and len(top_indices) < 5:
                    top_indices.append(idx)

        st.subheader(f"🎯 Top {len(top_indices)} Most Relevant Reviews")

        for i, idx in enumerate(top_indices, 1):
            # Use the appropriate text column for display
            display_text = df.iloc[idx][text_column] if text_column in df.columns else df.iloc[idx]['review']
            rating = df.iloc[idx]['rating']
            category = df.iloc[idx]['category']
            score = similarities[idx]

            # Rating color
            rating_class = "rating-positive" if rating >= 4 else "rating-neutral" if rating >= 3 else "rating-negative"

            st.markdown(f'''
            <div class="result-card">
                <h4>#{i} Review</h4>
                <p><strong>💬</strong> {display_text}</p>
                <p><strong>🏷️ Category:</strong> {category}</p>
                <p><strong>⭐ Rating:</strong> <span class="rating-badge {rating_class}">{rating}/5 stars</span></p>
                <p><strong>🎯 Relevance:</strong> <span class="similarity-score">{score:.4f}</span></p>
            </div>
            ''', unsafe_allow_html=True)

        # Summary
        avg_rating = df.iloc[top_indices]['rating'].mean()
        categories_found = df.iloc[top_indices]['category'].unique()

        st.subheader("📈 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
        with col2:
            st.metric("Categories Found", len(categories_found))
        with col3:
            st.metric("Top Similarity", f"{similarities[top_indices[0]]:.4f}")

    else:
        st.warning("⚠️ Please enter a question to search.")

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Try questions like 'customer service', 'claims process', 'premium costs', or 'coverage options'")