import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Insurance Review Summary App")

st.write("Upload processed reviews CSV to generate summaries by insurer/category.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(df)} reviews!")

        # Check required columns
        required_cols = ['review', 'rating', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.write("Required columns: review, rating, category")
        else:
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Summary by category
            st.subheader("Summary by Category")
            category_summary = df.groupby('category').agg({
                'rating': ['mean', 'count'],
                'review': 'count'
            }).round(2)
            category_summary.columns = ['Avg Rating', 'Rating Count', 'Review Count']
            st.dataframe(category_summary)

            # Visualization
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Average rating by category
            sns.barplot(data=df, x='category', y='rating', ax=ax[0])
            ax[0].set_title('Average Rating by Category')
            ax[0].tick_params(axis='x', rotation=45)

            # Review count by category
            df['category'].value_counts().plot(kind='bar', ax=ax[1])
            ax[1].set_title('Review Count by Category')
            ax[1].tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            df['sentiment'] = df['rating'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.write("Please ensure your CSV file is properly formatted.")

else:
    st.write("Please upload a CSV file with columns: review, rating, category")
    st.info("💡 Tip: You can use the processed reviews from data/processed/cleaned_reviews.csv")