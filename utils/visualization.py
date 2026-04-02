import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

def plot_rating_distribution(df, rating_col='rating'):
    """
    Plot distribution of star ratings
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=rating_col)
    plt.title('Distribution of Star Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

def plot_text_length_distribution(df, text_col='review'):
    """
    Plot distribution of text lengths
    """
    df['text_length'] = df[text_col].apply(len)
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='text_length', bins=30)
    plt.title('Distribution of Review Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

def generate_wordcloud(text_data, title='Word Cloud'):
    """
    Generate word cloud from text data
    """
    text = ' '.join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def plot_top_ngrams(text_data, n=2, top_k=20):
    """
    Plot top n-grams
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(text_data)
    ngram_counts = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()
    top_indices = ngram_counts.argsort()[-top_k:][::-1]
    top_ngrams = [ngrams[i] for i in top_indices]
    top_counts = [ngram_counts[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_counts, y=top_ngrams)
    plt.title(f'Top {top_k} {n}-grams')
    plt.xlabel('Frequency')
    plt.ylabel(f'{n}-gram')
    plt.show()