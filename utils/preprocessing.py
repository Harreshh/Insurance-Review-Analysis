import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Basic text cleaning: lowercase, remove punctuation, URLs, special chars
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

def remove_stopwords(text):
    """
    Remove stopwords from text
    """
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """
    Lemmatize text
    """
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

def correct_spelling(text):
    """
    Correct spelling using TextBlob
    """
    blob = TextBlob(text)
    return str(blob.correct())

def preprocess_text(text, correct_spell=False):
    """
    Full preprocessing pipeline
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    if correct_spell:
        text = correct_spelling(text)
    return text