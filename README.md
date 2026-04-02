# 🚀 NLP Project for Insurance Review Analysis

This is a comprehensive NLP project that involves building both supervised and unsupervised learning models for insurance review analysis, and deploying them in interactive web applications.

## 🎯 Project Overview

**Overall Aim:** To develop NLP-powered applications that can analyze insurance company reviews, predict ratings and categories, and provide interactive insights through Streamlit applications.

**Key Features:**
- 📊 **Data Analysis**: Clean and analyze 913+ insurance reviews
- 🤖 **ML Models**: Multiple supervised models (Naive Bayes, SVM, Logistic Regression, Neural Networks)
- 🔍 **Unsupervised Learning**: Topic modeling and word embeddings
- 🌐 **Web Apps**: Three interactive Streamlit applications
- 📈 **Visualizations**: Comprehensive charts and insights

## 📁 Project Structure

```
nlp_project2/
├── 📄 README.md                    # Project documentation
├── 📋 requirements.txt             # Python dependencies
├── 📊 sample_reviews.csv           # Sample dataset for testing
│
├── 📚 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Data cleaning & analysis
│   ├── 02_unsupervised_learning.ipynb  # Topic modeling
│   └── 03_supervised_learning.ipynb    # ML model training
│
├── 🤖 models/                      # Trained models
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   ├── nb_model.pkl               # Naive Bayes model
│   ├── lr_model.pkl               # Logistic Regression
│   ├── svm_model.pkl              # SVM model
│   ├── cat_model.pkl              # Category classifier
│   └── le_category.pkl            # Label encoder
│
├── 🌐 apps/                        # Streamlit applications
│   ├── prediction_app.py          # Sentiment analysis & prediction
│   ├── summary_app.py             # Review summaries & charts
│   └── rag_app.py                 # Q&A search system
│
├── 📊 data/                        # Datasets
│   ├── raw/                       # Original data
│   │   └── insurance_reviews.csv
│   └── processed/                 # Cleaned data
│       └── cleaned_reviews.csv
│
└── 🛠️ utils/                       # Utility functions
    ├── preprocessing.py           # Text cleaning functions
    └── visualization.py           # Plotting functions
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Setup
```bash
# 1. Clone or navigate to project directory
cd /Users/manthrikishankumar/nlp_project2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Download spaCy model
python -m spacy download en_core_web_sm
```

## 🚀 Running the Project

### Option 1: Quick Start (All Apps)
```bash
# Start all three apps simultaneously
cd /Users/manthrikishankumar/nlp_project2/apps
streamlit run prediction_app.py --server.port 8502 --server.headless true &
streamlit run summary_app.py --server.port 8501 --server.headless true &
streamlit run rag_app.py --server.port 8503 --server.headless true &
```

### Option 2: Individual Apps

#### Prediction App (Sentiment Analysis)
```bash
cd /Users/manthrikishankumar/nlp_project2/apps
streamlit run prediction_app.py --server.port 8502 --server.headless true
```
- **URL:** http://localhost:8502
- **Features:** Single review analysis, batch processing, model comparison

#### Summary App (Review Summaries)
```bash
cd /Users/manthrikishankumar/nlp_project2/apps
streamlit run summary_app.py --server.port 8501 --server.headless true
```
- **URL:** http://localhost:8501
- **Features:** Category summaries, visualizations, statistics

#### RAG App (Q&A System)
```bash
cd /Users/manthrikishankumar/nlp_project2/apps
streamlit run rag_app.py --server.port 8503 --server.headless true
```
- **URL:** http://localhost:8503
- **Features:** Natural language search, diverse results, file upload

### Option 3: Jupyter Notebooks
```bash
cd /Users/manthrikishankumar/nlp_project2

# Data Exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Unsupervised Learning
jupyter notebook notebooks/02_unsupervised_learning.ipynb

# Supervised Learning
jupyter notebook notebooks/03_supervised_learning.ipynb
```

## 📱 Application Features

### 1. 🔮 Prediction App (Port 8502)
**Purpose:** Analyze individual reviews or process CSV files in batch

**Features:**
- Single review sentiment analysis
- Rating prediction (1-5 stars)
- Category classification
- Model confidence scores
- Batch processing for CSV files
- Model comparison (Naive Bayes, SVM, Logistic Regression)

**Input:** Insurance review text or CSV file with 'review' column
**Output:** Predicted rating, category, and confidence scores

### 2. 📊 Summary App (Port 8501)
**Purpose:** Generate statistical summaries and visualizations

**Features:**
- Category-wise review summaries
- Average ratings by category
- Review count distributions
- Interactive charts and graphs
- CSV upload for custom datasets

**Input:** CSV file with columns: review, rating, category
**Output:** Statistical summaries and visualizations

### 3. 🔍 RAG App (Port 8503)
**Purpose:** Natural language search through insurance reviews

**Features:**
- Semantic search using TF-IDF
- Diverse result filtering (no duplicates)
- Similarity scoring
- File upload for custom datasets
- Professional UI with cards and badges

**Input:** Natural language questions
**Output:** Top 5 most relevant reviews with similarity scores

## 📊 Dataset Information

- **Total Reviews:** 913 unique reviews
- **Categories:** Customer Service, Claims, Pricing, Policy Terms
- **Rating Range:** 1-5 stars
- **Source:** Insurance company reviews

## 🤖 Machine Learning Models

### Supervised Models
- **Naive Bayes:** Text classification baseline
- **SVM:** Support Vector Machine for text classification
- **Logistic Regression:** Linear model for prediction
- **Neural Networks:** Deep learning models (when available)

### Unsupervised Models
- **TF-IDF Vectorization:** Text feature extraction
- **Word2Vec:** Word embeddings
- **Topic Modeling:** Latent Dirichlet Allocation (LDA)

## 🔧 Management Commands

```bash
# Stop all running Streamlit apps
pkill -f streamlit

# Check running processes
ps aux | grep streamlit

# Verify app status
curl -s http://localhost:8501 && echo "✅ Summary App OK"
curl -s http://localhost:8502 && echo "✅ Prediction App OK"
curl -s http://localhost:8503 && echo "✅ RAG App OK"
```

## 📈 Project Phases

1. **Data Exploration & Cleaning**
   - Text preprocessing and normalization
   - Duplicate removal
   - Exploratory data analysis

2. **Unsupervised Learning**
   - Topic modeling with LDA
   - Word embeddings with Word2Vec
   - Clustering analysis

3. **Supervised Learning**
   - Multi-class classification for ratings
   - Multi-label classification for categories
   - Model evaluation and comparison

4. **Application Development**
   - Streamlit web interfaces
   - Interactive visualizations
   - File upload capabilities

## 📚 Libraries & Dependencies

**Core Libraries:**
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning
- `streamlit` - Web applications

**NLP Libraries:**
- `nltk` - Natural language processing
- `spacy` - Advanced NLP
- `textblob` - Sentiment analysis
- `sentence-transformers` - Text embeddings

**ML & Deep Learning:**
- `tensorflow` - Neural networks
- `transformers` - Pre-trained models
- `gensim` - Topic modeling

## 🎯 Usage Examples

### Single Review Analysis
1. Open Prediction App (http://localhost:8502)
2. Enter: "Great customer service, very helpful staff"
3. Get: Rating prediction, category, confidence scores

### Batch Processing
1. Open Prediction App (http://localhost:8502)
2. Upload `sample_reviews.csv`
3. Click "Process All Reviews"
4. Download results with predictions

### Q&A Search
1. Open RAG App (http://localhost:8503)
2. Ask: "What do customers say about claims process?"
3. Get: Top 5 relevant reviews with similarity scores

## 🚨 Troubleshooting

**Common Issues:**
- **Port conflicts:** Change port numbers in commands
- **Missing dependencies:** Run `pip install -r requirements.txt`
- **NLTK data:** Run the download commands in setup
- **Model loading errors:** Ensure model files exist in `models/` directory

**App Won't Start:**
```bash
# Kill existing processes
pkill -f streamlit

# Clear cache
rm -rf ~/.streamlit/

# Restart with different port
streamlit run app.py --server.port 8504
```

## 📝 File Upload Format

**For Prediction App:**
```csv
review
"Excellent customer service"
"Poor claims processing"
```

**For Summary & RAG Apps:**
```csv
review,rating,category
"Excellent customer service",5,Customer Service
"Poor claims processing",2,Claims
```

## 🎉 Success Metrics

- ✅ **913 unique reviews** processed and cleaned
- ✅ **Multiple ML models** trained and evaluated
- ✅ **Three functional web apps** deployed
- ✅ **Interactive visualizations** implemented
- ✅ **File upload capabilities** working
- ✅ **Professional UI/UX** with modern design

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes.

---

**Ready to explore your insurance review data! 🚀**

**Quick Access:**
- 🔮 [Prediction App](http://localhost:8502)
- 📊 [Summary App](http://localhost:8501)
- 🔍 [RAG App](http://localhost:8503)