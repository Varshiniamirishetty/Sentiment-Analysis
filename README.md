# 🎬 AI-Based Sentiment, Emotion and Sarcasm Detection in Movie Reviews

An intelligent NLP-powered web application that automatically scrapes movie reviews 
from IMDb and performs sentiment analysis, emotion detection, and sarcasm identification 
using state-of-the-art transformer models.

---

## 📌 Project Overview

This project presents an end-to-end system for analyzing movie reviews collected 
directly from IMDb. It goes beyond simple star ratings by detecting the true sentiment, 
underlying emotions, and sarcastic intent behind each review using multiple pre-trained 
deep learning models deployed through an interactive Streamlit web interface.

---

## ✨ Features

- 🔍 **Automatic IMDb Review Scraping** — Fetches up to 300 reviews using a 
  3-strategy pipeline (AJAX, Requests, Selenium)
- 😊 **Sentiment Analysis** — Classifies reviews as Positive or Negative using DistilBERT
- 🎭 **Emotion Detection** — Identifies joy, anger, fear, sadness, disgust, 
  and surprise using DistilRoBERTa
- 🤔 **Sarcasm Detection** — Detects ironic reviews using TwitterRoBERTa 
  combined with rule-based patterns
- ✅ **Sentiment Correction** — Automatically flips sarcastic positive reviews to Negative
- ☁️ **Word Cloud** — Visualizes most frequently used words across all reviews
- 📊 **Interactive Charts** — Sentiment and emotion distribution bar charts
- 📁 **Multiple Input Modes** — IMDb URL, CSV file upload, or manual text entry

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Sentiment Model | distilbert-base-uncased-finetuned-sst-2-english |
| Emotion Model | j-hartmann/emotion-english-distilroberta-base |
| Sarcasm Model | cardiffnlp/twitter-roberta-base-irony |
| Web Scraping | Selenium, BeautifulSoup4, Requests |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, WordCloud |
| NLP Toolkit | HuggingFace Transformers, NLTK |
| Language | Python 3.8+ |

---

## 📁 Project Structure

movie-review-analyzer/
│
├── app.py               # Main application — all modules in one file
└── requirements.txt     # All required Python dependencies

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Step 3 — Run the Application
```bash
streamlit run app.py
```

### Step 6 — Open in Browser
http://localhost:8501

---

## 🚀 How to Use

1. Open the application in your browser
2. Select input type from the sidebar
3. For IMDb URL paste a link like:
https://www.imdb.com/title/tt1375666/reviews
4. Set the number of reviews using the slider
5. Click **Fetch Reviews**
6. Select analysis type from the sidebar
7. View results including metrics, charts and sarcasm flagged reviews
---

## 🧠 Models Used

### 1. Sentiment Analysis
- **Model:** distilbert-base-uncased-finetuned-sst-2-english
- **Output:** POSITIVE / NEGATIVE with confidence score

### 2. Emotion Detection
- **Model:** j-hartmann/emotion-english-distilroberta-base
- **Output:** anger, disgust, fear, joy, sadness, surprise, neutral

### 3. Sarcasm Detection
- **Model:** cardiffnlp/twitter-roberta-base-irony
- **Output:** IRONY / NON-IRONY
- **Hybrid:** Combined with rule-based regex patterns

---

## ⚠️ Notes

- Internet connection is required to fetch IMDb reviews
- Google Chrome must be installed for Selenium scraping
- ChromeDriver is managed automatically via webdriver-manager
- Models are cached after first load for faster subsequent runs
- If 0 reviews are fetched enable Debug Mode in the sidebar

---

## 🔮 Future Enhancements

- Multilingual review support
- Aspect-based sentiment analysis
- Support for Rotten Tomatoes and Metacritic
- Fine-tuning models on IMDb review data
- Exportable PDF report generation
- Deployment on Streamlit Cloud

---


## 📄 License

This project is developed for academic purposes.

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) for pre-trained transformer models
- [IMDb](https://www.imdb.com) for movie review data
- [Streamlit](https://streamlit.io) for the web application framework
- [CardiffNLP](https://github.com/cardiffnlp) for the irony detection model
