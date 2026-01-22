
# Sentiment Analysis NLP

This project implements a **Sentiment Analysis** solution using NLP techniques on the **Amazon Review dataset**. It includes both classical machine learning models (Logistic Regression with Count Vectorizer) and a **BERT-based deep learning model** for sentiment classification.

---

## **Project Overview**

With the explosion of online reviews and user-generated content, analyzing opinions automatically has become essential. This project aims to build an end-to-end solution that can:

- Preprocess and clean text data
- Convert text into numerical representations (Count Vectorizer, TF-IDF, BERT embeddings)
- Train and evaluate classification models
- Deploy an interactive interface using Streamlit

---


**Project Structure**
- `app/` 
  - `streamlit_app.py`
- `dataset/` 
  - `amazon_review.csv`
- `models/`  — Trained models and tokenizer files
  - ` count_vectorizer.pkl/`
  - `  log_model_count.pkl/`
  - ` special_tokens_map.json/`
  - ` tokenizer.json/`
  - `  tokenizer_config.json/`
  - `  training_args.bin/`
  - `  vocab.txt/`
- `notebooks/`
 - ` text_classification_sentiment_analysis_with_nlp.ipynb /`
- `config.json `
- `requirements.txt`
- `README.md`


## **Usage**

Run the Streamlit App
streamlit run app/streamlit_app.py


Enter text in the input box to predict its sentiment.

The app uses the fine-tuned BERT model by default.

The predicted sentiment (positive/negative) is displayed along with a confidence score.

Use Models in Scripts

src/preprocessing.py → Text cleaning and tokenization

src/vectorization.py → Convert text to Count Vectorizer, TF-IDF, or BERT embeddings

src/train_model.py → Train Logistic Regression or fine-tune BERT

src/predict.py → Predict sentiment from text

Models

Logistic Regression + Count Vectorizer
Accuracy: ~87%

BERT-based Model (bert-base-uncased, fine-tuned)
Accuracy: ~92%
F1-score: 95%

The BERT model captures semantic context and outperforms classical approaches.
=======
# sentiment-analysis-nlp
NLP project for sentiment analysis on Amazon Review dataset




