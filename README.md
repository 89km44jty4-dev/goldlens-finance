# GoldLens Finance

A Streamlit Voice of Customer analytics and financial news app.

## Features
- Upload Excel/CSV customer feedback
- Gemini-based Product, Topic, Sentiment and Urgency labels
- TF-IDF and KMeans topic clustering
- Dashboard and early-warning review
- AI management summary
- Vietstock RSS news page

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Environment variables
GEMINI_API_KEY: your Gemini API key
GEMINI_MODEL: gemini-3.8-flash
