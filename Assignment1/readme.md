# Real-Time Market Sentiment Analyzer

## 📌 Overview
This project implements a LangChain-powered pipeline that analyzes real-time market sentiment for a given company. It uses:
- **LangChain** for chaining operations
- **Google Gemini-2.0-flash** via **Vertex AI** for sentiment analysis
- **mlflow** for observability and prompt tracing

## ⚙️ Features
- Accepts a company name as input
- Extracts or generates its stock ticker symbol
- Fetches recent news headlines using LangChain tools
- Analyzes sentiment and named entities using Gemini-2.0-flash
- Outputs structured JSON with sentiment profile
- Logs prompts, outputs, and metadata using mlflow

