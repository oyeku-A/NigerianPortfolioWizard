# NigerianPortfolioWizard

A data-driven portfolio optimisation tool tailored to the Nigerian stock market using Modern Portfolio Theory (MPT) and powered by EODHD API.

## 💡 About the Project

This project was developed as my final year project, being a Computer Science Major, focusing on applying financial theory and data science to create optimised investment portfolios for Nigerian retail investors. It leverages Modern Portfolio Theory (MPT) to balance risk and return, utilising real-time and historical market data from EODHD.

## 🎯 Features

- Real-time and historical stock data from Nigerian Exchange via EODHD APIs
- Portfolio construction using expected returns, variances, and covariances
- Efficient frontier and risk-return visualisations
- Customizable constraints (e.g. risk tolerance)

## ⚙️ Technologies Used

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- SciPy for optimisation
- EODHD API for data access
- Jupyter Notebook for prototyping
- Streamlit for UI

## 📊 Data Source
https://eodhd.com/exchange/XNSA
[EODHD Financial APIs](https://eodhd.com/exchange/XNSA) – used for pulling End-Of-Day stock data for listed Nigerian companies.

## 🇳🇬 Why Nigeria?

Access to well-diversified, data-driven portfolios is still limited for average Nigerian investors. This tool is designed to bridge that gap, helping users make smarter, more structured investment decisions.

## 🚀 Getting Started

1. Clone this repository
2. Add your EODHD API key in `.env` or directly into the config
3. Run the main script or Jupyter notebook to build and test portfolios
