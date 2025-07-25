# IntelliWealth — AI-Powered Portfolio Optimizer

IntelliWealth is a full-stack AI-enabled platform designed to revolutionize portfolio optimization for both individual investors and financial advisors. By combining machine learning, sentiment analysis, and predictive analytics, it empowers users to make personalized, data-driven investment decisions based on real-time market conditions.

---

## 🧠 Problem Statement

Traditional investment tools struggle with dynamic markets, manual tracking, and lack personalized guidance. IntelliWealth bridges this gap with:

- Real-time AI-driven portfolio recommendations  
- Goal-based financial strategies  
- Predictive analytics using time-series and sentiment models  
- Visual, interactive dashboards and stress testing  

---

## 🚀 Key Features

- **AI-Driven Risk Profiling**: Uses XGBoost and historical data to assess user risk appetite  
- **Dynamic Asset Allocation**: Real-time optimization based on user goals and market fluctuations  
- **Predictive Market Analytics**: Time-series forecasting and sentiment interpretation using FinBERT, LDA  
- **Interactive Visualizations**: Built with D3.js and Chart.js for portfolio analysis and stress tests  
- **Centralized Dashboard**: Live updates, personalized insights, market news summaries  
- **Stock Report Generator**: Auto-generates detailed stock reports and insights  

---

## 🏗️ Project Architecture

### ➤ Tech Stack

| Layer         | Technology                                     |
|---------------|------------------------------------------------|
| **Frontend**  | React.js, Chart.js, D3.js                      |
| **Backend**   | FastAPI, Python (Scikit-learn, TensorFlow, NLTK, Hugging Face) |
| **Database**  | PostgreSQL                                     |
| **Deployment**| AWS, GitHub Actions (CI/CD), OAuth 2.0         |
| **APIs**      | Yahoo Finance, Alpha Vantage                   |

---

## 📊 Core Methodology

- **LSTM Time-Series Forecasting** for price prediction  
- **Sentiment Analysis** using FinBERT and VADER on financial tweets/news  
- **Portfolio Optimization**:  
  - Sharpe Ratio Maximization using `scipy.optimize`  
  - Volatility minimized using covariance matrices and log returns  
- **Feature Engineering**:  
  - Z-score-based volatility, daily returns  
  - PCA for dimensionality reduction  

---

## 🌍 Real-World Use Cases

- **Retail Investors**: Personalized portfolio suggestions  
- **Financial Advisors**: Automates risk assessment and client recommendations  
- **Analysts & Researchers**: In-depth forecasting and sentiment-based insights  
- **FinTech Startups**: Plug-and-play microservices for portfolio management  

---

## 🧩 System Modules

## 📁 Project Directory Structure
 ```
📦 CSC-206/
├── backend/ # FastAPI server with ML models
│ ├── models/ # Risk profiling, LSTM, FinBERT, etc.
│ ├── routes/ # API endpoints
│ └── utils/ # Preprocessing, optimization functions
├── frontend/ # React.js application
│ ├── components/ # Charts, forms, UI elements
│ └── pages/ # Dashboard, login, reports
├── data/ # Historical financial data & feature files
├── requirements.txt # Backend dependencies
├── package.json # Frontend dependencies
└── README.md # Project overview



---

## 🔐 Security & Auth

- OAuth 2.0 Google Authentication  
- Secure API endpoints and user data handling  

---

## 🧪 Results & Visuals

- **Optimized Portfolio Visualization**: Allocations, risk scores, equity projections  
- **Comparative Analysis**: vs. S&P 500  
- **Live Market Sentiment Charts**  
- **Auto-Generated Reports and News Summary**  

(Screenshots are included in the project report PDF)

---

## 🔮 Future Enhancements

- Predicting volatility independently with LightGBM  
- Support for short-term investment optimization using higher-frequency data  
- User-personalized sentiment analytics  
- Inclusion of macroeconomic indicators (e.g., RSI, Bollinger Bands, VIX)  

---

## 📚 References

- [Yahoo Finance](https://finance.yahoo.com)  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [TensorFlow](https://www.tensorflow.org/)  
- [LSTM - Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)  

---

## 👥 Team - Group 22

| Name               | Enrollment No. | Role                       |
|--------------------|----------------|-----------------------------|
| Shiv Shakti Kumar  | 23114091       | ML & Backend Developer     |
| Nisarg Prajapati   | 23114073       | Frontend Developer         |
| Vanjale Pranjal    | 22115158       | Data Scientist             |
| Samrat Middha      | 21117109       | DevOps & Security          |
| Shreyash Sinha     | 21111037       | Financial Analyst & UI     |
| Kaustubh Dwivedi   | 22117066       | Sentiment Analyst & Tester |

---

## 📥 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Munnond/CSC-206.git
cd CSC-206

# Backend Setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend Setup
cd ../frontend
npm install
npm start
