# 🛒 Superstore Sales Analytics & Forecasting Dashboard

A full-stack data analytics and machine learning solution that combines **Streamlit** for interactive data exploration and modeling, and **Power BI** for dynamic business dashboards. This project enables users to upload structured retail data, train predictive models (classification & regression), forecast future sales, and visualize KPIs such as region-wise sales, returns, and shipping delays.

---

## 📌 Features

- 📁 Upload `.csv` or `.xlsx` dataset
- 📊 Perform automatic EDA: summary statistics, correlations, distribution plots
- 🎯 Train classification or regression models (RandomForest, XGBoost, etc.)
- 📈 Forecast next 5 months of sales using Facebook Prophet
- 🔍 Analyze feature importance and model metrics
- 📥 Export cleaned dataset for use in Power BI
- 📊 Power BI dashboard: region-wise sales, returns, shipping delays, and profits

---

## 🧰 Tech Stack

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| Streamlit        | Frontend app for ML workflow     |
| Pandas / NumPy   | Data loading & preprocessing     |
| Scikit-learn     | ML models, train-test split      |
| XGBoost / RandomForest | Advanced ML algorithms       |
| Prophet (by Meta)| Time-series forecasting          |
| Plotly / Seaborn | Interactive and static visuals   |
| Power BI         | Business dashboard visualization |
| Google Colab     | ML model prototyping             |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/superstore-dashboard.git
cd superstore-dashboard
