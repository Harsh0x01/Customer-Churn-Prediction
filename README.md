# Customer Churn Prediction
### XGBoost · SHAP Explainability · Retention Strategy Engine · Streamlit

A production-grade machine learning system that predicts customer churn for banking, explains the reasoning behind each prediction using SHAP, and automatically generates retention strategies — built as an end-to-end data science project.

---

## What Makes This Different

Most churn projects stop at a prediction. This one goes further:

- **SHAP Explainability** — explains *why* a customer is likely to churn, not just that they will
- **Retention Strategy Engine** — maps SHAP reasons to real business actions automatically
- **Threshold Tuning** — optimized for business use (0.35) to maximize churn recall
- **Full Pipeline** — from raw data to deployed interactive web app

---

## Live Demo

![App Screenshot](assets/app_screenshot.png)

---

## Project Flow

```
Dataset → EDA → Preprocessing → Model Training → Best Model → SHAP → Retention Engine → Streamlit App → Deploy
```

---

## Results

| Model | Accuracy | ROC-AUC | Churn F1 |
|---|---|---|---|
| Logistic Regression | 71.00% | 0.70 | 0.48 |
| Random Forest | 83.90% | 0.74 | 0.59 |
| **XGBoost** | **85.15%** | **0.85** | **0.59** |

- **Cross Validation AUC:** 0.96 ± 0.04
- **Threshold:** 0.35 (tuned for business use — maximizes churn recall)
- **Churn Recall after tuning:** 65%

---

## How It Works

**1. Predict**
XGBoost model predicts churn probability for a customer based on 10 features.

**2. Explain**
SHAP (SHapley Additive exPlanations) identifies the top features driving that prediction — Age, IsActiveMember, Balance, Tenure etc.

**3. Act**
The Retention Engine reads the SHAP reasons and maps them to specific business actions:

| SHAP Reason | Retention Action |
|---|---|
| High Age | Assign dedicated senior relationship manager |
| Inactive Member | Launch re-engagement campaign with cashback rewards |
| High Balance | Upgrade to premium account with better interest rates |
| Low Tenure | Activate early loyalty program |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Imbalance | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Dashboard | Power BI |

---

## Dataset

**Churn Modelling** — Bank customer churn dataset from Kaggle

- 10,000 rows · 14 columns
- Target: `Exited` (1 = Churned, 0 = Stayed)
- Class imbalance: 20% churn / 80% stayed → handled with SMOTE

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                  # Streamlit web app
├── churn data.csv          # Dataset
├── requirements.txt        # Dependencies
│
├── model/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   ├── scaler.pkl          # StandardScaler
│   └── feature_names.pkl   # Feature names
│
└── notebook/
    └── churn_notebook.ipynb  # Full analysis notebook
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/Harsh0x01/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Key Insights from EDA

- Customers aged **40-60** churn the most
- **Germany** has the highest churn rate proportionally
- **Female** customers churn more than males
- Customers with **high balance** are more likely to churn despite having more money
- **Inactive members** are significantly more likely to leave

---

## Author

**Harsh** — B.Tech Computer Science (Data Science), Maharana Pratap Engineering College

[![GitHub](https://img.shields.io/badge/GitHub-Harsh0x01-181717?style=flat&logo=github)](https://github.com/Harsh0x01)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)

---

## License

MIT License — free to use and modify
