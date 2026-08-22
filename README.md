# Behavioral Customer Value Prediction Using RFM Analytics

<p align="center">
  <b>Machine Learning system for predicting future high-value customers from historical purchasing behaviour using RFM-based customer analytics.</b>
</p>

---

## Overview

**Behavioral Customer Value Prediction** is a machine learning project that uses historical retail transaction data to identify customers who are likely to become **high-value customers in a future period**.

The project transforms transaction-level data into customer-level behavioural features, including **Recency, Frequency, Average Order Value, Average Quantity, Total Items, Unique Products, and Number of Transactions**.

A time-based approach is used to separate historical behaviour from future spending, helping prevent **data leakage** during model development. 0

Several classification models were evaluated using both the original training data and **SMOTE-balanced training data**, including Logistic Regression, Decision Tree, Random Forest, and XGBoost. 1

---

## Key Highlights

- **541,909** original transactions
- SQL-based data cleaning and aggregation
- Customer-level behavioural feature engineering
- Time-based train/test design
- RFM-oriented customer behaviour analysis
- Logistic Regression, Decision Tree, Random Forest and XGBoost
- SMOTE for addressing class imbalance
- Hyperparameter tuning with GridSearchCV
- Evaluation using Accuracy, Precision, Recall and F1 Score

---

## Technologies

**Python · Pandas · NumPy · SQL · SQLite · Scikit-learn · XGBoost · Imbalanced-learn · Matplotlib · Seaborn · Google Colab**

---

## Project Structure

- `Customer_Value_Prediction.ipynb` — Complete data analysis, feature engineering and machine learning workflow
- `CUSTOMER VALUE PREDICTION REPORT.pdf` — Detailed project report
- `customer_value_logistic_smote.pkl` — Saved machine learning model

---

## Models Evaluated

| Model | Approach |
|---|---|
| Logistic Regression | Baseline |
| Logistic Regression | SMOTE |
| Logistic Regression | GridSearchCV + SMOTE |
| Decision Tree | Baseline |
| Decision Tree | SMOTE |
| Random Forest | Baseline |
| Random Forest | SMOTE |
| XGBoost | Baseline |
| XGBoost | SMOTE |

The models were compared using F1 Score alongside Precision, Recall and Accuracy to assess their ability to identify future high-value customers. 2

---

## Objective

The goal is to move beyond simply describing past customer behaviour and build a predictive system that can help businesses identify **which customers are more likely to generate high future value**.

This can support targeted retention, customer engagement and data-driven marketing decisions.

---

## Author

**Olalemi Olaoluwakintan**

Data Scientist | Data Analyst | RAN Engineer

[GitHub](https://github.com/Pro-phet123)
