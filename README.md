<div align="center">

# 🎯 Customer Value Prediction

### Predicting Tomorrow's High-Value Customers from Today's Purchasing Behaviour

An end-to-end machine learning pipeline that transforms raw retail transactions into actionable customer intelligence — built with a strict, leakage-free, time-based design.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Data%20Cleaning-07405e?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-blueviolet)](https://xgboost.ai/)
[![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)](https://imbalanced-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

</div>

---

## 📌 Overview

This project builds a **customer-targeting decision-support system** that predicts which customers are likely to become **high-value** based purely on their historical purchasing behaviour — using **541,909 raw retail transactions** from the Online Retail dataset.

The core engineering challenge wasn't building a model. It was building one **without cheating**. An early iteration returned **100% accuracy** — a red flag, not a win. Root cause: **data leakage**, where information from the evaluation window bled into the features. The entire pipeline was redesigned around a strict **time-based split** to eliminate it.

> **TL;DR:** Raw transactions → SQL cleaning → time-based feature/target split → 7 behavioural features → SMOTE → 4 algorithms benchmarked → GridSearchCV tuning → business-ready recommendation.

---

## 🧠 Problem Statement

Given a customer's **historical** purchasing behaviour, predict whether they will become a **high-value customer** in a **future** time window — defined as spending above the median future spend (**£439.61**).

| | |
|---|---|
| **Type** | Binary Classification |
| **Target** | `1` = High-Value Customer, `0` = Low-Value Customer |
| **Class Balance** | 71.65% (Class 0) / 28.35% (Class 1) — moderate imbalance |
| **Evaluation Focus** | Recall & F1 (missed high-value customers = lost revenue opportunity) |

---

## 🗂️ Dataset

**Source:** Online Retail Transactions Dataset

| Stage | Detail |
|---|---|
| Raw transactions | 541,909 |
| Raw features | 8 |
| Duplicate records removed | 5,268 |
| Missing `CustomerID` removed | 135,080 |
| Missing `Description` removed | 1,454 |
| **Final modelling dataset** | **3,616 customers × 7 engineered features** |

**Time split:**
- 🕰️ **Historical window** — before `2011-10-01` → used to engineer customer features
- 🔮 **Future window** — from `2011-10-01` onward → used only to compute future spend & define the target

---

## ⚙️ Pipeline Architecture

```
Raw Transactions (SQLite)
        │
        ▼
   SQL Data Cleaning
   • Remove null CustomerID
   • Remove cancelled invoices
   • Remove Quantity/UnitPrice ≤ 0
   • Remove duplicates
        │
        ▼
  Time-Based Partitioning
   ┌─────────────┴─────────────┐
   ▼                           ▼
Historical Data          Future Data
(< Oct 1, 2011)          (≥ Oct 1, 2011)
   │                           │
   ▼                           ▼
Feature Engineering      Target Construction
(7 behavioural features)  (median-split future spend)
   └─────────────┬─────────────┘
                 ▼
        Customer-Level Dataset
          (3,616 customers)
                 │
                 ▼
        Train/Test Split (80/20, stratified)
                 │
                 ▼
        ┌────────┴────────┐
        ▼                 ▼
  Baseline Models     SMOTE-Balanced Models
        │                 │
        └────────┬────────┘
                 ▼
        GridSearchCV Tuning
                 │
                 ▼
     Model Comparison & Selection
```

---

## 🧩 Engineered Features

| Feature | Description |
|---|---|
| **Recency** | Days since the customer's most recent historical purchase |
| **Frequency** | Number of distinct invoices |
| **Average Order Value** | Mean monetary value per purchase |
| **Average Quantity** | Mean quantity purchased per order |
| **Total Items** | Total quantity of products purchased |
| **Unique Products** | Count of distinct products purchased |
| **Number of Transactions** | Total transaction records per customer |

**Key EDA insight:** Frequency, Unique Products, Number of Transactions, and Total Items all correlate **positively** with future high value; Recency correlates **negatively**. High-value customers buy more often, buy more variety, and buy more recently.

---

## 🚦 The Leakage Story

> Every good ML case study has a "wait, that's too good" moment. This is ours.

1. **First pass:** RFM features engineered without a time boundary → **100% accuracy**.
2. **Investigation:** the model had implicit access to future information — it wasn't predicting, it was peeking.
3. **Fix:** rebuilt the pipeline so features and target are separated by a hard temporal cutoff, with **zero overlap**.
4. **Result:** an honest baseline — Accuracy dropped to a believable **78.73%**, and the *real* problem (low Recall) became visible for the first time.

This is the difference between a model that looks good in a notebook and one that would survive contact with production data.

---

## 📊 Results

### Baseline → Optimized (Logistic Regression)

| Stage | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | 78.73% | 72.97% | 39.51% | 51.27% |
| + SMOTE | 76.52% | 57.92% | **62.44%** | **60.09%** |
| + GridSearchCV | 74.72% | 54.58% | **63.90%** | 58.88% |

**Recall nearly doubled (39.51% → 63.90%)** — the model went from missing 6 in 10 future high-value customers to catching nearly 2 in 3.

### Full Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 🏆 **Logistic Regression + SMOTE** | 76.52% | 57.92% | 62.44% | **60.09%** |
| Logistic Regression + GridSearchCV | 74.72% | 54.58% | **63.90%** | 58.88% |
| Random Forest + SMOTE | 76.10% | 57.69% | 58.54% | 58.11% |
| Random Forest | 78.73% | 67.59% | 47.80% | 56.00% |
| XGBoost | 77.35% | 62.42% | 50.24% | 55.68% |
| XGBoost + SMOTE | 75.55% | 57.22% | 54.15% | 55.64% |
| Logistic Regression (baseline) | **78.73%** | **72.97%** | 39.51% | 51.27% |
| Decision Tree + SMOTE | 68.09% | 44.35% | 49.76% | 46.90% |
| Decision Tree | 69.06% | 45.45% | 46.34% | 45.89% |

> ⚠️ **Note:** the highest-Accuracy models (baseline Logistic Regression, Random Forest) have the *worst* Recall. In a customer-targeting problem, that's the trap — Accuracy rewards predicting the majority class, Recall measures whether you actually catch the customers who matter.

---

## 💡 Key Takeaways

- **Accuracy alone is not a sufficient metric** for imbalanced business problems — a model can score 78% while missing 6 in 10 of the customers you actually care about.
- **Data leakage is often invisible until you go looking for it.** A too-good-to-be-true result is a debugging signal, not a milestone.
- **More optimization ≠ better generalization.** The GridSearchCV model scored highest on cross-validation F1 but slightly underperformed the simpler SMOTE model on the held-out test set — optimizing on training data doesn't guarantee real-world improvement.
- **The right model depends on the business objective**, not a single leaderboard metric. Balanced Precision/Recall → Logistic Regression + SMOTE. Maximum coverage of potential high-value customers → GridSearchCV variant.

---

## 🏢 Business Applications

| Use Case | How the Model Helps |
|---|---|
| 🎯 **Targeted Marketing** | Prioritize spend on customers with high predicted future value |
| 🔁 **Customer Retention** | Flag at-risk high-value customers for loyalty campaigns |
| 🛒 **Cross-Selling** | Identify customers with broad purchasing patterns |
| 🎁 **Personalized Promotions** | Segment customers by predicted value tier |
| 💰 **Resource Allocation** | Focus limited marketing budget on high-potential segments |

> The model is designed as a **decision-support tool** — a prioritization layer for human marketing decisions, not an autonomous targeting system.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Cleaning | SQLite, SQL |
| Data Manipulation | pandas, NumPy |
| Class Balancing | imbalanced-learn (SMOTE) |
| Modelling | scikit-learn, XGBoost |
| Hyperparameter Tuning | GridSearchCV |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter / Google Colab |

---

## 📁 Project Structure

```
customer-value-prediction/
├── data/
│   └── online_retail.csv
├── notebooks/
│   └── Customer_Value_Prediction.ipynb
├── reports/
│   └── Customer_Value_Prediction_Report.docx
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/customer-value-prediction.git
cd customer-value-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/Customer_Value_Prediction.ipynb
```

**Core dependencies:**
```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
```

---

## 🔮 Future Improvements

- [ ] Benchmark LightGBM and CatBoost
- [ ] Optimize classification threshold instead of using the default 0.5
- [ ] Add ROC-AUC and Precision-Recall AUC evaluation
- [ ] Engineer spending-trend and customer lifetime value (CLV) features
- [ ] Apply SHAP for model explainability
- [ ] Reduce influence of extreme monetary outliers via log-transformation
- [ ] Validate on a second time period / retail dataset

---

## 👤 Author

**Olalemi Olaoluwakintan Emmanuel**
Data Scientist & AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/olaoluwakintan-olalemi-a99182192)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-black?logo=vercel&logoColor=white)](https://github.com/Pro-phet123)

*Built and mentored as part of a hands-on data science mentorship project.*

---

<div align="center">

⭐ **If this project helped you understand time-based leakage prevention or imbalanced classification, consider giving it a star.**

</div>
