# 💳 Credit Card Fraud Detection Model

A machine learning model that detects fraudulent credit card transactions using Random Forest and SMOTE oversampling.

## 📊 Results
- **ROC-AUC Score: 0.98**
- **86% of fraud cases detected** (Recall)
- Trained on 284,807 real transactions (0.17% fraud rate)

## 🛠️ Technologies
- Python, scikit-learn, Pandas, NumPy
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)

## ⚙️ How It Works
1. Loads and explores the dataset (284k transactions)
2. Handles severe class imbalance using SMOTE oversampling
3. Trains a Random Forest classifier
4. Evaluates with confusion matrix, ROC curve & feature importance
5. Tunes decision threshold to maximize fraud recall

## 🚀 How to Run
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the project folder
3. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn imbalanced-learn`
4. Run: `python "Fraud detection.py"`

## 📈 Output Charts
- Transaction amount distribution & class imbalance
- Confusion matrix, ROC curve & top 15 feature importances

---
Built by **Mahmoud Abdelwahab Shaaban** — Communications & Information Engineering, Zewail City
