# ============================================================
#  Credit Card Fraud Detection Model
#  Mahmoud Abdelwahab Shaaban — Resume Project #1
# ============================================================
#
#  SETUP (run once in your terminal):
#  pip install pandas scikit-learn matplotlib seaborn imbalanced-learn
#
#  DATASET:
#  Download from Kaggle → search "Credit Card Fraud Detection"
#  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#  Place the file "creditcard.csv" in the same folder as this script.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df["Class"].value_counts())
print(f"\nFraud percentage: {df['Class'].mean()*100:.4f}%")

# ── 2. EXPLORE DATA ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Transaction amount distribution
axes[0].hist(df[df["Class"]==0]["Amount"], bins=50, alpha=0.7, label="Legit", color="steelblue")
axes[0].hist(df[df["Class"]==1]["Amount"], bins=50, alpha=0.7, label="Fraud", color="coral")
axes[0].set_title("Transaction Amount Distribution")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].set_xlim(0, 2000)

# Class imbalance
class_counts = df["Class"].value_counts()
axes[1].bar(["Legitimate", "Fraud"], class_counts.values, color=["steelblue", "coral"])
axes[1].set_title("Class Imbalance")
axes[1].set_ylabel("Number of Transactions")
for i, v in enumerate(class_counts.values):
    axes[1].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("1_data_exploration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 1_data_exploration.png")

# ── 3. PREPROCESS ─────────────────────────────────────────────
# Scale Amount and Time (the only non-PCA features)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

# Drop original unscaled columns
df.drop(["Amount", "Time"], axis=1, inplace=True)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train/test split (stratified to keep fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

# ── 4. HANDLE CLASS IMBALANCE WITH SMOTE ─────────────────────
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Legit: {sum(y_train_res==0):,}  |  Fraud: {sum(y_train_res==1):,}")

# ── 5. TRAIN MODEL ────────────────────────────────────────────
print("\nTraining Random Forest... (takes ~1-2 minutes)")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1,          # use all CPU cores
    class_weight="balanced"
)
model.fit(X_train_res, y_train_res)
print("Training complete!")

# ── 6. EVALUATE ───────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── 7. VISUALIZE RESULTS ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# --- ROC Curve ---
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], color="steelblue")
axes[1].plot([0,1],[0,1],"k--", alpha=0.5)
axes[1].set_title("ROC Curve")

# --- Feature Importance (Top 15) ---
importances = pd.Series(model.feature_importances_, index=X.columns)
top15 = importances.nlargest(15)
top15.sort_values().plot(kind="barh", ax=axes[2], color="steelblue")
axes[2].set_title("Top 15 Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("2_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 2_model_results.png")

# ── 8. FRAUD THRESHOLD TUNING ────────────────────────────────
print("\n" + "="*50)
print("THRESHOLD TUNING (default is 0.5)")
print("="*50)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob >= threshold).astype(int)
    tp = ((y_pred_t==1) & (y_test==1)).sum()
    fp = ((y_pred_t==1) & (y_test==0)).sum()
    fn = ((y_pred_t==0) & (y_test==1)).sum()
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp+fn) > 0 else 0
    print(f"Threshold {threshold:.1f} → Precision: {precision:.3f}  Recall: {recall:.3f}  "
          f"Fraud caught: {tp}/{tp+fn}")

print("\nDone! Check the saved PNG files in your folder.")
print("You can use these charts directly in your portfolio or CV.")