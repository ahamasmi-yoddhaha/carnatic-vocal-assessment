import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Resolve project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_path = os.path.join(project_root, "data", "processed", "features_dataset.csv")
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 65)
print("[*] STEP 4 & STEP 5: SINGER-INDEPENDENT SPLIT & LOGISTIC REGRESSION")
print("=" * 65)

print(f"Loading features dataset from: {data_path}...")
df = pd.read_csv(data_path)
print(f"Total samples loaded: {len(df):,}")

# Feature columns
feature_cols = [
    "rms", "delta_rms", "zcr", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "energy_low", "energy_mid", "energy_high",
    "f0", "delta_f0", "pitch_confidence", "is_voiced",
    "cents", "cents_octave", "cents_dev", "abs_cents_dev", "pitch_stability"
]

target_col = "binary_label"

# Singer-Independent Partition
test_singers = ['s_008', 's_004', 's_012', 's_E30303', 's_test C30408', 's_t E30307', 's_test E30310']

train_mask = ~df["s_id"].isin(test_singers)
test_mask = df["s_id"].isin(test_singers)

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, target_col].values

X_test = df.loc[test_mask, feature_cols].values
y_test = df.loc[test_mask, target_col].values

train_singers = df.loc[train_mask, "s_id"].nunique()
test_singers_count = df.loc[test_mask, "s_id"].nunique()

print(f"\n[+] Partition Summary:")
print(f"  * Training Set : {len(X_train):,} frames ({len(X_train)/len(df)*100:.1f}%) across {train_singers} singers")
print(f"    - Mistake Rate: {np.mean(y_train)*100:.2f}%")
print(f"  * Test Set     : {len(X_test):,} frames ({len(X_test)/len(df)*100:.1f}%) across {test_singers_count} UNSEEN singers")
print(f"    - Mistake Rate: {np.mean(y_test)*100:.2f}%\n")

# Step 2: Feature Scaling (StandardScaler fit ONLY on train)
print("[+] Performing Feature Standardization (Z-score scaling)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  * StandardScaler fit on training features only (No Data Leakage).")

# Step 3: Train Logistic Regression Model
print("\n[+] Training Logistic Regression Classifier...")
clf = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train_scaled, y_train)
print("  * Model convergence achieved successfully!")

# Step 4: Model Evaluation
print("\n" + "=" * 65)
print("[+] MODEL EVALUATION ON UNSEEN TEST SINGERS")
print("=" * 65)

# Predictions
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)
y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n--- Benchmark Performance Metrics ---")
print(f"  * Training Accuracy   : {train_acc*100:.2f}%")
print(f"  * Test Accuracy       : {test_acc*100:.2f}%")
print(f"  * Test Precision      : {test_prec*100:.2f}%")
print(f"  * Test Recall         : {test_rec*100:.2f}%")
print(f"  * Test F1-Score       : {test_f1*100:.2f}%")
print(f"  * Test ROC-AUC Score  : {test_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n--- Confusion Matrix (Unseen Test Set: {len(y_test):,} frames) ---")
print(f"                      Predicted Correct (0)    Predicted Mistake (1)")
print(f"  Actual Correct (0) :      {tn:<10,} (TN)           {fp:<10,} (FP)")
print(f"  Actual Mistake (1) :      {fn:<10,} (FN)           {tp:<10,} (TP)")

# Classification Report
print(f"\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_test_pred, target_names=["Correct (0)", "Mistake (1)"]))

# Feature Coefficients Interpretation
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": clf.coef_[0],
    "Abs_Weight": np.abs(clf.coef_[0])
}).sort_values("Abs_Weight", ascending=False).reset_index(drop=True)

print(f"\n--- Logistic Regression Feature Coefficients (Impact Ranking) ---")
print(f"{'Rank':<5} {'Feature':<22} {'Weight':<10} {'Direction / Impact'}")
print("-" * 65)
for i, r in coef_df.iterrows():
    direction = "Increases Mistake Probability (+)" if r['Coefficient'] > 0 else "Increases Correct Probability (-)"
    print(f"{i+1:<5} {r['Feature']:<22} {r['Coefficient']:<+10.4f} {direction}")

print("=" * 65)
