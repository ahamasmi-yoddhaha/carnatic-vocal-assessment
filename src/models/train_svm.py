import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
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

print("=" * 65)
print("[*] STEP 5: SUPPORT VECTOR MACHINE (SVM) TRAINING & EVALUATION")
print("=" * 65)

print(f"Loading features dataset from: {data_path}...")
df = pd.read_csv(data_path)
print(f"Total samples loaded: {len(df):,}")

# Feature columns (18 acoustic & relative Carnatic pitch features)
feature_cols = [
    "rms", "delta_rms", "zcr", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "energy_low", "energy_mid", "energy_high",
    "f0", "delta_f0", "pitch_confidence", "is_voiced",
    "cents", "cents_octave", "cents_dev", "abs_cents_dev", "pitch_stability"
]

target_col = "binary_label"

# Singer-Independent Partition (Unseen Test Singers)
test_singers = ['s_008', 's_004', 's_012', 's_E30303', 's_test C30408', 's_t E30307', 's_test E30310']

train_mask = ~df["s_id"].isin(test_singers)
test_mask = df["s_id"].isin(test_singers)

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, target_col].values

X_test = df.loc[test_mask, feature_cols].values
y_test = df.loc[test_mask, target_col].values

train_singers_count = df.loc[train_mask, "s_id"].nunique()
test_singers_count = df.loc[test_mask, "s_id"].nunique()

print(f"\n[+] Partition Summary:")
print(f"  * Training Set : {len(X_train):,} frames ({len(X_train)/len(df)*100:.1f}%) across {train_singers_count} singers")
print(f"    - Mistake Rate: {np.mean(y_train)*100:.2f}%")
print(f"  * Test Set     : {len(X_test):,} frames ({len(X_test)/len(df)*100:.1f}%) across {test_singers_count} UNSEEN singers")
print(f"    - Mistake Rate: {np.mean(y_test)*100:.2f}%\n")

# Step 2: Feature Standardization (Z-score scaling)
print("[+] Performing Feature Standardization (Z-score scaling)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  * StandardScaler fit on training features only (No Data Leakage).")

# Step 3: Train Support Vector Machine (Linear SVM via Hinge Loss SGD)
print("\n[+] Training Support Vector Machine (Hinge-Loss Maximum Margin)...")
svm = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=1e-4,
    max_iter=1000,
    tol=1e-3,
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train_scaled, y_train)
print("  * SVM optimization converged successfully!")

# Step 4: Model Evaluation
print("\n" + "=" * 65)
print("[+] SVM MODEL EVALUATION ON UNSEEN TEST SINGERS")
print("=" * 65)

# Predictions and Decision Function (Margin distance)
y_train_pred = svm.predict(X_train_scaled)
y_test_pred = svm.predict(X_test_scaled)
y_test_decision = svm.decision_function(X_test_scaled)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_decision)

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

# Hyperplane Feature Weights (Support Vector Separating Plane)
weights_df = pd.DataFrame({
    "Feature": feature_cols,
    "Weight": svm.coef_[0],
    "Abs_Weight": np.abs(svm.coef_[0])
}).sort_values("Abs_Weight", ascending=False).reset_index(drop=True)

print(f"\n--- SVM Separating Hyperplane Weights (Influence Ranking) ---")
print(f"{'Rank':<5} {'Feature':<22} {'Weight':<10} {'Margin Push Direction'}")
print("-" * 65)
for i, r in weights_df.iterrows():
    push = "Pushes Toward Mistake Boundary (+)" if r['Weight'] > 0 else "Pushes Toward Correct Boundary (-)"
    print(f"{i+1:<5} {r['Feature']:<22} {r['Weight']:<+10.4f} {push}")

print("=" * 65)

