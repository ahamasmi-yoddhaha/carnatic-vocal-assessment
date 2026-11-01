import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
print("[*] STEP 5: RANDOM FOREST ENSEMBLE TRAINING & EVALUATION")
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

# Step 2: Train Random Forest Classifier
print("[+] Training Random Forest Classifier (100 Trees, Max Depth 14)...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
print("  * Random Forest ensemble training complete!")

# Step 3: Model Evaluation
print("\n" + "=" * 65)
print("[+] RANDOM FOREST EVALUATION ON UNSEEN TEST SINGERS")
print("=" * 65)

# Predictions & Probabilities
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
y_test_proba = rf.predict_proba(X_test)[:, 1]

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

# Gini Feature Importances (Tree-based Information Gain)
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances,
    "Percentage": importances * 100
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(f"\n--- Random Forest Gini Feature Importances ---")
print(f"{'Rank':<5} {'Feature':<22} {'Importance':<12} {'Contribution'}")
print("-" * 65)
for i, r in feat_imp_df.iterrows():
    bar = "#" * int(r['Percentage'] * 1.5)
    print(f"{i+1:<5} {r['Feature']:<22} {r['Percentage']:>5.2f}%       {bar}")

print("=" * 65)

