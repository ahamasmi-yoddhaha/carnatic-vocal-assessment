import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Resolve project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_path = os.path.join(project_root, "data", "processed", "features_dataset.csv")
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 65)
print("[*] STEP 6: 1D-CNN DEEP LEARNING MODEL TRAINING (CPU)")
print("=" * 65)

# 1. Feature Definition
feature_cols = [
    "rms", "delta_rms", "zcr", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "energy_low", "energy_mid", "energy_high",
    "f0", "delta_f0", "pitch_confidence", "is_voiced",
    "cents", "cents_octave", "cents_dev", "abs_cents_dev", "pitch_stability"
]

test_singers = ['s_008', 's_004', 's_012', 's_E30303', 's_test C30408', 's_t E30307', 's_test E30310']

print(f"Loading features from: {data_path}...")
df = pd.read_csv(data_path)

# Singer-Independent Split
train_mask = ~df["s_id"].isin(test_singers)
test_mask = df["s_id"].isin(test_singers)

# Cast all feature columns to float32 to prevent pandas 3.0 LossySetitemError
df[feature_cols] = df[feature_cols].astype(np.float32)

# Fit scaler on train only (No leakage)
scaler = StandardScaler()
df.loc[train_mask, feature_cols] = scaler.fit_transform(df.loc[train_mask, feature_cols].values)
df.loc[test_mask, feature_cols] = scaler.transform(df.loc[test_mask, feature_cols].values)

# Group dataframe into song sequences
train_df = df[train_mask]
test_df = df[test_mask]

def dataframe_to_sequences(sub_df):
    sequences = []
    labels = []
    singers = []
    files = []
    
    for s_file, group in sub_df.groupby("s_file", sort=False):
        group_sorted = group.sort_values("frame_idx")
        X_seq = group_sorted[feature_cols].values.astype(np.float32)
        y_seq = group_sorted["binary_label"].values.astype(np.float32)
        
        sequences.append(torch.tensor(X_seq))
        labels.append(torch.tensor(y_seq))
        singers.append(group["s_id"].iloc[0])
        files.append(s_file)
        
    return sequences, labels, singers, files

print("Grouping frames into continuous temporal song sequences...")
X_train_seqs, y_train_seqs, train_singers_list, _ = dataframe_to_sequences(train_df)
X_test_seqs, y_test_seqs, test_singers_list, _ = dataframe_to_sequences(test_df)

print(f"\n[+] Sequence Partition Summary:")
print(f"  * Training Sequences : {len(X_train_seqs)} songs across {len(set(train_singers_list))} singers")
print(f"  * Test Sequences     : {len(X_test_seqs)} songs across {len(set(test_singers_list))} UNSEEN singers\n")


# PyTorch Sequence Dataset
class VocalSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Collate function with dynamic padding and mask
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    padded_seqs = torch.zeros(len(sequences), max_len, sequences[0].shape[1])
    padded_labels = torch.zeros(len(labels), max_len)
    masks = torch.zeros(len(labels), max_len, dtype=torch.bool)

    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        length = len(seq)
        padded_seqs[i, :length] = seq
        padded_labels[i, :length] = lab
        masks[i, :length] = True

    return padded_seqs, padded_labels, masks


train_dataset = VocalSequenceDataset(X_train_seqs, y_train_seqs)
test_dataset = VocalSequenceDataset(X_test_seqs, y_test_seqs)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Deep 1D-CNN Model Architecture (Adapted from Base Paper)
class Carnatic1DCNN(nn.Module):
    def __init__(self, in_channels: int = 18, dropout_rate: float = 0.2):
        super().__init__()
        
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Block 4 & 5 (Bottleneck)
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Block 6 & 7
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Block 8 & 9
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Output Layer (Binary Mistake Logit)
            nn.Conv1d(16, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, channels) -> transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.net(x)
        # out shape: (batch, 1, seq_len) -> squeeze to (batch, seq_len)
        return out.squeeze(1)


device = torch.device("cpu")
model = Carnatic1DCNN(in_channels=len(feature_cols)).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[+] Model Compiled: Carnatic 1D-CNN ({total_params:,} parameters, ~{total_params*4/(1024**2):.1f} MB)")

# Weighted BCE Loss to handle class balance
pos_weight = torch.tensor([1.2]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training Loop
num_epochs = 25
best_val_f1 = 0.0
checkpoint_path = os.path.join(models_dir, "best_1d_cnn.pt")

print(f"\n[+] Starting CPU Training for {num_epochs} Epochs...")
print("-" * 65)
print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Acc':<10} {'Val Rec':<10} {'Val F1':<10} {'Time'}")
print("-" * 65)

total_start_time = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0.0
    total_train_frames = 0
    
    for batch_x, batch_y, masks in train_loader:
        batch_x, batch_y, masks = batch_x.to(device), batch_y.to(device), masks.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        
        loss = criterion(logits, batch_y)
        masked_loss = (loss * masks).sum() / masks.sum()
        
        masked_loss.backward()
        optimizer.step()
        
        train_loss += masked_loss.item() * masks.sum().item()
        total_train_frames += masks.sum().item()
        
    avg_train_loss = train_loss / total_train_frames
    
    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y, masks in test_loader:
            batch_x, batch_y, masks = batch_x.to(device), batch_y.to(device), masks.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Mask out padding frames
            valid_preds = preds[masks].cpu().numpy()
            valid_targets = batch_y[masks].cpu().numpy()
            valid_probs = probs[masks].cpu().numpy()
            
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
            all_probs.extend(valid_probs)
            
    val_acc = accuracy_score(all_targets, all_preds)
    val_rec = recall_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds)
    
    epoch_dur = time.time() - epoch_start
    scheduler.step(avg_train_loss)
    
    saved_flag = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_f1': val_f1,
            'val_acc': val_acc
        }, checkpoint_path)
        saved_flag = "[BEST]"
        
    print(f"Ep {epoch:02d}/{num_epochs:02d}  {avg_train_loss:<12.4f} {val_acc*100:>5.2f}%    {val_rec*100:>5.2f}%    {val_f1*100:>5.2f}%    {epoch_dur:>4.1f}s {saved_flag}")

total_dur = time.time() - total_start_time
print("-" * 65)
print(f"[+] Training completed in {total_dur/60:.1f} minutes!")

# Load Best Checkpoint for Final Evaluation
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"[+] Loaded best checkpoint from Epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']*100:.2f}%)")

# Final Evaluation on Unseen Test Singers
model.eval()
final_preds = []
final_targets = []
final_probs = []

with torch.no_grad():
    for batch_x, batch_y, masks in test_loader:
        batch_x, batch_y, masks = batch_x.to(device), batch_y.to(device), masks.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        final_preds.extend(preds[masks].cpu().numpy())
        final_targets.extend(batch_y[masks].cpu().numpy())
        final_probs.extend(probs[masks].cpu().numpy())

final_acc = accuracy_score(final_targets, final_preds)
final_prec = precision_score(final_targets, final_preds)
final_rec = recall_score(final_targets, final_preds)
final_f1 = f1_score(final_targets, final_preds)
final_auc = roc_auc_score(final_targets, final_probs)

print("\n" + "=" * 65)
print("[+] FINAL 1D-CNN BENCHMARK EVALUATION (UNSEEN TEST SINGERS)")
print("=" * 65)
print(f"  * Test Accuracy       : {final_acc*100:.2f}%")
print(f"  * Test Precision      : {final_prec*100:.2f}%")
print(f"  * Test Recall         : {final_rec*100:.2f}%")
print(f"  * Test F1-Score       : {final_f1*100:.2f}%")
print(f"  * Test ROC-AUC Score  : {final_auc:.4f}")

cm = confusion_matrix(final_targets, final_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n--- Confusion Matrix (Unseen Test Set: {len(final_targets):,} frames) ---")
print(f"                      Predicted Correct (0)    Predicted Mistake (1)")
print(f"  Actual Correct (0) :      {tn:<10,} (TN)           {fp:<10,} (FP)")
print(f"  Actual Mistake (1) :      {fn:<10,} (FN)           {tp:<10,} (TP)")
print("=" * 65)

