import os
import sys
import pandas as pd
import numpy as np

# Resolve project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(os.path.join(project_root, "src", "features"))

from extract_features import extract_features_from_audio

frames_index_path = os.path.join(project_root, "data", "processed", "frames_index.csv")
raw_dir = os.path.join(project_root, "data", "raw", "M3")
out_path = os.path.join(project_root, "data", "processed", "features_dataset.csv")

print("=" * 65)
print("[*] STEP 3: EXTRACTING ACOUSTIC & RELATIVE PITCH FEATURES")
print("=" * 65)

if not os.path.exists(frames_index_path):
    print(f"Error: {frames_index_path} not found. Please run build_frame_dataset.py first.")
    sys.exit(1)

frames_df = pd.read_csv(frames_index_path)
unique_recordings = frames_df[["s_file", "s_scale", "subset"]].drop_duplicates()
print(f"Total audio files to process: {len(unique_recordings)}")

feature_tables = []
file_count = 0

for _, row in unique_recordings.iterrows():
    s_file = row["s_file"]
    s_scale = row["s_scale"]
    subset = str(row["subset"]).zfill(3)
    
    audio_path = os.path.join(raw_dir, subset, "student", s_file)
    if not os.path.exists(audio_path):
        continue
        
    feat_df = extract_features_from_audio(
        audio_path=audio_path,
        s_scale=s_scale,
        frame_size_sec=0.1,
        hop_size_sec=0.05
    )
    
    if feat_df.empty:
        continue
        
    feat_df["s_file"] = s_file
    feature_tables.append(feat_df)
    file_count += 1
    
    if file_count % 50 == 0 or file_count == len(unique_recordings):
        print(f"  Processed {file_count}/{len(unique_recordings)} audio files...")

all_feats_df = pd.concat(feature_tables, ignore_index=True)
print(f"\nMerging extracted features with ground-truth labels...")

# Merge on (s_file, frame_idx)
merged_df = pd.merge(
    frames_df,
    all_feats_df,
    on=["s_file", "frame_idx"],
    how="inner"
)

# Reorder columns: identifiers, labels, then features
feature_cols = [
    "rms", "delta_rms", "zcr", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "energy_low", "energy_mid", "energy_high",
    "f0", "delta_f0", "pitch_confidence", "is_voiced",
    "cents", "cents_octave", "cents_dev", "abs_cents_dev", "pitch_stability"
]

meta_cols = ["s_file", "t_file", "s_id", "s_scale", "frame_idx", "t_start", "t_end", "binary_label", "category"]
final_df = merged_df[meta_cols + feature_cols]

print(f"Saving {len(final_df):,} feature rows to: {out_path}")
final_df.to_csv(out_path, index=False)

print("\n" + "=" * 65)
print("[+] FEATURE EXTRACTION COMPLETE!")
print("=" * 65)
print(f"Total Rows (Frames)   : {len(final_df):,}")
print(f"Total Feature Columns : {len(feature_cols)}")
print(f"Features: {', '.join(feature_cols)}\n")

print("--- Feature Correlation with Mistake Label (Top 8) ---")
corr = final_df[feature_cols + ["binary_label"]].corr()["binary_label"].drop("binary_label").abs().sort_values(ascending=False)
for col, val in corr.head(8).items():
    print(f"  * {col:<22} : Pearson |r| = {val:.4f}")
print("=" * 65)

