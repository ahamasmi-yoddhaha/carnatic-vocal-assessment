import os
import glob
import pandas as pd
from segmentation import segment_audio_file

# Resolve project paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
raw_dir = os.path.join(project_root, "data", "raw", "M3")
processed_dir = os.path.join(project_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

print("=" * 65)
print("[*] STEP 1 & STEP 2: AUDIO SEGMENTATION & GROUND-TRUTH ALIGNMENT")
print("=" * 65)
print(f"Data Source : {raw_dir}")
print(f"Destination : {processed_dir}\n")

all_frames = []
total_files_processed = 0

for subset in ["001", "002"]:
    sub_path = os.path.join(raw_dir, subset)
    meta_file = os.path.join(sub_path, f"metadata_{subset}.csv")
    if not os.path.exists(meta_file):
        continue
    
    meta_df = pd.read_csv(meta_file)
    print(f"Processing Subset {subset} ({len(meta_df)} recordings)...")
    
    for _, row in meta_df.iterrows():
        s_file = row["s_file"]
        audio_path = os.path.join(sub_path, "student", s_file)
        label_path = os.path.join(sub_path, "labels", f"{s_file}.txt")
        
        if not os.path.exists(audio_path):
            continue
            
        row_dict = row.to_dict()
        row_dict["subset"] = subset
        
        frames_df = segment_audio_file(
            audio_path=audio_path,
            label_path=label_path,
            metadata_row=row_dict,
            frame_size_sec=0.1,  # 100 ms frame window
            hop_size_sec=0.05    # 50 ms hop (50% overlap)
        )
        
        frames_df["subset"] = subset
        all_frames.append(frames_df)
        total_files_processed += 1

combined_df = pd.concat(all_frames, ignore_index=True)
out_csv = os.path.join(processed_dir, "frames_index.csv")
combined_df.to_csv(out_csv, index=False)

print("\n" + "=" * 65)
print("[+] SEGMENTATION & ALIGNMENT COMPLETE!")
print("=" * 65)
print(f"Total audio files processed : {total_files_processed}")
print(f"Total frames generated      : {len(combined_df):,}")
print(f"Saved frames index to       : {out_csv}\n")

print("--- Frame Distribution (Binary) ---")
bin_counts = combined_df["binary_label"].value_counts()
print(f"  * Correct (0) : {bin_counts.get(0, 0):,} ({bin_counts.get(0, 0)/len(combined_df)*100:.1f}%)")
print(f"  * Mistake (1) : {bin_counts.get(1, 0):,} ({bin_counts.get(1, 0)/len(combined_df)*100:.1f}%)")

print("\n--- Detailed Mistake Categories ---")
cat_counts = combined_df["category"].value_counts()
for cat, count in cat_counts.items():
    print(f"  * {cat:<10} : {count:,} ({count/len(combined_df)*100:.2f}%)")

print("\n--- Unique Singers (s_id) ---")
singers = combined_df["s_id"].unique()
print(f"Singers found ({len(singers)}): {', '.join(map(str, singers))}")
print("=" * 65)
