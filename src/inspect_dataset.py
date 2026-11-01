import os
import glob

# Ensure paths work whether run from project root or from inside src/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
raw_dir = os.path.join(project_root, "data", "raw", "M3")

print("=" * 60)
print("[*] CARNATIC VOCAL ASSESSMENT - M3 DATASET INSPECTOR")
print("=" * 60)
print(f"Project Root: {project_root}")
print(f"M3 Data Path: {raw_dir}")

if not os.path.exists(raw_dir):
    print(f"Error: Path {raw_dir} does not exist.")
    exit(1)

print("\nSubsets found:", [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])

for subset in ["001", "002"]:
    sub_path = os.path.join(raw_dir, subset)
    if not os.path.exists(sub_path):
        continue
    students = glob.glob(os.path.join(sub_path, "student", "*.wav"))
    teachers = glob.glob(os.path.join(sub_path, "teacher", "*.wav"))
    labels = glob.glob(os.path.join(sub_path, "labels", "*.txt"))
    meta_file = os.path.join(sub_path, f"metadata_{subset}.csv")
    notes_file = os.path.join(sub_path, "note_sequence.csv")
    
    meta_rows = 0
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta_rows = len(f.readlines()) - 1
            
    print(f"\n--- Subset {subset} ---")
    print(f"  * Teacher Reference WAVs : {len(teachers)}")
    print(f"  * Student Attempt WAVs   : {len(students)}")
    print(f"  * Ground-Truth Label TXT : {len(labels)}")
    print(f"  * Metadata Pairs (CSV)   : {meta_rows}")
    print(f"  * Note Sequence Map      : {'Found' if os.path.exists(notes_file) else 'Missing'}")
    
print("\n" + "=" * 60)
print("Dataset inspection complete. Ready for pipeline processing!")
print("=" * 60)
