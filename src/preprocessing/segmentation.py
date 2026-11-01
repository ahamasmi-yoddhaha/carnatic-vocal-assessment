import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from typing import List, Dict, Tuple, Optional


def parse_label_file(label_path: str) -> List[Dict]:
    """
    Parses a mistake label text file.
    Format of each line: <start_time> <end_time> <mistake_tag>
    Returns a list of intervals: [{'start': float, 'end': float, 'tag': str}, ...]
    """
    intervals = []
    if not os.path.exists(label_path):
        return intervals

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    start_t = float(parts[0])
                    end_t = float(parts[1])
                    tag = parts[2].upper()
                    intervals.append({
                        "start": start_t,
                        "end": end_t,
                        "tag": tag
                    })
                except ValueError:
                    continue
    return intervals


def get_frame_label(frame_start: float, frame_end: float, intervals: List[Dict], threshold: float = 0.5) -> Tuple[int, str]:
    """
    Determines if a frame [frame_start, frame_end] corresponds to a mistake.
    
    Args:
        frame_start: Frame start in seconds
        frame_end: Frame end in seconds
        intervals: List of ground-truth mistake intervals from parse_label_file
        threshold: Minimum overlap ratio with frame to be considered a mistake (default 0.5)
        
    Returns:
        (binary_label, primary_tag)
        binary_label: 0 for Correct, 1 for Mistake
        primary_tag: "CORRECT" or the mistake code (e.g. "F", "S", "T", "A", etc.)
    """
    frame_dur = frame_end - frame_start
    best_overlap = 0.0
    best_tag = "CORRECT"

    for item in intervals:
        overlap_start = max(frame_start, item["start"])
        overlap_end = min(frame_end, item["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_tag = item["tag"]

    if frame_dur > 0 and (best_overlap / frame_dur) >= threshold:
        return 1, best_tag
    
    return 0, "CORRECT"


def segment_audio_file(
    audio_path: str,
    label_path: str,
    metadata_row: Dict,
    frame_size_sec: float = 0.1,
    hop_size_sec: float = 0.05
) -> pd.DataFrame:
    """
    Slices a single student audio file into overlapping frames and aligns with labels.
    
    Returns a DataFrame containing frame timestamps, sample indices, and labels.
    """
    sr, audio = wavfile.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if stereo
    
    duration = len(audio) / sr
    intervals = parse_label_file(label_path)
    
    frame_len_samples = int(frame_size_sec * sr)
    hop_len_samples = int(hop_size_sec * sr)
    
    records = []
    frame_idx = 0
    start_sample = 0
    
    while start_sample + frame_len_samples <= len(audio):
        end_sample = start_sample + frame_len_samples
        t_start = start_sample / sr
        t_end = end_sample / sr
        
        bin_label, tag = get_frame_label(t_start, t_end, intervals, threshold=0.5)
        
        records.append({
            "s_file": metadata_row.get("s_file"),
            "t_file": metadata_row.get("t_file"),
            "s_id": metadata_row.get("s_id"),
            "s_scale": metadata_row.get("s_scale"),
            "t_scale": metadata_row.get("t_scale"),
            "frame_idx": frame_idx,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "t_start": round(t_start, 4),
            "t_end": round(t_end, 4),
            "sample_rate": sr,
            "binary_label": bin_label,
            "category": tag
        })
        
        start_sample += hop_len_samples
        frame_idx += 1
        
    return pd.DataFrame(records)

