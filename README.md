# 🎵 Carnatic Vocal Assessment System

An intelligent web-based vocal assessment system for Carnatic music learners. The system allows users to record foundational swara exercises (e.g., *Sarali Varisai* in Raga *Mayamalavagowla*) and receive objective, pedagogical feedback on pitch accuracy (*Sruti Shuddham*), rhythm (*Laya*), and note stability (*Karvai*).

---

## 🏗️ Architecture Overview

The system is built on a hybrid architecture combining deep learning and musical signal processing:
1. **Pitch Extraction**: Deep learning-based $F_0$ contour extraction using CREPE.
2. **Carnatic Swara Mapping**: Relative pitch conversion from absolute frequency (Hz) to cents relative to the singer's tonic (*Adhara Shadjam*).
3. **Temporal Alignment**: Dynamic Time Warping (DTW) to align learner recordings with gold-standard teacher templates.
4. **Mistake Detection & Scoring**: CNN/CRNN comparative models (adapted from singing mistake pedagogy research) evaluating pitch deviation, timing lag, and voice stability.
5. **Interactive Web Dashboard**: Browser-based recording interface with visual pitch overlay and multidimensional scores.

---

## 📁 Project Structure

```text
carnatic-vocal-assessment/
├── data/
│   ├── raw/             # Raw audio recordings & third-party datasets
│   ├── processed/       # Resampled, normalized, and trimmed audio
│   └── references/      # Gold-standard teacher templates & swara boundaries
├── notebooks/           # Exploratory data analysis & model experiments
├── src/
│   ├── preprocessing/   # Audio normalization, silence trimming, tonic detection
│   ├── features/        # CREPE pitch tracking, cents conversion, swara quantizer
│   ├── models/          # CNN / CRNN mistake detection architectures
│   └── scoring/         # DTW alignment, Sruti and Laya scoring engine
├── web/                 # Frontend recording UI and API server
├── tests/               # Unit and integration tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
