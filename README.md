# Music Genre Classification

An end-to-end project that classifies **10 different music genres** using
classical machine learning algorithms on acoustic features extracted from
audio recordings in the **GTZAN** dataset.

The project covers the full pipeline — from data loading and feature
extraction to modeling, hyperparameter optimization, and visualization —
through a single Jupyter notebook. All reusable logic is organized as
Python modules under the `src/` directory.

---

## Project Goals

- Compare multiple machine learning algorithms on the same dataset and
  identify which one performs best at music genre classification
- Compare each model's **baseline** version with a **GridSearchCV**-optimized version
- Evaluate models from multiple angles using a wide range of performance and reliability metrics
- Present results with clear, informative visualizations

---

## Dataset: GTZAN

- **Source:** [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Content:** 1000 audio clips in total — 100 clips per genre, each 30 seconds long (`.wav`)
- **Classes (10 genres):**
  `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

> **Note:** The raw audio files (`data/raw/genres_original/`) are excluded
> from the repository via `.gitignore` because of their size. To run the
> project locally, download the dataset from Kaggle and place it under
> `data/raw/genres_original/`.

---

## Folder Structure

```
MusicGenreClassification/
├── data/
│   ├── raw/                      # GTZAN raw data (gitignored)
│   │   ├── genres_original/      # Genre folders → .wav files
│   │   ├── images_original/      # Mel-spectrogram images
│   │   ├── features_30_sec.csv   # Pre-computed 30-second features
│   │   └── features_3_sec.csv    # Pre-computed 3-second features
│   └── interim/
│       ├── metadata.csv          # File paths + genre labels
│       └── extracted_features.csv# Features extracted via librosa
├── notebooks/
│   └── 01_main_pipeline.ipynb    # Main pipeline (all experiments)
├── src/
│   ├── config.py                 # Paths, constants, GENRES list
│   ├── data_loader.py            # Scan and label .wav files
│   ├── feature_extraction.py     # Librosa-based feature extraction
│   ├── preprocessing.py          # Label encoding + StandardScaler
│   ├── split_utils.py            # Stratified train / val / test split
│   ├── train.py                  # Model training + timing
│   ├── evaluate.py               # Metric computation
│   └── visualize.py              # All plots
├── .gitignore
└── README.md
```

---

## Tech Stack

- **Python 3**
- **librosa** — audio processing and feature extraction
- **scikit-learn** — models, preprocessing, metrics, GridSearchCV
- **xgboost** — gradient boosting model
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **Jupyter Notebook** (originally developed on Google Colab)

---

## Pipeline Overview

The main notebook (`notebooks/01_main_pipeline.ipynb`) is organized into
10 main sections:

1. **Setup** — Import libraries, define paths
2. **Data Loading** — Scan `.wav` files, build metadata, inspect class distribution
3. **Feature Extraction** — Extract acoustic features from each audio file
4. **Label Encoding** — Convert genre names into numeric labels
5. **Train / Validation / Test Split** — Stratified split (`70% / 10% / 20%`)
6. **Feature Scaling** — Standardization with `StandardScaler`
7. **Model Training** — Train and evaluate 6 different algorithms
8. **Model Comparison** — Compare all baseline models side by side
9. **GridSearchCV** — Hyperparameter optimization for each model
10. **Baseline vs. GridSearch** — Compare optimized models against their baselines

---

## Extracted Features

The `extract_basic_features()` function in `src/feature_extraction.py`
reduces each audio file to a **single fixed-length feature vector**.
For each feature, both the **mean** and the **standard deviation** are
computed (to capture average behavior as well as variability):

| Feature | Description |
|---|---|
| **Zero Crossing Rate** | How often the signal crosses the zero axis |
| **Spectral Centroid** | The "center of mass" of the spectrum (bright vs. dark sound) |
| **Spectral Rolloff** | Frequency below which most of the spectral energy is concentrated |
| **Chroma STFT** | Which of the 12 pitch classes are dominant |
| **RMS Energy** | Overall loudness of the signal |
| **Tempo** | Beats per minute (BPM) |
| **MFCC (1–13)** | Coefficients modeling human auditory perception of frequency |

---

## Models

| # | Model | Library |
|---|---|---|
| 1 | Logistic Regression | scikit-learn |
| 2 | Random Forest | scikit-learn |
| 3 | Support Vector Machine (SVM) | scikit-learn |
| 4 | K-Nearest Neighbors (KNN) | scikit-learn |
| 5 | XGBoost | xgboost |
| 6 | Multi-Layer Perceptron (MLP) | scikit-learn |

Each model is first trained with default parameters (**baseline**), then
tuned via `GridSearchCV` to obtain an optimized version.

---

## Evaluation Metrics

Models are not judged by accuracy alone — they are evaluated with a
**multi-dimensional metric set** (`src/evaluate.py`):

**Classification Metrics**
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-Score (macro & weighted)
- Specificity (macro)

**Reliability / Probability Metrics**
- ROC-AUC (One-vs-Rest, macro)
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)
- Log Loss

**Additional Information**
- Confusion Matrix (per-class error analysis)
- Classification Report (per-genre precision / recall / F1)
- Training time (seconds)

---

## Visualizations

The `src/visualize.py` module produces the following plots:

- Genre distribution bar plot
- 3-panel metric summary per model
- Confusion matrix heatmaps (individual and combined)
- Bar plot and heatmap comparing all models
- Training time and log loss comparisons
- Macro-average ROC curves
- Baseline vs. GridSearch accuracy comparison

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone <repo-url>
cd MusicGenreClassification
```

### 2. Install dependencies
```bash
pip install librosa scikit-learn xgboost pandas numpy matplotlib seaborn jupyter
```

### 3. Download the dataset
Download the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
and place it under `data/raw/genres_original/`. The folder structure
should look like this:

```
data/raw/genres_original/
├── blues/
├── classical/
├── country/
└── ... (10 genres in total)
```

### 4. Update the path in `config.py`
The project was originally written for Google Colab, so `BASE_DIR` in
`src/config.py` points to
`/content/drive/MyDrive/MusicGenreClassification`. When running locally,
update it to match your own path:

```python
BASE_DIR = Path(".").resolve()   # or the absolute path to the project
```

### 5. Open the notebook
```bash
jupyter notebook notebooks/01_main_pipeline.ipynb
```

Run the cells in order to execute the full pipeline from start to finish.

---

## Key Constants

Defined in `src/config.py`:

| Constant | Value | Description |
|---|---|---|
| `RANDOM_STATE` | `42` | Reproducibility for all splits and models |
| `SAMPLE_RATE` | `22050` | Sampling rate used by librosa |
| `TEST_SIZE` | `0.20` | Test set ratio |
| `VAL_SIZE` | `0.10` | Validation set ratio |
