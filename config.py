"""
config.py - central place for all paths and settings.
Every other file imports from here, so a setting is changed in ONE place only.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Paths ----------
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "social_media_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
PERFORMANCE_MODEL_PATH = os.path.join(MODEL_DIR, "performance_model.pkl")
SENTIMENT_METRICS_PATH = os.path.join(MODEL_DIR, "sentiment_metrics.json")
PERFORMANCE_METRICS_PATH = os.path.join(MODEL_DIR, "performance_metrics.json")

# ---------- Data ----------
REQUIRED_COLUMNS = ["statement", "brand", "date"]
OPTIONAL_COLUMNS = ["sentiment", "engagement"]
SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]
PERFORMANCE_LABELS = ["Declining", "Stable", "Improving"]

# ---------- Brand score weights (see README: Brand Performance Score) ----------
WEIGHT_POSITIVE = 1.0
WEIGHT_NEUTRAL = 0.5
WEIGHT_NEGATIVE = 0.0

# ---------- Prediction settings ----------
# If the score changes by less than this many points, it counts as "Stable".
STABLE_THRESHOLD = 2.0
# Label rule (used only for TRAINING): compare the average score of the NEXT
# LABEL_HORIZON_WEEKS weeks with the average of the last BASELINE_WEEKS weeks.
LABEL_HORIZON_WEEKS = 2
BASELINE_WEEKS = 3
# Minimum number of weekly periods an uploaded dataset needs for a prediction.
MIN_PERIODS_FOR_PREDICTION = 6

RANDOM_STATE = 42
