# PROJECT_PROGRESS.md

## Completed
- Phase 1 — Project setup and environment ✓ (folders, requirements.txt, config.py, run scripts)
- Phase 2 — Dataset preparation ✓ (`dataset/generate_sample_data.py` → `social_media_data.csv`, synthetic)
- Phase 3 — Data preprocessing ✓ (`preprocessing/text_preprocessing.py`, `utils/data_loader.py`)
- Phase 4 — Sentiment-analysis model ✓ (TF-IDF + Logistic Regression)
- Phase 5 — Topic/keyword analysis ✓ (`analysis/topic_analysis.py`)
- Phase 6 — Brand-performance score ✓ (`analysis/brand_analysis.py`)
- Phase 7 — Historical trend analysis ✓ (day/week/month + trend detection)
- Phase 8 — Prediction model ✓ (Random Forest, time-based split)
- Phase 9 — Streamlit dashboard ✓ (`app.py`, 8 pages)
- Phase 10 — Statement analyzer ✓ (page inside `app.py`)
- Phase 11 — Testing and error handling ✓ (`tests/test_pipeline.py`, 16 tests passing)
- Phase 12 — README, documentation, report, PPT, viva ✓ (`README.md`, `docs/`)
- Optional MySQL support ✓ (`database/schema.sql`, `database/db_utils.py`) — written but needs a local MySQL
  server to test; not tested by the developer of this zip.

## Current
- Project is complete and runnable. Only user-side tasks remain: run it, take screenshots, write the report.

## Files created
app.py, config.py, train_all.py, requirements.txt, requirements-mysql.txt, run_app.bat, run_app.sh,
dataset/{generate_sample_data.py, social_media_data.csv, example_upload_no_labels.csv},
model/{sentiment_model.pkl, tfidf_vectorizer.pkl, performance_model.pkl, sentiment_metrics.json, performance_metrics.json},
preprocessing/text_preprocessing.py, utils/data_loader.py,
analysis/{sentiment_analysis.py, topic_analysis.py, brand_analysis.py, pipeline.py},
training/{train_sentiment_model.py, train_performance_model.py},
database/{schema.sql, db_utils.py}, tests/test_pipeline.py,
docs/{REPORT_GUIDE.md, PPT_OUTLINE.md, VIVA_QA.md}, README.md

## Next task (suggestions)
- Replace/augment the synthetic dataset with a real labelled dataset and retrain (`python train_all.py`).
- Add screenshots to the report/PPT; fill in the real numbers from the Model Evaluation page.
- Optional: test MySQL integration; improve topic detection.

## Important decisions
- **Streamlit + CSV first**; MySQL optional and imported lazily so the app never depends on it.
- **No NLTK**: built-in stop-word list (negations `not/no/never/nor` are deliberately kept).
- **One `clean_text()`** used for training and prediction.
- **Synthetic data** with 7% label noise and fictional brands; documented honestly. Sentiment accuracy ≈ 92.6%,
  performance-model accuracy ≈ 46.7% vs 31.7% baseline (values from `model/*_metrics.json`).
- **Performance label** = mean(next 2 weeks' score) − mean(last 3 weeks' score), ±2 points threshold. Weekly granularity.
- **Time-based split** for the performance model (no leakage from the future).
- Prediction needs ≥ 6 weekly periods (`MIN_PERIODS_FOR_PREDICTION` in config.py).
- Saved models are scikit-learn-version-sensitive → app offers a "Train models now" button if loading fails.
- The score/prediction are social-media perception indicators only, never financial performance.
