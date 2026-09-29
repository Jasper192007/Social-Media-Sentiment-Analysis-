"""Social media sentiment analytics dashboard."""
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.brand_analysis import (
    brand_impact,
    detect_trend,
    overall_score,
    period_summary,
    sentiment_percentages,
)
from analysis.pipeline import load_performance_model, predict_brand_performance, process_dataframe
from analysis.sentiment_analysis import ModelNotFoundError, load_sentiment_model, predict_single
from analysis.topic_analysis import detect_topics, topic_frequencies, topic_sentiment_table, top_keywords
from config import DATASET_PATH, PERFORMANCE_METRICS_PATH, SENTIMENT_METRICS_PATH
from utils.data_loader import DataValidationError, load_and_validate

st.set_page_config(page_title="Social Media Sentiment Analysis", page_icon=":material/analytics:", layout="wide")

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#16835D", "Neutral": "#7B8794", "Negative": "#C44747"}
PRED_COLORS = {"Improving": "#16835D", "Stable": "#B7791F", "Declining": "#C44747"}
PRED_BADGES = {"Improving": "green", "Stable": "orange", "Declining": "red"}
DISCLAIMER = (
    "Social-media-based brand perception only. This score does not represent sales, revenue, "
    "stock price or financial performance."
)
FEATURE_LABELS = {
    "positive_pct": "Positive sentiment (%)",
    "neutral_pct": "Neutral sentiment (%)",
    "negative_pct": "Negative sentiment (%)",
    "mentions": "Mentions",
    "avg_engagement": "Average engagement",
    "prev_score": "Previous score",
    "score_change": "Score change",
    "rolling_score_3": "3-week rolling score",
    "score_vs_rolling": "Score vs. 3-week average",
    "trend_slope_3": "3-week trend slope",
    "rolling_score_6": "6-week rolling score",
    "trend_slope_6": "6-week trend slope",
    "service_complaint_rate": "Service complaint share (%)",
}


@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_model():
    return load_sentiment_model()


@st.cache_resource(show_spinner=False)
def get_performance_model():
    return load_performance_model()


@st.cache_data(show_spinner="Reading and validating data...")
def load_raw(file_bytes):
    source = DATASET_PATH if file_bytes is None else io.BytesIO(file_bytes)
    return load_and_validate(source)


@st.cache_data(show_spinner="Analyzing social media data...")
def get_processed(file_bytes, use_file_labels):
    dataframe, warnings = load_raw(file_bytes)
    model, vectorizer = get_sentiment_model()
    return process_dataframe(dataframe, model, vectorizer, use_file_labels), warnings


def train_models_button():
    st.warning("A required model file is missing or could not be loaded.")
    if st.button("Train models", type="primary", icon=":material/model_training:"):
        with st.spinner("Training models from the built-in dataset..."):
            from training.train_performance_model import train_performance
            from training.train_sentiment_model import train_sentiment

            train_sentiment(verbose=False)
            train_performance(verbose=False)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Training complete.")
        st.rerun()
    st.stop()


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (OSError, json.JSONDecodeError):
        return None


def style_figure(figure, height=340):
    figure.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "DM Sans, sans-serif", "color": "#23324A", "size": 12},
        title={"font": {"size": 16, "color": "#23324A"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 12, "r": 12, "t": 42, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E9EDF3", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#E9EDF3", zeroline=False)
    return figure


def show_chart(figure, key, height=340):
    style_figure(figure, height)
    st.plotly_chart(figure, width="stretch", key=key, config={"displayModeBar": False})


def sentiment_distribution_figure(dataframe):
    counts = dataframe["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    figure = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.62,
        color=counts.index,
        color_discrete_map=COLORS,
    )
    figure.update_traces(textinfo="percent", textposition="inside", sort=False)
    figure.update_layout(showlegend=True, legend_title_text="Sentiment")
    return figure


def sentiment_trend_figure(dataframe, frequency="Week"):
    summary = period_summary(dataframe, frequency)
    trend_data = summary.melt(
        id_vars="period",
        value_vars=["positive_pct", "neutral_pct", "negative_pct"],
        var_name="sentiment",
        value_name="Share of mentions (%)",
    )
    trend_data["sentiment"] = trend_data["sentiment"].str.replace("_pct", "", regex=False).str.capitalize()
    figure = px.line(
        trend_data,
        x="period",
        y="Share of mentions (%)",
        color="sentiment",
        color_discrete_map=COLORS,
        markers=len(summary) < 40,
    )
    figure.update_yaxes(range=[0, 100], title="Mentions (%)")
    figure.update_xaxes(title="Date")
    return figure, summary


def performance_trend_figure(summary, show_trend_line=False):
    figure = px.line(
        summary,
        x="period",
        y="score",
        markers=len(summary) < 60,
        color_discrete_sequence=["#3557C8"],
    )
    figure.update_yaxes(range=[0, 100], title="Score (0–100)")
    figure.update_xaxes(title="Date")
    if show_trend_line and len(summary) >= 3:
        slope, intercept = np.polyfit(np.arange(len(summary)), summary["score"], 1)
        figure.add_scatter(
            x=summary["period"],
            y=slope * np.arange(len(summary)) + intercept,
            mode="lines",
            name="Linear trend",
            line={"dash": "dash", "color": "#B7791F"},
        )
    return figure


def confusion_figure(matrix, labels):
    return px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Mentions"},
        aspect="auto",
    )


def get_prediction():
    try:
        model = get_performance_model()
        history = data[data["brand"] == selected_brand]
        return predict_brand_performance(history, model)
    except (ModelNotFoundError, ValueError) as error:
        return None, str(error)


def show_prediction_card(prediction):
    if isinstance(prediction, tuple):
        st.caption(prediction[1])
        return
    label = prediction["prediction"]
    with st.container(border=True):
        st.caption("BRAND PERCEPTION TREND")
        st.badge(label.upper(), color=PRED_BADGES.get(label, "blue"))
        st.markdown("Based on recent sentiment, engagement and historical brand-performance trends.")
        st.caption(f"Latest complete week ending {prediction['as_of_week'].date()}.")


def render_dashboard():
    st.title("Social Media Sentiment Analysis", icon=":material/analytics:")
    st.subheader("For Brand Performance Prediction")
    st.write(
        "Analyze social media content using NLP and Machine Learning to understand sentiment, discover "
        "discussion topics, monitor brand perception and predict social-media-based performance trends."
    )
    st.caption(f"Current view: {selected_brand} · {len(filtered_data):,} mentions · selected date range")

    percentages = sentiment_percentages(filtered_data)
    with st.container(horizontal=True):
        st.metric("Total mentions", f"{len(filtered_data):,}", border=True)
        st.metric("Positive sentiment", f"{percentages['positive']:.1f}%", border=True)
        st.metric("Negative sentiment", f"{percentages['negative']:.1f}%", border=True)
        st.metric("Brand score", f"{overall_score(filtered_data):.1f} / 100", border=True)

    st.header("Sentiment overview", divider="gray")
    distribution_column, sentiment_column = st.columns(2)
    with distribution_column, st.container(border=True):
        st.subheader("Sentiment distribution")
        show_chart(sentiment_distribution_figure(filtered_data), "dashboard-sentiment-distribution", 310)
    with sentiment_column, st.container(border=True):
        st.subheader("Sentiment trend")
        sentiment_figure, _ = sentiment_trend_figure(filtered_data, "Week")
        show_chart(sentiment_figure, "dashboard-sentiment-trend", 310)

    st.header("Brand performance", divider="gray")
    trend_column, prediction_column = st.columns([1.65, 1])
    with trend_column, st.container(border=True):
        st.subheader("Social-media-based performance trend")
        weekly_summary = period_summary(filtered_data, "Week")
        if len(weekly_summary) >= 2:
            show_chart(performance_trend_figure(weekly_summary), "dashboard-performance-trend", 340)
        else:
            st.caption("Select a wider date range to view the performance trend.")
    with prediction_column:
        show_prediction_card(get_prediction())

    topic_counts = topic_frequencies(filtered_data).head(6).sort_values("count")
    topic_column, score_column = st.columns(2)
    with topic_column, st.container(border=True):
        st.subheader("Top discussion topics")
        topic_figure = px.bar(
            topic_counts,
            x="count",
            y="topic",
            orientation="h",
            color_discrete_sequence=["#3557C8"],
            labels={"count": "Mentions", "topic": ""},
        )
        topic_figure.update_layout(showlegend=False)
        show_chart(topic_figure, "dashboard-topics", 320)
    with score_column, st.container(border=True):
        st.subheader("Score formula")
        st.markdown("**Positive % × 1 + Neutral % × 0.5 + Negative % × 0**")
        st.metric("Current brand score", f"{overall_score(filtered_data):.1f} / 100")
        st.caption("Calculated from the selected brand and date range.")

    st.header("How the system works", divider="gray")
    workflow_steps = [
        ("database", "Collect data", "Load social media posts or a CSV."),
        ("cleaning_services", "Preprocess text", "Normalize text before analysis."),
        ("sentiment_satisfied", "Analyze sentiment", "Classify positive, neutral or negative."),
        ("topic", "Identify topics", "Match posts to discussion themes."),
        ("speed", "Calculate score", "Apply the transparent score formula."),
        ("show_chart", "Track trends", "Compare scores over time."),
        ("query_stats", "Predict trend", "Estimate improving, stable or declining."),
    ]
    for start in range(0, len(workflow_steps), 4):
        workflow_columns = st.columns(min(4, len(workflow_steps) - start))
        for column, (icon, title, detail) in zip(workflow_columns, workflow_steps[start:start + 4]):
            with column, st.container(border=True):
                st.markdown(f":material/{icon}:")
                st.markdown(f"**{title}**")
                st.caption(detail)

    st.header("Key features", divider="gray")
    features = [
        ("table_chart", "Dataset analysis", "Inspect and filter uploaded mentions."),
        ("sentiment_satisfied", "Sentiment analysis", "Review sentiment counts and trends."),
        ("search", "Topic analysis", "Explore discussion themes and keywords."),
        ("monitoring", "Brand performance", "Track a transparent perception score."),
        ("query_stats", "Prediction", "Review predicted perception momentum."),
        ("chat", "Text analyzer", "Analyze one post with the trained model."),
        ("science", "Model evaluation", "Inspect held-out metrics and confusion matrices."),
    ]
    for start in range(0, len(features), 4):
        feature_columns = st.columns(min(4, len(features) - start))
        for column, (icon, title, detail) in zip(feature_columns, features[start:start + 4]):
            with column, st.container(border=True):
                st.markdown(f":material/{icon}:")
                st.markdown(f"**{title}**")
                st.caption(detail)

    st.header("Technologies used", divider="gray")
    technologies = [
        "Python", "Pandas", "NumPy", "Scikit-learn", "TF-IDF", "Logistic Regression",
        "Random Forest", "Plotly", "Streamlit",
    ]
    st.markdown(" ".join(f":blue-badge[{technology}]" for technology in technologies))
    st.caption("Text cleaning uses the project's built-in offline stop-word list; NLTK is not a runtime dependency.")
    st.caption(DISCLAIMER)


def render_dataset():
    st.title("Dataset Analysis", icon=":material/table_chart:")
    for warning in load_warnings:
        if "duplicate row" in warning.lower():
            duplicate_count = warning.split()[1]
            st.info(f"{duplicate_count} duplicate row(s) removed.", icon=":material/info:")
        else:
            st.caption(warning)

    counts = filtered_data["sentiment"].value_counts()
    with st.container(horizontal=True):
        st.metric("Total mentions", f"{len(filtered_data):,}", border=True)
        st.metric("Positive", f"{counts.get('Positive', 0):,}", border=True)
        st.metric("Negative", f"{counts.get('Negative', 0):,}", border=True)
        st.metric("Neutral", f"{counts.get('Neutral', 0):,}", border=True)
    st.metric("Selected brand", selected_brand, border=True)

    st.subheader("Dataset overview")
    overview_columns = st.columns(3)
    overview_columns[0].metric(
        "Date range", f"{filtered_data['date'].min().date()} – {filtered_data['date'].max().date()}"
    )
    overview_columns[1].metric("Brands", f"{data['brand'].nunique():,}")
    overview_columns[2].metric("Rows after validation", f"{len(data):,}")

    st.subheader("Data preview")
    preview = filtered_data[[
        "date", "brand", "statement", "clean_statement", "sentiment", "topic", "engagement"
    ]]
    st.dataframe(
        preview.head(200),
        width="stretch",
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "statement": st.column_config.TextColumn("Social media text", width="large"),
            "clean_statement": st.column_config.TextColumn("Processed text", width="large"),
            "engagement": st.column_config.NumberColumn("Engagement", format="%d"),
        },
    )
    st.caption("Text preprocessing lowercases content and removes links, mentions, punctuation and stop words.")

    with st.expander("Save results to MySQL (optional)", icon=":material/database:"):
        st.caption("Requires MySQL, `requirements-mysql.txt` and `database/schema.sql`. The app works without MySQL.")
        if st.button("Save filtered mentions", icon=":material/save:"):
            try:
                from database.db_utils import save_statements

                saved_count = save_statements(filtered_data)
                st.success(f"Saved {saved_count:,} mentions.")
            except Exception as error:
                st.error(str(error))


def render_sentiment():
    st.title("Sentiment Analysis", icon=":material/sentiment_satisfied:")
    percentages = sentiment_percentages(filtered_data)
    metric_columns = st.columns(3, border=True)
    metric_columns[0].metric("Positive", f"{percentages['positive']:.1f}%")
    metric_columns[1].metric("Neutral", f"{percentages['neutral']:.1f}%")
    metric_columns[2].metric("Negative", f"{percentages['negative']:.1f}%")

    distribution_column, counts_column = st.columns(2)
    with distribution_column, st.container(border=True):
        st.subheader("Sentiment distribution")
        show_chart(sentiment_distribution_figure(filtered_data), "sentiment-distribution", 330)
    with counts_column, st.container(border=True):
        st.subheader("Sentiment counts")
        counts = filtered_data["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        count_data = counts.rename_axis("Sentiment").reset_index(name="Mentions")
        figure = px.bar(count_data, x="Sentiment", y="Mentions", color="Sentiment", color_discrete_map=COLORS)
        figure.update_layout(showlegend=False)
        show_chart(figure, "sentiment-counts", 330)

    st.subheader("Sentiment statistics")
    statistics = filtered_data.groupby("sentiment").agg(
        mentions=("statement", "count"),
        average_engagement=("engagement", "mean"),
        average_confidence=("confidence", "mean"),
    ).reindex(SENTIMENT_ORDER).round(2)
    statistics["share (%)"] = (100 * statistics["mentions"] / statistics["mentions"].sum()).round(2)
    st.dataframe(statistics, width="stretch")
    st.caption("Confidence is the model's average probability for its prediction; CSV labels do not have model confidence.")

    st.subheader("Example mentions")
    selected_sentiment = st.segmented_control("Sentiment", SENTIMENT_ORDER, default="Positive")
    examples = filtered_data[filtered_data["sentiment"] == selected_sentiment]
    st.dataframe(examples[["date", "statement", "engagement"]].head(8), width="stretch", hide_index=True)


def render_topics():
    st.title("Topic Analysis", icon=":material/search:")
    topic_counts = topic_frequencies(filtered_data)
    sentiment_by_topic = topic_sentiment_table(filtered_data).reset_index().rename(columns={"topics": "topic"})
    topic_column, sentiment_column = st.columns(2)
    with topic_column, st.container(border=True):
        st.subheader("Most discussed topics")
        ordered_topics = topic_counts.sort_values("count")
        figure = px.bar(
            ordered_topics,
            x="count",
            y="topic",
            orientation="h",
            color_discrete_sequence=["#3557C8"],
            labels={"count": "Mentions", "topic": ""},
        )
        figure.update_layout(showlegend=False)
        show_chart(figure, "topic-frequency-chart", 370)
    with sentiment_column, st.container(border=True):
        st.subheader("Sentiment by topic")
        figure = px.bar(
            sentiment_by_topic,
            x="topic",
            y=SENTIMENT_ORDER,
            color_discrete_map=COLORS,
            barmode="stack",
            labels={"topic": "Topic", "value": "Mentions", "variable": "Sentiment"},
        )
        show_chart(figure, "topic-sentiment-chart", 370)

    st.subheader("Topic frequency")
    st.dataframe(
        topic_counts.rename(columns={"topic": "Topic", "count": "Mentions"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Topics use explainable keyword matching. General means no listed topic keyword was found; "
        "a mention may have multiple topics."
    )

    keyword_columns = st.columns(2)
    for column, sentiment in zip(keyword_columns, ["Positive", "Negative"]):
        with column, st.container(border=True):
            st.subheader(f"Common words in {sentiment.lower()} mentions")
            keywords = top_keywords(filtered_data, 10, sentiment).sort_values("count")
            figure = px.bar(
                keywords,
                x="count",
                y="keyword",
                orientation="h",
                color_discrete_sequence=[COLORS[sentiment]],
                labels={"count": "Mentions", "keyword": ""},
            )
            figure.update_layout(showlegend=False)
            show_chart(figure, f"keywords-{sentiment.lower()}", 300)


def render_brand_performance():
    st.title("Brand Performance", icon=":material/monitoring:")
    score = overall_score(filtered_data)
    percentage_values = sentiment_percentages(filtered_data)
    st.metric("Social-media-based brand performance score", f"{score:.1f} / 100", border=True)
    st.caption(
        f"Positive {percentage_values['positive']:.1f}% × 1 + Neutral {percentage_values['neutral']:.1f}% × 0.5 "
        f"+ Negative {percentage_values['negative']:.1f}% × 0 = {score:.1f}"
    )

    frequency = st.segmented_control(
        "Trend interval", ["Day", "Week", "Month"], default="Week", key="trend-interval"
    )
    summary = period_summary(filtered_data, frequency or "Week")
    if len(summary) < 2:
        st.warning(f"Only {len(summary)} interval is available. Choose a wider date range or a shorter interval.")
        return

    trend = detect_trend(summary["score"])
    trend_columns = st.columns([1, 3])
    trend_columns[0].metric("Perception trend", trend["trend"])
    trend_columns[1].caption(
        f"Estimated change: {trend['total_change']:+.1f} points over the selected {frequency.lower()} intervals."
    )
    with st.container(border=True):
        st.subheader("Brand performance trend")
        show_chart(performance_trend_figure(summary, show_trend_line=True), "brand-performance-trend", 420)

    sentiment_figure, _ = sentiment_trend_figure(filtered_data, frequency)
    with st.container(border=True):
        st.subheader("Sentiment trend")
        show_chart(sentiment_figure, "brand-sentiment-trend", 360)

    with st.expander("Period-level data"):
        st.dataframe(
            summary.round({column: 2 for column in summary.columns if column != "period"}),
            width="stretch",
            hide_index=True,
        )
    st.caption(DISCLAIMER)


def render_prediction():
    st.title("Brand Performance Prediction", icon=":material/query_stats:")
    prediction = get_prediction()
    if isinstance(prediction, tuple):
        message = prediction[1]
        if "model" in message.lower():
            st.error(message)
            train_models_button()
        st.warning(message)
        return

    label = prediction["prediction"]
    with st.container(border=True):
        st.caption("PREDICTED SOCIAL-MEDIA BRAND PERCEPTION TREND")
        st.badge(label.upper(), color=PRED_BADGES.get(label, "blue"))
        st.subheader(f"{selected_brand} · {prediction['as_of_week'].date()}")
        st.write("Based on recent sentiment, engagement and historical brand-performance trends.")

    st.subheader("Model probability")
    probabilities = pd.DataFrame(prediction["probabilities"].items(), columns=["Outcome", "Probability"])
    probability_figure = px.bar(
        probabilities,
        x="Outcome",
        y="Probability",
        color="Outcome",
        color_discrete_map=PRED_COLORS,
        labels={"Probability": "Model probability"},
    )
    probability_figure.update_yaxes(range=[0, 1], tickformat=".0%")
    probability_figure.update_layout(showlegend=False)
    show_chart(probability_figure, "prediction-probabilities", 330)

    st.subheader("Model input features")
    feature_data = pd.DataFrame(prediction["features"].items(), columns=["Feature", "Value"])
    feature_data["Feature"] = feature_data["Feature"].map(FEATURE_LABELS).fillna(feature_data["Feature"])
    feature_data["Value"] = feature_data["Value"].astype(float).round(2)
    st.dataframe(feature_data, width="stretch", hide_index=True)

    with st.container(border=True):
        st.subheader("Weekly score history")
        show_chart(performance_trend_figure(prediction["weekly"]), "prediction-weekly-history", 340)

    metrics = load_json(PERFORMANCE_METRICS_PATH)
    if metrics:
        st.caption(
            f"Held-out accuracy: {metrics['accuracy']:.1%}; most-frequent-class baseline: "
            f"{metrics['baseline_accuracy_most_frequent']:.1%}. Treat this as an indicator, not a guarantee."
        )
    with st.expander("Save prediction to MySQL (optional)", icon=":material/database:"):
        if st.button("Save weekly performance and prediction", icon=":material/save:"):
            try:
                from database.db_utils import save_performance

                saved_count = save_performance(selected_brand, prediction["weekly"], label)
                st.success(f"Saved {saved_count} rows.")
            except Exception as error:
                st.error(str(error))
    st.caption(DISCLAIMER)


def set_example_text():
    st.session_state["analyzer-text"] = st.session_state["analyzer-example"]


def render_text_analyzer():
    st.title("Social Media Text Analyzer", icon=":material/chat:")
    st.write("Enter a social-media post, review or comment to analyze its sentiment and topic.")
    try:
        model, vectorizer = get_sentiment_model()
    except ModelNotFoundError as error:
        st.error(str(error))
        train_models_button()

    examples = [
        "The delivery was fast and the product quality was excellent.",
        "Customer support ignored my refund request.",
        "I bought this product yesterday.",
        "The shoes are comfortable but very expensive.",
    ]
    st.selectbox(
        "Example text",
        ["Choose an example", *examples],
        key="analyzer-example",
        on_change=set_example_text,
    )
    with st.form("text-analysis-form", border=False):
        text = st.text_area(
            "Enter social media text",
            height=130,
            placeholder="The delivery was fast and the product quality was excellent.",
            key="analyzer-text",
        )
        analyze = st.form_submit_button("Analyze text", type="primary", icon=":material/search:")

    if analyze:
        if not text.strip():
            st.warning("Enter text to analyze, or choose an example above.")
            return
        try:
            result = predict_single(text, model, vectorizer)
        except ValueError as error:
            st.warning(str(error))
            return

        topics = detect_topics(text)
        metric_columns = st.columns(4, border=True)
        metric_columns[0].metric("Sentiment", result["sentiment"])
        metric_columns[1].metric("Confidence", f"{result['confidence']:.1%}")
        metric_columns[2].metric("Topic", ", ".join(topics))
        metric_columns[3].metric("Brand impact", result["sentiment"])
        st.caption(brand_impact(result["sentiment"]))

        probability_data = pd.DataFrame(result["probabilities"].items(), columns=["Sentiment", "Probability"])
        figure = px.bar(
            probability_data,
            x="Sentiment",
            y="Probability",
            color="Sentiment",
            color_discrete_map=COLORS,
            labels={"Probability": "Model probability"},
        )
        figure.update_yaxes(range=[0, 1], tickformat=".0%")
        figure.update_layout(showlegend=False)
        show_chart(figure, "text-analysis-probabilities", 300)
        with st.expander("Preprocessed text"):
            st.code(result["cleaned_text"] or "No tokens remain after preprocessing.")


def render_model_evaluation():
    st.title("Model Evaluation", icon=":material/science:")
    st.caption("Metrics are from the training scripts' held-out test sets. The included dataset is synthetic.")
    sentiment_metrics = load_json(SENTIMENT_METRICS_PATH)
    performance_metrics = load_json(PERFORMANCE_METRICS_PATH)

    model_sections = [
        ("Sentiment classification model", "Logistic Regression", sentiment_metrics),
        ("Brand performance prediction model", "Random Forest", performance_metrics),
    ]
    for title, model_name, metrics in model_sections:
        st.header(title, divider="gray")
        st.caption(f"Model: {model_name}")
        if not metrics:
            st.warning("Evaluation metrics are missing. Run `python train_all.py` to regenerate them.")
            continue

        metric_columns = st.columns(4, border=True)
        metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        metric_columns[1].metric("Precision", f"{metrics['precision_weighted']:.3f}")
        metric_columns[2].metric("Recall", f"{metrics['recall_weighted']:.3f}")
        metric_columns[3].metric("F1 score", f"{metrics['f1_weighted']:.3f}")
        st.caption(f"Training rows: {metrics['train_size']:,} · Test rows: {metrics['test_size']:,}")

        matrix_column, details_column = st.columns([1, 1.2])
        with matrix_column, st.container(border=True):
            st.subheader("Confusion matrix")
            show_chart(
                confusion_figure(metrics["confusion_matrix"], metrics["labels"]),
                f"confusion-{model_name.lower().replace(' ', '-')}",
                330,
            )
        with details_column:
            if "baseline_accuracy_most_frequent" in metrics:
                st.metric("Most-frequent-class baseline", f"{metrics['baseline_accuracy_most_frequent']:.3f}")
            st.subheader("Per-class metrics")
            st.dataframe(pd.DataFrame(metrics["per_class"]).T.round(3), width="stretch")

        if "feature_importances" in metrics:
            importance = pd.DataFrame(metrics["feature_importances"].items(), columns=["Feature", "Importance"])
            importance = importance.sort_values("Importance")
            figure = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#3557C8"],
            )
            figure.update_layout(showlegend=False)
            with st.container(border=True):
                st.subheader("Prediction feature importance")
                show_chart(figure, "prediction-feature-importance", 360)
        st.caption(metrics.get("note", ""))


PAGE_GROUPS = {
    "MAIN": [
        st.Page(render_dashboard, title="Dashboard", icon=":material/home:", default=True),
        st.Page(render_dataset, title="Dataset", icon=":material/table_chart:"),
        st.Page(render_sentiment, title="Sentiment", icon=":material/sentiment_satisfied:"),
        st.Page(render_topics, title="Topics", icon=":material/search:"),
    ],
    "ANALYTICS": [
        st.Page(render_brand_performance, title="Brand Performance", icon=":material/monitoring:"),
        st.Page(render_prediction, title="Prediction", icon=":material/query_stats:"),
    ],
    "TOOLS": [
        st.Page(render_text_analyzer, title="Text Analyzer", icon=":material/chat:"),
        st.Page(render_model_evaluation, title="Model Evaluation", icon=":material/science:"),
    ],
}

with st.sidebar:
    st.title("Social Sentiment AI", icon=":material/analytics:")
    st.caption("Brand Performance Prediction")

page = st.navigation(PAGE_GROUPS, position="sidebar", expanded=True)
needs_data = page.title in {
    "Dashboard", "Dataset", "Sentiment", "Topics", "Brand Performance", "Prediction",
}

with st.sidebar:
    st.subheader("Data source")
    source_choice = st.segmented_control(
        "Choose a source",
        ["Built-in sample", "Upload CSV"],
        default="Built-in sample",
        key="data-source",
    )
    uploaded_file = None
    if source_choice == "Upload CSV":
        st.caption("Required: statement, brand, date. Optional: sentiment, engagement.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

data = None
filtered_data = None
selected_brand = None
load_warnings = []
if needs_data:
    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    if source_choice == "Upload CSV" and uploaded_file is None:
        st.info("Upload a CSV in the sidebar, or switch to the built-in sample dataset.", icon=":material/upload:")
        st.stop()
    try:
        raw_data, _ = load_raw(file_bytes)
        use_file_labels = False
        if "sentiment" in raw_data.columns and raw_data["sentiment"].notna().any():
            sentiment_source = st.sidebar.segmented_control(
                "Sentiment source",
                ["Trained model", "Labels from CSV"],
                default="Trained model",
                key="sentiment-source",
            )
            use_file_labels = sentiment_source == "Labels from CSV"
        data, load_warnings = get_processed(file_bytes, use_file_labels)
    except (DataValidationError, ModelNotFoundError, ValueError) as error:
        st.error(str(error))
        if isinstance(error, ModelNotFoundError):
            train_models_button()
        st.stop()
    except Exception as error:
        st.error(f"The data could not be processed. Check the CSV format and try again. Details: {error}")
        st.stop()

    brand_options = sorted(data["brand"].unique())
    with st.sidebar:
        selected_brand = st.selectbox("Brand", brand_options, key="selected-brand")
        date_min = data["date"].min().date()
        date_max = data["date"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="selected-date-range",
        )

    selected_brand_data = data[data["brand"] == selected_brand]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = date_min, date_max
    filtered_data = selected_brand_data[
        (selected_brand_data["date"].dt.date >= start_date)
        & (selected_brand_data["date"].dt.date <= end_date)
    ]
    if filtered_data.empty:
        st.warning("No mentions match this brand and date range. Adjust the filters.")
        st.stop()

page.run()
