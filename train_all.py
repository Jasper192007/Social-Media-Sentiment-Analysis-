"""Train both models in the right order:  python train_all.py"""
from training.train_sentiment_model import train_sentiment
from training.train_performance_model import train_performance

if __name__ == "__main__":
    print("=== 1/2  Sentiment model ===")
    train_sentiment()
    print("\n=== 2/2  Brand performance model ===")
    train_performance()
    print("\nAll models trained and saved in the model/ folder.")
