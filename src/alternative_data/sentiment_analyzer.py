"""
Financial Sentiment Analysis

Implements NLP-based sentiment analysis for financial markets:
- FinBERT (BERT fine-tuned on financial text)
- News sentiment extraction
- Social media sentiment (Twitter, Reddit)
- Earnings call transcript analysis
- Sentiment scoring and aggregation

Provides alternative data signals for alpha generation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Install with: pip install transformers")


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    text: str
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # confidence score (0-1)
    sentiment_value: float  # normalized sentiment (-1 to +1)
    timestamp: datetime


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment for a ticker"""
    ticker: str
    date: datetime
    num_articles: int
    avg_sentiment: float  # -1 to +1
    sentiment_std: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    weighted_sentiment: float  # Volume-weighted or importance-weighted


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analysis

    FinBERT is BERT fine-tuned on financial news and reports.
    Achieves state-of-the-art performance on financial text classification.

    Model: ProsusAI/finbert
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT analyzer

        Args:
            model_name: Hugging Face model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers torch")

        print(f"Loading FinBERT model: {model_name}...")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create pipeline for easy inference
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU (use 0 for GPU)
        )

        print("✅ FinBERT model loaded successfully")

    def analyze_text(self, text: str, timestamp: Optional[datetime] = None) -> SentimentScore:
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze
            timestamp: Timestamp of the text

        Returns:
            SentimentScore object
        """
        # Truncate if too long (BERT has 512 token limit)
        if len(text) > 500:
            text = text[:500]

        # Get prediction
        result = self.pipeline(text)[0]

        # Extract label and score
        label = result['label'].lower()
        confidence = result['score']

        # Convert to normalized sentiment (-1 to +1)
        if label == 'positive':
            sentiment_value = confidence
        elif label == 'negative':
            sentiment_value = -confidence
        else:  # neutral
            sentiment_value = 0.0

        return SentimentScore(
            text=text,
            label=label,
            score=confidence,
            sentiment_value=sentiment_value,
            timestamp=timestamp or datetime.now()
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """
        Analyze sentiment of multiple texts (faster than one-by-one)

        Args:
            texts: List of texts

        Returns:
            List of SentimentScore objects
        """
        # Truncate long texts
        truncated_texts = [text[:500] if len(text) > 500 else text for text in texts]

        # Batch prediction
        results = self.pipeline(truncated_texts)

        sentiment_scores = []
        for text, result in zip(texts, results):
            label = result['label'].lower()
            confidence = result['score']

            if label == 'positive':
                sentiment_value = confidence
            elif label == 'negative':
                sentiment_value = -confidence
            else:
                sentiment_value = 0.0

            sentiment_scores.append(SentimentScore(
                text=text,
                label=label,
                score=confidence,
                sentiment_value=sentiment_value,
                timestamp=datetime.now()
            ))

        return sentiment_scores


class NewsSentimentAggregator:
    """
    Aggregate sentiment from multiple news sources

    Handles:
    - News headline sentiment
    - Article sentiment
    - Time-weighted aggregation
    - Source credibility weighting
    """

    def __init__(self, analyzer: FinBERTAnalyzer):
        """Initialize with a sentiment analyzer"""
        self.analyzer = analyzer

    def aggregate_daily_sentiment(self,
                                  articles: List[Dict],
                                  ticker: str,
                                  date: datetime,
                                  decay_hours: float = 24.0) -> AggregatedSentiment:
        """
        Aggregate sentiment for a ticker on a given day

        Args:
            articles: List of dicts with keys: 'text', 'timestamp', 'importance'
            ticker: Stock ticker
            date: Date to aggregate
            decay_hours: Time decay for older news (default 24 hours)

        Returns:
            AggregatedSentiment object
        """
        if not articles:
            return AggregatedSentiment(
                ticker=ticker,
                date=date,
                num_articles=0,
                avg_sentiment=0.0,
                sentiment_std=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=0.0,
                weighted_sentiment=0.0
            )

        # Analyze all articles
        texts = [article['text'] for article in articles]
        sentiments = self.analyzer.analyze_batch(texts)

        # Calculate weights (time decay + importance)
        weights = []
        sentiment_values = []

        for article, sentiment in zip(articles, sentiments):
            # Time decay
            hours_old = (date - article['timestamp']).total_seconds() / 3600
            time_weight = np.exp(-hours_old / decay_hours)

            # Importance weight
            importance = article.get('importance', 1.0)

            # Combined weight
            weight = time_weight * importance
            weights.append(weight)
            sentiment_values.append(sentiment.sentiment_value)

        weights = np.array(weights)
        sentiment_values = np.array(sentiment_values)

        # Normalize weights
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Calculate metrics
        avg_sentiment = np.mean(sentiment_values)
        sentiment_std = np.std(sentiment_values)
        weighted_sentiment = np.sum(weights * sentiment_values)

        # Count labels
        labels = [s.label for s in sentiments]
        positive_ratio = labels.count('positive') / len(labels)
        negative_ratio = labels.count('negative') / len(labels)
        neutral_ratio = labels.count('neutral') / len(labels)

        return AggregatedSentiment(
            ticker=ticker,
            date=date,
            num_articles=len(articles),
            avg_sentiment=avg_sentiment,
            sentiment_std=sentiment_std,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            weighted_sentiment=weighted_sentiment
        )

    def create_sentiment_time_series(self,
                                    articles_by_date: Dict[datetime, List[Dict]],
                                    ticker: str) -> pd.DataFrame:
        """
        Create time series of daily sentiment

        Args:
            articles_by_date: Dict mapping dates to lists of articles
            ticker: Stock ticker

        Returns:
            DataFrame with daily sentiment metrics
        """
        results = []

        for date, articles in sorted(articles_by_date.items()):
            agg_sentiment = self.aggregate_daily_sentiment(articles, ticker, date)

            results.append({
                'date': date,
                'ticker': ticker,
                'num_articles': agg_sentiment.num_articles,
                'avg_sentiment': agg_sentiment.avg_sentiment,
                'weighted_sentiment': agg_sentiment.weighted_sentiment,
                'sentiment_std': agg_sentiment.sentiment_std,
                'positive_ratio': agg_sentiment.positive_ratio,
                'negative_ratio': agg_sentiment.negative_ratio,
                'neutral_ratio': agg_sentiment.neutral_ratio
            })

        return pd.DataFrame(results).set_index('date')


class SentimentSignalGenerator:
    """
    Generate trading signals from sentiment data

    Converts sentiment scores into actionable trading signals
    """

    def __init__(self):
        pass

    def sentiment_to_signal(self,
                           sentiment_ts: pd.DataFrame,
                           method: str = 'threshold',
                           **kwargs) -> pd.Series:
        """
        Convert sentiment to trading signal

        Args:
            sentiment_ts: DataFrame with sentiment metrics
            method: 'threshold', 'momentum', 'z_score', 'change'
            **kwargs: Method-specific parameters

        Returns:
            Series of trading signals (-1, 0, 1)
        """
        if method == 'threshold':
            # Simple threshold-based signals
            pos_threshold = kwargs.get('pos_threshold', 0.2)
            neg_threshold = kwargs.get('neg_threshold', -0.2)

            signals = pd.Series(0, index=sentiment_ts.index)
            signals[sentiment_ts['weighted_sentiment'] > pos_threshold] = 1
            signals[sentiment_ts['weighted_sentiment'] < neg_threshold] = -1

        elif method == 'momentum':
            # Sentiment momentum
            window = kwargs.get('window', 5)
            sentiment_ma = sentiment_ts['weighted_sentiment'].rolling(window).mean()
            sentiment_momentum = sentiment_ts['weighted_sentiment'] - sentiment_ma

            pos_threshold = kwargs.get('pos_threshold', 0.1)
            neg_threshold = kwargs.get('neg_threshold', -0.1)

            signals = pd.Series(0, index=sentiment_ts.index)
            signals[sentiment_momentum > pos_threshold] = 1
            signals[sentiment_momentum < neg_threshold] = -1

        elif method == 'z_score':
            # Z-score normalization
            window = kwargs.get('window', 20)
            rolling_mean = sentiment_ts['weighted_sentiment'].rolling(window).mean()
            rolling_std = sentiment_ts['weighted_sentiment'].rolling(window).std()
            z_score = (sentiment_ts['weighted_sentiment'] - rolling_mean) / rolling_std

            pos_threshold = kwargs.get('pos_threshold', 1.0)
            neg_threshold = kwargs.get('neg_threshold', -1.0)

            signals = pd.Series(0, index=sentiment_ts.index)
            signals[z_score > pos_threshold] = 1
            signals[z_score < neg_threshold] = -1

        elif method == 'change':
            # Sentiment change (delta)
            sentiment_change = sentiment_ts['weighted_sentiment'].diff()

            pos_threshold = kwargs.get('pos_threshold', 0.1)
            neg_threshold = kwargs.get('neg_threshold', -0.1)

            signals = pd.Series(0, index=sentiment_ts.index)
            signals[sentiment_change > pos_threshold] = 1
            signals[sentiment_change < neg_threshold] = -1

        else:
            raise ValueError(f"Unknown method: {method}")

        return signals


if __name__ == '__main__':
    """Test sentiment analyzer"""

    if not TRANSFORMERS_AVAILABLE:
        print("Install transformers: pip install transformers torch")
        exit(1)

    print("Financial Sentiment Analysis - Demo\n")

    # Initialize FinBERT
    analyzer = FinBERTAnalyzer()

    # Test texts (financial news examples)
    test_texts = [
        "Apple reports record quarterly earnings, beats analyst expectations by 20%",
        "Tesla stock plunges after disappointing delivery numbers",
        "Federal Reserve maintains interest rates, signals cautious approach",
        "Amazon announces major expansion into healthcare sector",
        "Oil prices surge amid supply concerns and geopolitical tensions",
        "Microsoft cloud revenue growth slows, shares drop in after-hours trading",
    ]

    print("\nAnalyzing individual texts:\n")
    for text in test_texts:
        sentiment = analyzer.analyze_text(text)
        emoji = "📈" if sentiment.label == "positive" else "📉" if sentiment.label == "negative" else "➡️"
        print(f"{emoji} [{sentiment.label.upper():8s}] ({sentiment.sentiment_value:+.3f}) - {text[:60]}...")

    # Test batch analysis
    print("\n\nTesting batch analysis (faster)...")
    batch_sentiments = analyzer.analyze_batch(test_texts)
    print(f"Analyzed {len(batch_sentiments)} texts in batch mode")

    # Aggregate sentiment
    print("\n\nTesting sentiment aggregation...")
    aggregator = NewsSentimentAggregator(analyzer)

    # Create fake articles
    articles = [
        {'text': text, 'timestamp': datetime.now() - timedelta(hours=i), 'importance': 1.0}
        for i, text in enumerate(test_texts)
    ]

    agg = aggregator.aggregate_daily_sentiment(articles, 'AAPL', datetime.now())

    print(f"\nAggregated Sentiment for AAPL:")
    print(f"  Articles:           {agg.num_articles}")
    print(f"  Avg Sentiment:      {agg.avg_sentiment:+.3f}")
    print(f"  Weighted Sentiment: {agg.weighted_sentiment:+.3f}")
    print(f"  Positive Ratio:     {agg.positive_ratio:.1%}")
    print(f"  Negative Ratio:     {agg.negative_ratio:.1%}")
    print(f"  Neutral Ratio:      {agg.neutral_ratio:.1%}")

    print("\n✅ Sentiment analyzer working!")
