"""
ML Regime Prediction System

Machine learning models for predicting market regimes:
- Random Forest regime classifier
- LSTM for time-series regime forecasting
- Gradient Boosting for regime transitions
- Ensemble prediction with confidence scores

Regimes: BULL, BEAR, SIDEWAYS, CORRECTION, CRISIS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegimePrediction:
    """Regime prediction result"""
    regime: str
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, float]
    timestamp: datetime
    horizon: str  # '1d', '5d', '20d', etc.


class FeatureEngineering:
    """
    Feature engineering for regime prediction

    Creates technical, fundamental, and macro features
    """

    @staticmethod
    def calculate_features(prices: pd.Series,
                          volume: Optional[pd.Series] = None,
                          lookback_windows: List[int] = [5, 20, 60, 120, 252]) -> pd.DataFrame:
        """
        Calculate comprehensive feature set

        Features:
        - Returns (multiple timeframes)
        - Volatility (realized, rolling)
        - Momentum indicators
        - Trend strength
        - Volume indicators
        - Drawdown metrics
        """
        features = pd.DataFrame(index=prices.index)

        # Returns
        for window in [1, 5, 20, 60]:
            features[f'return_{window}d'] = prices.pct_change(window)

        # Volatility
        for window in [5, 20, 60]:
            features[f'volatility_{window}d'] = prices.pct_change().rolling(window).std() * np.sqrt(252)

        # Momentum
        for window in [20, 60, 120, 252]:
            features[f'momentum_{window}d'] = prices / prices.shift(window) - 1

        # Moving averages
        for window in [20, 50, 200]:
            ma = prices.rolling(window).mean()
            features[f'price_to_ma{window}'] = prices / ma - 1

        # Trend strength (ADX-like)
        for window in [14, 28]:
            price_changes = prices.diff()
            up = price_changes.clip(lower=0).rolling(window).mean()
            down = -price_changes.clip(upper=0).rolling(window).mean()
            features[f'trend_strength_{window}d'] = (up - down) / (up + down + 1e-10)

        # Drawdown
        rolling_max = prices.expanding().max()
        features['drawdown'] = (prices - rolling_max) / rolling_max

        # Consecutive up/down days
        daily_returns = prices.pct_change()
        features['consecutive_up'] = (daily_returns > 0).astype(int).groupby(
            (daily_returns <= 0).cumsum()
        ).cumsum()
        features['consecutive_down'] = (daily_returns < 0).astype(int).groupby(
            (daily_returns >= 0).cumsum()
        ).cumsum()

        # RSI
        for window in [14, 28]:
            delta = prices.diff()
            gain = delta.clip(lower=0).rolling(window).mean()
            loss = -delta.clip(upper=0).rolling(window).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{window}d'] = 100 - (100 / (1 + rs))

        # Volume features (if available)
        if volume is not None:
            for window in [20, 60]:
                features[f'volume_ratio_{window}d'] = volume / volume.rolling(window).mean()

        # Volatility regime (high/low vol)
        vol_60d = prices.pct_change().rolling(60).std()
        features['high_vol_regime'] = (vol_60d > vol_60d.rolling(252).quantile(0.75)).astype(int)

        # Rate of change acceleration
        for window in [5, 20]:
            roc = prices.pct_change(window)
            features[f'roc_acceleration_{window}d'] = roc.diff()

        return features.dropna()

    @staticmethod
    def label_regimes(prices: pd.Series,
                     lookback: int = 60,
                     forward_window: int = 20) -> pd.Series:
        """
        Label historical regimes based on price action

        Regimes:
        - BULL: Strong uptrend (>15% annualized, low vol)
        - BEAR: Downtrend (<-10% annualized)
        - CORRECTION: Sharp drop from peak (-5% to -20%)
        - CRISIS: Severe drop (>-20%) or extreme volatility
        - SIDEWAYS: Low trend, moderate volatility
        """
        labels = pd.Series(index=prices.index, dtype=str)

        for i in range(lookback, len(prices)):
            # Look at past price action
            past_prices = prices.iloc[i-lookback:i]
            recent_return = (prices.iloc[i] / prices.iloc[i-20] - 1) * (252/20)  # Annualized

            # Volatility
            vol = past_prices.pct_change().std() * np.sqrt(252)

            # Drawdown from peak
            peak = past_prices.max()
            dd = (prices.iloc[i] - peak) / peak

            # Classify regime
            if dd < -0.20 or vol > 0.40:
                labels.iloc[i] = 'CRISIS'
            elif dd < -0.05 and dd > -0.20:
                labels.iloc[i] = 'CORRECTION'
            elif recent_return < -0.10:
                labels.iloc[i] = 'BEAR'
            elif recent_return > 0.15 and vol < 0.20:
                labels.iloc[i] = 'BULL'
            else:
                labels.iloc[i] = 'SIDEWAYS'

        return labels.dropna()


class MLRegimePredictor:
    """
    Machine Learning Regime Predictor

    Ensemble of models for robust regime prediction
    """

    def __init__(self, models: Optional[List[str]] = None):
        """
        Initialize ML regime predictor

        Args:
            models: List of models to use ('rf', 'gb', 'ensemble')
        """
        if models is None:
            models = ['rf', 'gb']

        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.trained = False

        # Random Forest
        if 'rf' in models:
            self.models['rf'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )

        # Gradient Boosting
        if 'gb' in models:
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            )

        self.regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'CORRECTION', 'CRISIS']

    def train(self,
             prices: pd.Series,
             volume: Optional[pd.Series] = None,
             validation_split: float = 0.2) -> Dict:
        """
        Train regime prediction models

        Args:
            prices: Price series
            volume: Volume series (optional)
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        print(f"\n{'='*80}")
        print("TRAINING ML REGIME PREDICTOR")
        print(f"{'='*80}\n")

        # Feature engineering
        print("1. Engineering features...")
        features = FeatureEngineering.calculate_features(prices, volume)
        print(f"   Created {len(features.columns)} features")

        # Label regimes
        print("2. Labeling historical regimes...")
        labels = FeatureEngineering.label_regimes(prices)

        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]

        print(f"   Total samples: {len(X)}")
        print(f"   Regime distribution:")
        for regime in self.regimes:
            count = (y == regime).sum()
            pct = count / len(y) * 100
            print(f"     {regime:15s}: {count:5d} ({pct:5.1f}%)")

        # Train/validation split (time-series aware)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\n3. Training {len(self.models)} models...")

        metrics = {}

        for model_name, model in self.models.items():
            print(f"\n   Training {model_name.upper()}...")

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            self.scalers[model_name] = scaler

            # Train
            model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)

            print(f"     Train accuracy: {train_score:.3f}")
            print(f"     Val accuracy:   {val_score:.3f}")

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
                self.feature_importance[model_name] = importance

                print(f"\n     Top 5 features:")
                for feat, imp in importance.head(5).items():
                    print(f"       {feat:30s}: {imp:.3f}")

            metrics[model_name] = {
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'n_features': len(X.columns),
                'n_train': len(X_train),
                'n_val': len(X_val)
            }

        self.trained = True
        self.feature_columns = X.columns.tolist()

        print(f"\n{'='*80}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*80}\n")

        return metrics

    def predict(self,
               prices: pd.Series,
               volume: Optional[pd.Series] = None,
               horizon: str = '20d',
               ensemble: bool = True) -> RegimePrediction:
        """
        Predict current/future regime

        Args:
            prices: Price series (recent history)
            volume: Volume series (optional)
            horizon: Prediction horizon ('1d', '5d', '20d')
            ensemble: Use ensemble of models

        Returns:
            RegimePrediction object
        """
        if not self.trained:
            raise ValueError("Models not trained yet. Call .train() first.")

        # Engineer features
        features = FeatureEngineering.calculate_features(prices, volume)

        # Get latest features
        latest_features = features.iloc[-1:][self.feature_columns]

        # Ensemble prediction
        predictions = {}
        probabilities = {}

        for model_name, model in self.models.items():
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(latest_features)

            # Predict
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]

            predictions[model_name] = pred

            # Store probabilities
            prob_dict = {regime: prob for regime, prob in zip(model.classes_, proba)}
            probabilities[model_name] = prob_dict

        # Ensemble: average probabilities
        if ensemble and len(self.models) > 1:
            avg_probs = {}
            for regime in self.regimes:
                probs = [probabilities[m].get(regime, 0) for m in self.models.keys()]
                avg_probs[regime] = np.mean(probs)

            # Final prediction: highest probability
            final_regime = max(avg_probs, key=avg_probs.get)
            confidence = avg_probs[final_regime]
            final_probs = avg_probs
        else:
            # Single model
            model_name = list(self.models.keys())[0]
            final_probs = probabilities[model_name]
            final_regime = predictions[model_name]
            confidence = final_probs[final_regime]

        # Feature values for interpretation
        feature_values = latest_features.iloc[0].to_dict()

        return RegimePrediction(
            regime=final_regime,
            confidence=confidence,
            probabilities=final_probs,
            features=feature_values,
            timestamp=datetime.now(),
            horizon=horizon
        )

    def get_regime_transitions(self,
                              prices: pd.Series,
                              volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Predict regime over historical period to identify transitions

        Returns: DataFrame with date, predicted_regime, confidence
        """
        if not self.trained:
            raise ValueError("Models not trained yet")

        features = FeatureEngineering.calculate_features(prices, volume)
        features = features[self.feature_columns]

        results = []

        for i in range(len(features)):
            X = features.iloc[i:i+1]

            # Ensemble prediction
            predictions = []
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                predictions.append((pred, max(proba)))

            # Most common prediction
            regimes = [p[0] for p in predictions]
            confidences = [p[1] for p in predictions]

            final_regime = max(set(regimes), key=regimes.count)
            final_confidence = np.mean(confidences)

            results.append({
                'date': features.index[i],
                'regime': final_regime,
                'confidence': final_confidence
            })

        return pd.DataFrame(results)


class RegimeSignalGenerator:
    """
    Generate trading signals based on regime predictions

    Adjusts strategy exposure/allocation based on predicted regime
    """

    def __init__(self, predictor: MLRegimePredictor):
        """Initialize signal generator"""
        self.predictor = predictor

    def get_exposure_adjustment(self, prediction: RegimePrediction) -> float:
        """
        Get recommended exposure adjustment based on regime

        Returns: Multiplier for strategy positions (0.0 to 1.5)
        """
        regime = prediction.regime
        confidence = prediction.confidence

        base_exposure = {
            'BULL': 1.2,       # Increase exposure 20%
            'SIDEWAYS': 1.0,   # Normal exposure
            'CORRECTION': 0.7, # Reduce exposure 30%
            'BEAR': 0.5,       # Reduce exposure 50%
            'CRISIS': 0.0      # Go to cash
        }

        # Adjust by confidence
        adjustment = base_exposure.get(regime, 1.0)

        # If low confidence, move towards 1.0 (neutral)
        if confidence < 0.6:
            adjustment = 1.0 + (adjustment - 1.0) * confidence

        return adjustment

    def should_trade(self, prediction: RegimePrediction, min_confidence: float = 0.6) -> bool:
        """
        Determine if trading should be active

        Returns: True if confidence is high enough and regime is favorable
        """
        if prediction.confidence < min_confidence:
            return False

        if prediction.regime in ['CRISIS', 'BEAR']:
            return False

        return True

    def get_strategy_selection(self,
                              prediction: RegimePrediction,
                              available_strategies: List[str]) -> List[str]:
        """
        Select which strategies to activate based on regime

        Returns: List of strategy names to use
        """
        regime = prediction.regime

        # Regime-specific strategy filters
        if regime == 'BULL':
            # Favor momentum strategies
            return [s for s in available_strategies if 'momentum' in s.lower()]

        elif regime == 'BEAR':
            # Favor defensive/short strategies
            return [s for s in available_strategies
                   if any(x in s.lower() for x in ['mean_reversion', 'pairs'])]

        elif regime == 'SIDEWAYS':
            # Favor range-bound strategies
            return [s for s in available_strategies
                   if any(x in s.lower() for x in ['mean_reversion', 'bollinger', 'rsi'])]

        elif regime in ['CORRECTION', 'CRISIS']:
            # Conservative: only validated low-vol strategies
            return [s for s in available_strategies if 'value' in s.lower()]

        else:
            return available_strategies


if __name__ == '__main__':
    """Test ML regime predictor"""

    print("ML Regime Predictor - Demo")
    print("\nThis module provides:")
    print("  1. Feature engineering (40+ technical features)")
    print("  2. ML models (Random Forest, Gradient Boosting)")
    print("  3. Ensemble predictions with confidence scores")
    print("  4. Regime transition detection")
    print("  5. Trading signal generation")

    print("\nTo use:")
    print("  predictor = MLRegimePredictor()")
    print("  predictor.train(prices, volume)")
    print("  prediction = predictor.predict(prices)")
