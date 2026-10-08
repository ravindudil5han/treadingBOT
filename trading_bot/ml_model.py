"""Machine Learning Signal Classifier Module.

Implements an XGBoost directional classifier trained to predict next-bar returns.
Enforces strict probability thresholding (e.g. >= 65% confidence) to filter out market noise
and emit only high-conviction trade setups.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from trading_bot.config import MLConfig

logger = logging.getLogger("trading_bot.ml_model")


@dataclass
class SignalPrediction:
    """Quantitative directional classification prediction and confidence telemetry."""
    symbol: str
    direction: str  # "LONG", "SHORT", "FLAT"
    confidence: float  # 0.0 to 1.0 (e.g., 0.73 = 73%)
    probabilities: Dict[str, float]  # {"FLAT": p0, "LONG": p1, "SHORT": p2}
    threshold_met: bool
    top_features: List[Tuple[str, float]]
    timestamp: float


class DirectionalClassifier:
    """XGBoost-based directional classifier with walk-forward validation and strict confidence gating."""

    CLASS_MAP = {0: "FLAT", 1: "LONG", 2: "SHORT"}
    REVERSE_CLASS_MAP = {"FLAT": 0, "LONG": 1, "SHORT": 2}

    def __init__(self, cfg: MLConfig, symbol: str = "GLOBAL"):
        self.cfg = cfg
        self.symbol = symbol
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.last_trained_time: float = 0.0
        self.train_bars_count: int = 0
        self.eval_metrics: Dict[str, float] = {}

    @property
    def model_path(self) -> Path:
        clean_sym = self.symbol.replace("/", "_").replace(":", "_")
        return Path("models") / f"{clean_sym}_xgb.json"

    @property
    def metadata_path(self) -> Path:
        clean_sym = self.symbol.replace("/", "_").replace(":", "_")
        return Path("models") / f"{clean_sym}_meta.json"

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train XGBoost directional model using strict chronological walk-forward split.

        Args:
            X: Feature matrix DataFrame
            y: Target series (0: FLAT, 1: LONG, 2: SHORT)

        Returns:
            Dict[str, float]: Out-of-sample evaluation metrics
        """
        if len(X) < 80:
            raise ValueError(f"Insufficient samples to train ML model: {len(X)} < 80")

        self.feature_names = list(X.columns)
        n_samples = len(X)
        split_idx = int(n_samples * self.cfg.train_test_split_ratio)

        # Chronological train/test split (no shuffling, preserves time causal structure)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Handle class imbalance if present
        classes_present = np.unique(y_train)
        logger.debug(f"Training [{self.symbol}] on {len(X_train)} samples, testing on {len(X_test)} samples. Classes: {classes_present}")

        # Instantiate XGBoost Multi-Class Classifier
        self.model = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            min_child_weight=self.cfg.min_child_weight,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

        # Fit model on training set
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate on Out-Of-Sample test data
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        test_acc = float(accuracy_score(y_test, y_pred))
        try:
            # Ensure all 3 class columns accounted for
            test_loss = float(log_loss(y_test, y_pred_proba, labels=[0, 1, 2]))
        except Exception:
            test_loss = 0.0

        # Calculate high-confidence subset accuracy
        conf_mask = np.max(y_pred_proba, axis=1) >= self.cfg.confidence_threshold
        high_conf_samples = int(np.sum(conf_mask))
        high_conf_acc = float(accuracy_score(y_test[conf_mask], y_pred[conf_mask])) if high_conf_samples > 0 else 0.0

        self.eval_metrics = {
            "test_accuracy": test_acc,
            "test_log_loss": test_loss,
            "high_conf_samples": high_conf_samples,
            "high_conf_accuracy": high_conf_acc,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        self.last_trained_time = time.time()
        self.train_bars_count = len(X)

        # Persist model and metadata
        self.save()

        logger.info(
            f"[{self.symbol}] XGBoost Training Complete | Test Acc: {test_acc*100:.1f}% | "
            f"High-Conf Acc (>{self.cfg.confidence_threshold*100:.0f}%): {high_conf_acc*100:.1f}% "
            f"({high_conf_samples} samples)"
        )
        return self.eval_metrics

    def predict(self, X_latest: pd.DataFrame) -> SignalPrediction:
        """Run inference on the latest candle features with strict probability thresholding.

        Args:
            X_latest: Single-row or multi-row DataFrame containing identical feature columns.

        Returns:
            SignalPrediction: Directional signal and confidence breakdown.
        """
        if self.model is None:
            if self.load():
                logger.info(f"Loaded existing model from disk for {self.symbol}")
            else:
                raise RuntimeError(f"Model for {self.symbol} is neither trained nor loaded from disk.")

        # Ensure correct column alignment
        X_aligned = X_latest[self.feature_names].copy()

        # Predict class probabilities
        probs = self.model.predict_proba(X_aligned)[0]  # shape: (3,) -> [P(FLAT), P(LONG), P(SHORT)]

        # Map to class dictionary
        p_flat = float(probs[0])
        p_long = float(probs[1]) if len(probs) > 1 else 0.0
        p_short = float(probs[2]) if len(probs) > 2 else 0.0

        prob_dict = {
            "FLAT": p_flat,
            "LONG": p_long,
            "SHORT": p_short
        }

        # Strict Probability Thresholding
        threshold = self.cfg.confidence_threshold
        direction = "FLAT"
        confidence = p_flat
        threshold_met = False

        if p_long >= threshold and p_long > p_short:
            direction = "LONG"
            confidence = p_long
            threshold_met = True
        elif p_short >= threshold and p_short > p_long:
            direction = "SHORT"
            confidence = p_short
            threshold_met = True
        else:
            direction = "FLAT"
            confidence = max(p_flat, p_long, p_short)
            threshold_met = False

        top_feats = self.get_top_features(top_n=5)

        return SignalPrediction(
            symbol=self.symbol,
            direction=direction,
            confidence=confidence,
            probabilities=prob_dict,
            threshold_met=threshold_met,
            top_features=top_feats,
            timestamp=time.time()
        )

    def get_top_features(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract top feature importances by Gain."""
        if self.model is None:
            return []
        try:
            importances = self.model.feature_importances_
            pairs = list(zip(self.feature_names, [float(x) for x in importances]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        except Exception:
            return []

    def save(self) -> None:
        """Save trained model and feature metadata to disk."""
        if self.model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.model_path))

        meta = {
            "symbol": self.symbol,
            "feature_names": self.feature_names,
            "eval_metrics": self.eval_metrics,
            "last_trained_time": self.last_trained_time,
            "train_bars_count": self.train_bars_count,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self) -> bool:
        """Load model from disk if available."""
        if not self.model_path.exists() or not self.metadata_path.exists():
            return False
        try:
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.eval_metrics = meta.get("eval_metrics", {})
            self.last_trained_time = meta.get("last_trained_time", 0.0)

            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            return False


class MultiSymbolMLManager:
    """Coordinates and manages individual directional classifiers across multiple perpetual assets."""

    def __init__(self, ml_cfg: MLConfig):
        self.cfg = ml_cfg
        self.classifiers: Dict[str, DirectionalClassifier] = {}

    def get_classifier(self, symbol: str) -> DirectionalClassifier:
        if symbol not in self.classifiers:
            self.classifiers[symbol] = DirectionalClassifier(self.cfg, symbol=symbol)
        return self.classifiers[symbol]

    def train_or_update(self, symbol: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        clf = self.get_classifier(symbol)
        return clf.train(X, y)

    def predict_signal(self, symbol: str, X_latest: pd.DataFrame) -> SignalPrediction:
        clf = self.get_classifier(symbol)
        return clf.predict(X_latest)
