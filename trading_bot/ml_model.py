"""Machine Learning Stacking Ensemble Classifier Module.

Combines regularized XGBoost, Random Forest, and Extra Trees classifiers
into a high-accuracy meta-ensemble. Enforces multi-model consensus gating,
probability thresholding, and walk-forward chronological evaluation to filter
out false breakouts and capture high-conviction futures trend continuations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
    ensemble_consensus: bool = True
    consensus_ratio: float = 1.0
    model_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class DirectionalClassifier:
    """Stacking Ensemble directional classifier with walk-forward validation and strict consensus gating."""

    CLASS_MAP = {0: "FLAT", 1: "LONG", 2: "SHORT"}
    REVERSE_CLASS_MAP = {"FLAT": 0, "LONG": 1, "SHORT": 2}

    def __init__(self, cfg: MLConfig, symbol: str = "GLOBAL"):
        self.cfg = cfg
        self.symbol = symbol
        self.model: Optional[xgb.XGBClassifier] = None  # Kept for direct backwards compatibility
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.et_model: Optional[ExtraTreesClassifier] = None
        self.feature_names: List[str] = []
        self.last_trained_time: float = 0.0
        self.train_bars_count: int = 0
        self.eval_metrics: Dict[str, float] = {}

    @property
    def model_path(self) -> Path:
        clean_sym = self.symbol.replace("/", "_").replace(":", "_")
        return Path("models") / f"{clean_sym}_xgb.json"

    @property
    def rf_path(self) -> Path:
        clean_sym = self.symbol.replace("/", "_").replace(":", "_")
        return Path("models") / f"{clean_sym}_rf.joblib"

    @property
    def et_path(self) -> Path:
        clean_sym = self.symbol.replace("/", "_").replace(":", "_")
        return Path("models") / f"{clean_sym}_et.joblib"

    @property
    def metadata_path(self) -> Path:
        clean_sym = self.symbol.replace("/", "_").replace(":", "_")
        return Path("models") / f"{clean_sym}_meta.json"

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train Stacking Ensemble models using strict chronological walk-forward split.

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

        classes_present = np.unique(y_train)
        logger.debug(f"Training [{self.symbol}] on {len(X_train)} samples, testing on {len(X_test)} samples. Classes: {classes_present}")

        # -------------------------------------------------------------
        # 1. Model 1: XGBoost Gradient Boosted Decision Trees
        # -------------------------------------------------------------
        self.xgb_model = xgb.XGBClassifier(
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
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        self.model = self.xgb_model  # Backwards compatibility alias

        y_pred_proba_xgb = self.xgb_model.predict_proba(X_test)
        acc_xgb = float(accuracy_score(y_test, np.argmax(y_pred_proba_xgb, axis=1)))

        # -------------------------------------------------------------
        # 2. Model 2 & 3: Random Forest & Extra Trees Ensembles
        # -------------------------------------------------------------
        if self.cfg.use_ensemble:
            tree_depth = max(3, self.cfg.max_depth + 1)
            self.rf_model = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=tree_depth,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            y_pred_proba_rf = self.rf_model.predict_proba(X_test)
            acc_rf = float(accuracy_score(y_test, np.argmax(y_pred_proba_rf, axis=1)))

            self.et_model = ExtraTreesClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=tree_depth,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            )
            self.et_model.fit(X_train, y_train)
            y_pred_proba_et = self.et_model.predict_proba(X_test)
            acc_et = float(accuracy_score(y_test, np.argmax(y_pred_proba_et, axis=1)))

            # Stacking consensus blend (0.45 XGBoost + 0.35 Random Forest + 0.20 Extra Trees)
            y_pred_proba_ens = 0.45 * y_pred_proba_xgb + 0.35 * y_pred_proba_rf + 0.20 * y_pred_proba_et
            y_pred_proba_ens = y_pred_proba_ens / (np.sum(y_pred_proba_ens, axis=1, keepdims=True) + 1e-12)
        else:
            y_pred_proba_ens = y_pred_proba_xgb
            acc_rf = 0.0
            acc_et = 0.0

        y_pred_ens = np.argmax(y_pred_proba_ens, axis=1)
        test_acc = float(accuracy_score(y_test, y_pred_ens))

        try:
            test_loss = float(log_loss(y_test, y_pred_proba_ens, labels=[0, 1, 2]))
        except Exception:
            test_loss = 0.0

        # Calculate high-confidence subset accuracy
        conf_mask = np.max(y_pred_proba_ens, axis=1) >= self.cfg.confidence_threshold
        high_conf_samples = int(np.sum(conf_mask))
        high_conf_acc = float(accuracy_score(y_test[conf_mask], y_pred_ens[conf_mask])) if high_conf_samples > 0 else 0.0

        self.eval_metrics = {
            "test_accuracy": test_acc,
            "xgb_accuracy": acc_xgb,
            "rf_accuracy": acc_rf,
            "et_accuracy": acc_et,
            "test_log_loss": test_loss,
            "high_conf_samples": high_conf_samples,
            "high_conf_accuracy": high_conf_acc,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        self.last_trained_time = time.time()
        self.train_bars_count = len(X)

        # Persist models and metadata
        self.save()

        logger.info(
            f"[{self.symbol}] Ensemble Training Complete | Test Acc: {test_acc*100:.1f}% "
            f"(XGB: {acc_xgb*100:.1f}%, RF: {acc_rf*100:.1f}%, ET: {acc_et*100:.1f}%) | "
            f"High-Conf Acc (>{self.cfg.confidence_threshold*100:.0f}%): {high_conf_acc*100:.1f}% "
            f"({high_conf_samples} samples)"
        )
        return self.eval_metrics

    def predict(self, X_latest: pd.DataFrame) -> SignalPrediction:
        """Run inference on the latest candle features with multi-model consensus gating.

        Args:
            X_latest: Single-row DataFrame containing identical feature columns.

        Returns:
            SignalPrediction: Directional signal, ensemble confidence, and vote breakdown.
        """
        if self.xgb_model is None:
            if not self.load():
                raise RuntimeError(f"Model for {self.symbol} is neither trained nor loaded from disk.")

        X_aligned = X_latest[self.feature_names].copy()

        # 1. XGBoost inference
        p_xgb = self.xgb_model.predict_proba(X_aligned)[0]
        dir_xgb = self.CLASS_MAP[int(np.argmax(p_xgb))]

        model_breakdown = {
            "xgboost": {
                "direction": dir_xgb,
                "confidence": float(np.max(p_xgb)),
                "probs": {"FLAT": float(p_xgb[0]), "LONG": float(p_xgb[1]), "SHORT": float(p_xgb[2])}
            }
        }

        # 2. Ensemble models inference if available
        if self.cfg.use_ensemble and self.rf_model is not None and self.et_model is not None:
            p_rf = self.rf_model.predict_proba(X_aligned)[0]
            p_et = self.et_model.predict_proba(X_aligned)[0]

            dir_rf = self.CLASS_MAP[int(np.argmax(p_rf))]
            dir_et = self.CLASS_MAP[int(np.argmax(p_et))]

            model_breakdown["random_forest"] = {
                "direction": dir_rf,
                "confidence": float(np.max(p_rf)),
                "probs": {"FLAT": float(p_rf[0]), "LONG": float(p_rf[1]), "SHORT": float(p_rf[2])}
            }
            model_breakdown["extra_trees"] = {
                "direction": dir_et,
                "confidence": float(np.max(p_et)),
                "probs": {"FLAT": float(p_et[0]), "LONG": float(p_et[1]), "SHORT": float(p_et[2])}
            }

            # Weighted consensus probability vector
            probs = 0.45 * p_xgb + 0.35 * p_rf + 0.20 * p_et
            probs = probs / (np.sum(probs) + 1e-12)
            all_dirs = [dir_xgb, dir_rf, dir_et]
        else:
            probs = p_xgb
            all_dirs = [dir_xgb]

        p_flat = float(probs[0])
        p_long = float(probs[1]) if len(probs) > 1 else 0.0
        p_short = float(probs[2]) if len(probs) > 2 else 0.0

        prob_dict = {"FLAT": p_flat, "LONG": p_long, "SHORT": p_short}

        # Directional probability candidate
        raw_direction = "FLAT"
        threshold = self.cfg.confidence_threshold

        if p_long >= threshold and p_long > p_short:
            raw_direction = "LONG"
            raw_confidence = p_long
        elif p_short >= threshold and p_short > p_long:
            raw_direction = "SHORT"
            raw_confidence = p_short
        else:
            raw_direction = "FLAT"
            raw_confidence = max(p_flat, p_long, p_short)

        # -------------------------------------------------------------
        # Multi-Model Consensus Agreement Gating
        # -------------------------------------------------------------
        consensus_met = True
        consensus_ratio = 1.0

        if self.cfg.use_ensemble and len(all_dirs) >= 3 and raw_direction != "FLAT":
            matching_votes = sum(1 for d in all_dirs if d == raw_direction)
            consensus_ratio = matching_votes / len(all_dirs)

            # Strict rule: At least 2 models must vote in the candidate direction,
            # and NO model should directly oppose with the inverse direction (regime conflict)
            inverse_direction = "SHORT" if raw_direction == "LONG" else "LONG"
            has_conflict = inverse_direction in all_dirs

            if self.cfg.consensus_required:
                if matching_votes < 2 or has_conflict:
                    consensus_met = False
                    raw_direction = "FLAT"
                    raw_confidence = p_flat

        threshold_met = (raw_direction in ("LONG", "SHORT")) and (raw_confidence >= threshold) and consensus_met
        top_feats = self.get_top_features(top_n=5)

        return SignalPrediction(
            symbol=self.symbol,
            direction=raw_direction,
            confidence=raw_confidence,
            probabilities=prob_dict,
            threshold_met=threshold_met,
            top_features=top_feats,
            timestamp=time.time(),
            ensemble_consensus=consensus_met,
            consensus_ratio=consensus_ratio,
            model_breakdown=model_breakdown
        )

    def predict_forward_trajectory(
        self,
        X_latest: pd.DataFrame,
        current_price: float,
        atr: float,
        last_timestamp_ms: int,
        timeframe_ms: int = 900000,
        steps: int = 8
    ) -> List[Dict[str, Any]]:
        """Compute future price trajectory and probabilistic confidence cone for charting."""
        pred = self.predict(X_latest)
        p_long = pred.probabilities.get("LONG", 0.33)
        p_short = pred.probabilities.get("SHORT", 0.33)

        # Net directional drift magnitude per bar scaled by confidence and consensus
        net_bias = (p_long - p_short) * pred.consensus_ratio
        drift_per_bar = net_bias * (0.35 * atr)

        trajectory: List[Dict[str, Any]] = []
        running_price = current_price

        for step in range(1, steps + 1):
            future_ts = last_timestamp_ms + (step * timeframe_ms)
            running_price += drift_per_bar

            # Volatility cone expands with sqrt(step)
            vol_spread = (np.sqrt(step) * 0.70) * atr

            trajectory.append({
                "step": step,
                "timestamp": int(future_ts),
                "predicted_price": round(float(running_price), 4),
                "upper_band": round(float(running_price + vol_spread), 4),
                "lower_band": round(float(running_price - vol_spread), 4),
                "direction": pred.direction,
                "confidence": round(pred.confidence * 100.0, 1),
                "consensus_ratio": round(pred.consensus_ratio, 2)
            })

        return trajectory

    def get_top_features(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract top feature importances blended across ensemble models."""
        if self.xgb_model is None:
            return []
        try:
            xgb_imp = self.xgb_model.feature_importances_
            if self.rf_model is not None:
                rf_imp = self.rf_model.feature_importances_
                blended = 0.60 * xgb_imp + 0.40 * rf_imp
            else:
                blended = xgb_imp

            pairs = list(zip(self.feature_names, [float(x) for x in blended]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        except Exception:
            return []

    def save(self) -> None:
        """Save trained models and feature metadata to disk."""
        if self.xgb_model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.xgb_model.save_model(str(self.model_path))

        if self.rf_model is not None:
            joblib.dump(self.rf_model, str(self.rf_path))
        if self.et_model is not None:
            joblib.dump(self.et_model, str(self.et_path))

        meta = {
            "symbol": self.symbol,
            "feature_names": self.feature_names,
            "eval_metrics": self.eval_metrics,
            "last_trained_time": self.last_trained_time,
            "train_bars_count": self.train_bars_count,
            "has_ensemble": bool(self.rf_model is not None and self.et_model is not None)
        }
        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self) -> bool:
        """Load ensemble models from disk if available."""
        if not self.model_path.exists() or not self.metadata_path.exists():
            return False
        try:
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.eval_metrics = meta.get("eval_metrics", {})
            self.last_trained_time = meta.get("last_trained_time", 0.0)

            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(self.model_path))
            self.model = self.xgb_model

            if self.rf_path.exists():
                self.rf_model = joblib.load(str(self.rf_path))
            if self.et_path.exists():
                self.et_model = joblib.load(str(self.et_path))

            return True
        except Exception as e:
            logger.error(f"Failed to load models from {self.model_path}: {e}")
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

    def predict_trajectory(
        self,
        symbol: str,
        X_latest: pd.DataFrame,
        current_price: float,
        atr: float,
        last_timestamp_ms: int,
        timeframe_ms: int = 900000,
        steps: int = 8
    ) -> List[Dict[str, Any]]:
        clf = self.get_classifier(symbol)
        return clf.predict_forward_trajectory(
            X_latest=X_latest,
            current_price=current_price,
            atr=atr,
            last_timestamp_ms=last_timestamp_ms,
            timeframe_ms=timeframe_ms,
            steps=steps
        )
