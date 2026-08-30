"""Dual XGBoost model trainer with chronological validation split and Optuna tuning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from src.config import AppConfig

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

OBJECTIVE_BINARY_LOGISTIC = "binary:logistic"


class DualXGBoostTrainer:
    """Trains independent BUY and SELL XGBoost models with chronological validation and Optuna tuning."""

    def __init__(self, config: AppConfig):
        self.config = config

    def _detect_accelerator(self) -> Tuple[str, str]:
        """Detect if GPU acceleration is available for XGBoost."""
        try:
            test_clf = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
            x_test = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            y_test = np.array([0, 1, 0, 1], dtype=np.int32)
            test_clf.fit(x_test, y_test)
            print("    [+] GPU Acceleration (CUDA) detected and enabled.")
            return "hist", "cuda"
        except Exception:
            print("    [-] GPU not available or CUDA not supported. Using CPU hist.")
            return "hist", "cpu"

    def train(self, csv_path: Path, direction: str) -> Tuple[xgb.XGBClassifier, Dict[str, Any], List[str]]:
        """Train an optimized XGBoost model for BUY or SELL direction."""
        dir_upper = direction.upper()
        start_time = datetime.now()
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{start_str}] [*] Optimizing and Training XGBoost {dir_upper} Model on '{csv_path.name}'...")
        print(f"    [*] Training started at: [{start_str}]")

        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        if "label" not in df.columns:
            raise ValueError(f"Dataset '{csv_path.name}' missing 'label' column.")

        feature_names = [col for col in df.columns if col != "label"]
        total_samples = len(df)

        if total_samples < 10:
            raise ValueError(f"Dataset '{csv_path.name}' has too few samples ({total_samples}) for training.")

        # Chronological Time-Series Split (Zero lookahead / data leakage)
        val_size = int(total_samples * self.config.validation_percentage)
        if val_size < 10:
            val_size = max(5, int(total_samples * 0.2))
        train_size = total_samples - val_size

        x_data = df.drop(columns=["label"]).astype(np.float32)
        y_data = df["label"].astype(np.int32)

        x_train, x_val = x_data.iloc[:train_size], x_data.iloc[train_size:]
        y_train, y_val = y_data.iloc[:train_size], y_data.iloc[train_size:]

        total_positives = int((y_data == 1).sum())
        total_negatives = int((y_data == 0).sum())
        pos_pct = (total_positives / total_samples) * 100.0 if total_samples > 0 else 0.0
        neg_pct = (total_negatives / total_samples) * 100.0 if total_samples > 0 else 0.0

        train_positives = int((y_train == 1).sum())
        train_negatives = int((y_train == 0).sum())
        val_positives = int((y_val == 1).sum())
        val_negatives = int((y_val == 0).sum())

        print(f"\n    [{dir_upper}] Training Dataset Directional Class Balance:")
        print(
            f"        Total: {total_samples:,} samples | "
            f"Directional (y=1): {total_positives:,} ({pos_pct:.2f}%) | "
            f"Neutral/Adverse (y=0): {total_negatives:,} ({neg_pct:.2f}%)"
        )
        print(
            f"        Train: {len(x_train):,} (y=1: {train_positives:,}, y=0: {train_negatives:,}) | "
            f"Val: {len(x_val):,} (y=1: {val_positives:,}, y=0: {val_negatives:,})"
        )
        print(f"        Feature Vector Dimensions: {x_data.shape[1]}")

        tree_method, device = self._detect_accelerator()

        # Optuna Bayesian Hyperparameter Optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        optuna_metric = self.config.optuna_objective_metric.lower()

        def objective(trial: optuna.Trial) -> float:
            try:
                min_depth = max(2, self.config.xgb_max_depth - 1)
                max_depth = min(6, self.config.xgb_max_depth + 2)
                min_eta = max(0.001, self.config.xgb_eta * 0.2)
                max_eta = min(0.05, max(0.01, self.config.xgb_eta * 1.5))
                min_sub = max(0.4, self.config.xgb_subsample - 0.3)
                max_sub = min(1.0, self.config.xgb_subsample + 0.2)
                min_col = max(0.4, self.config.xgb_colsample_bytree - 0.3)
                max_col = min(1.0, self.config.xgb_colsample_bytree + 0.3)
                min_child = max(1.0, self.config.xgb_min_child_weight * 0.5)
                max_child = max(10.0, self.config.xgb_min_child_weight * 2.0)
                min_lam = max(0.01, self.config.xgb_lambda * 0.2)
                max_lam = max(10.0, self.config.xgb_lambda * 5.0)
                max_alp = max(5.0, self.config.xgb_alpha * 5.0)
                min_est = max(20, self.config.xgb_rounds // 4)
                max_est = max(60, self.config.xgb_rounds)

                params = {
                    "tree_method": tree_method,
                    "device": device if not trial.study.user_attrs.get("is_remote") else "cpu",
                    "max_depth": trial.suggest_int("max_depth", min_depth, max_depth),
                    "learning_rate": trial.suggest_float("learning_rate", min_eta, max_eta, log=True),
                    "subsample": trial.suggest_float("subsample", min_sub, max_sub),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", min_col, max_col),
                    "min_child_weight": trial.suggest_float("min_child_weight", min_child, max_child),
                    "reg_lambda": trial.suggest_float("reg_lambda", min_lam, max_lam),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.05, max_alp),
                    "n_estimators": trial.suggest_int("n_estimators", min_est, max_est),
                    "early_stopping_rounds": self.config.xgb_early_stopping_rounds,
                    "objective": OBJECTIVE_BINARY_LOGISTIC,
                    "eval_metric": "logloss",
                    "random_state": 42,
                }
                model = xgb.XGBClassifier(**params)
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                preds = model.predict_proba(x_val)[:, 1]

                if optuna_metric == "logloss":
                    return float(log_loss(y_val, preds, labels=[0, 1]))
                elif optuna_metric == "roc_auc":
                    try:
                        return float(1.0 - roc_auc_score(y_val, preds))
                    except ValueError:
                        return 1.0
                elif optuna_metric == "precision":
                    pred_labels = (preds >= self.config.eval_classification_threshold).astype(int)
                    return float(1.0 - precision_score(y_val, pred_labels, zero_division=0))
                elif optuna_metric == "f1":
                    pred_labels = (preds >= self.config.eval_classification_threshold).astype(int)
                    return float(1.0 - f1_score(y_val, pred_labels, zero_division=0))
                else:
                    return float(log_loss(y_val, preds, labels=[0, 1]))
            except Exception as exc:
                print(f"    [!] Trial {trial.number} failed with error: {exc}")
                return float("inf")

        study = optuna.create_study(direction="minimize")
        optuna_jobs = 1 if device == "cuda" else -1
        print(
            f"    [*] Starting Hyperparameter Optimization "
            f"({self.config.optuna_trials} trials, Objective: {optuna_metric.upper()})..."
        )
        study.optimize(objective, n_trials=self.config.optuna_trials, n_jobs=optuna_jobs)

        print(f"    [+] Best Params for {dir_upper}: {study.best_params}")

        # Train Final Model with Best Hyperparameters and Early Stopping
        final_params = dict(study.best_params)
        final_params["tree_method"] = tree_method
        final_params["device"] = device
        final_params["objective"] = OBJECTIVE_BINARY_LOGISTIC
        final_params["eval_metric"] = "logloss"
        final_params["early_stopping_rounds"] = self.config.xgb_early_stopping_rounds
        final_params["random_state"] = 42

        clf = xgb.XGBClassifier(**final_params)
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        val_preds_prob = clf.predict_proba(x_val)[:, 1]

        # Calculate Validation Metrics at User-Configured Decision Threshold
        target_thresh = self.config.eval_classification_threshold
        val_pred_labels = (val_preds_prob >= target_thresh).astype(int)

        try:
            auc = round(float(roc_auc_score(y_val, val_preds_prob)), 4)
        except ValueError:
            auc = 0.5

        loss = round(float(log_loss(y_val, val_preds_prob, labels=[0, 1])), 4)
        acc = round(float(accuracy_score(y_val, val_pred_labels)), 4)
        prec = round(float(precision_score(y_val, val_pred_labels, zero_division=0)), 4)
        rec = round(float(recall_score(y_val, val_pred_labels, zero_division=0)), 4)
        f1 = round(float(f1_score(y_val, val_pred_labels, zero_division=0)), 4)
        signals_count = int(val_pred_labels.sum())
        signals_pct = round((signals_count / len(y_val)) * 100.0, 2) if len(y_val) > 0 else 0.0

        metrics: Dict[str, Any] = {
            "direction": dir_upper,
            "total_samples": total_samples,
            "train_samples": len(x_train),
            "val_samples": len(x_val),
            "roc_auc": auc,
            "accuracy": acc,
            "log_loss": loss,
            "eval_threshold": target_thresh,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "signals_count": signals_count,
            "signals_pct": signals_pct,
            "best_iteration": int(getattr(clf, "best_iteration", len(clf.get_booster().get_dump()))),
        }

        print(f"\n    [{dir_upper}] VALIDATION METRICS (Target Threshold θ = {target_thresh:.2f}):")
        print(
            f"        ROC-AUC: {auc:.4f} | LogLoss: {loss:.4f} | Accuracy: {acc * 100:.2f}%\n"
            f"        Directional Precision (Win Rate): {prec * 100:.2f}% | "
            f"Momentum Recall: {rec * 100:.2f}% | F1-Score: {f1:.4f}\n"
            f"        Active Signals: {signals_count:,} / {len(y_val):,} bars ({signals_pct:.2f}% market participation)"
        )

        # Parametric Directional Sensitivity Grid (if enabled by user)
        if self.config.eval_enable_threshold_grid:
            step_span = self.config.eval_threshold_max - self.config.eval_threshold_min
            steps = int(round(step_span / self.config.eval_threshold_step)) + 1
            threshold_levels = [
                round(self.config.eval_threshold_min + i * self.config.eval_threshold_step, 4)
                for i in range(steps)
            ]

            grid_records = []
            print(f"\n    [{dir_upper}] PARAMETRIC DIRECTIONAL SENSITIVITY GRID (Validation Set: {len(y_val):,} bars):")
            print("        " + "-" * 90)
            grid_hdr = (
                f"        {'Threshold (θ)':<14} | {'Signals (Bars)':<15} | {'Frequency (%)':<15} | "
                f"{'Precision':<12} | {'Recall':<10} | {'F1-Score':<10}"
            )
            print(grid_hdr)
            print("        " + "-" * 90)

            for t_val in threshold_levels:
                t_preds = (val_preds_prob >= t_val).astype(int)
                t_signals = int(t_preds.sum())
                t_freq = (t_signals / len(y_val)) * 100.0 if len(y_val) > 0 else 0.0
                t_prec = float(precision_score(y_val, t_preds, zero_division=0)) * 100.0
                t_rec = float(recall_score(y_val, t_preds, zero_division=0)) * 100.0
                t_f1 = float(f1_score(y_val, t_preds, zero_division=0))
                grid_records.append({
                    "threshold": t_val,
                    "signals": t_signals,
                    "frequency_pct": round(t_freq, 2),
                    "precision_pct": round(t_prec, 2),
                    "recall_pct": round(t_rec, 2),
                    "f1": round(t_f1, 4),
                })
                print(
                    f"        {t_val:<14.2f} | {t_signals:<15,d} | {t_freq:<14.2f}% | "
                    f"{t_prec:<11.2f}% | {t_rec:<9.2f}% | {t_f1:<10.4f}"
                )
            print("        " + "-" * 90)
            metrics["threshold_grid"] = grid_records

        end_time = datetime.now()
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        total_seconds = int((end_time - start_time).total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        print(f"\n    [*] Training completed at: [{end_str}] (Elapsed: {elapsed_str})")
        return clf, metrics, feature_names
