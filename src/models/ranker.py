"""
Ranker Model using CatBoost for video recommendations.

The Ranker model refines recommendations by scoring candidate videos
that were retrieved by the Two-Tower model. It uses CatBoostClassifier
which handles categorical features natively.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from catboost import CatBoostClassifier, Pool

from src.models.model_config import RankerModelConfig, DEFAULT_RANKER_CONFIG
from src.models.metrics import compute_mrr, compute_ndcg, compute_map


class RankerModel:
    """
    CatBoost-based Ranker model for video recommendations.

    Takes combined user-video features and predicts the probability
    of user interaction (click, watch, like, etc.).
    """

    def __init__(
        self,
        config: RankerModelConfig = None,
        cat_features: List[str] = None,
    ):
        """
        Initialize the Ranker model.

        Args:
            config: Model configuration
            cat_features: List of categorical feature names
        """
        self.config = config or DEFAULT_RANKER_CONFIG
        self.cat_features = cat_features or self.config.cat_features
        self.model = None
        self.feature_importances_ = None
        self.training_history = {}

    def _create_model(self) -> CatBoostClassifier:
        """Create a CatBoost classifier with configured parameters."""
        return CatBoostClassifier(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            l2_leaf_reg=self.config.l2_leaf_reg,
            loss_function=self.config.loss_function,
            eval_metric=self.config.eval_metric,
            random_seed=self.config.random_seed,
            thread_count=self.config.thread_count,
            verbose=self.config.verbose,
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

    def _prepare_pool(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Pool:
        """
        Prepare CatBoost Pool from DataFrame.

        Args:
            X: Feature DataFrame
            y: Optional target Series

        Returns:
            CatBoost Pool object
        """
        # Find categorical feature indices
        cat_feature_indices = [
            i for i, col in enumerate(X.columns)
            if col in self.cat_features
        ]

        return Pool(
            data=X,
            label=y,
            cat_features=cat_feature_indices,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train the Ranker model.

        Args:
            X_train: Training features
            y_train: Training labels (0/1)
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Dictionary with training history and metrics
        """
        # Create model
        self.model = self._create_model()

        # Prepare pools
        train_pool = self._prepare_pool(X_train, y_train)
        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = self._prepare_pool(X_val, y_val)

        # Train model
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True if eval_pool else False,
        )

        # Store feature importances
        self.feature_importances_ = dict(zip(
            X_train.columns,
            self.model.get_feature_importance()
        ))

        # Compute training metrics
        train_metrics = self._compute_metrics(X_train, y_train, "train")

        # Compute validation metrics if available
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._compute_metrics(X_val, y_val, "val")

        self.training_history = {
            "train": train_metrics,
            "val": val_metrics,
            "best_iteration": self.model.get_best_iteration() if eval_pool else self.config.iterations,
        }

        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        probas = self.model.predict_proba(X)
        return probas[:, 1] if probas.ndim > 1 else probas

    def rank_candidates(
        self,
        user_features: pd.DataFrame,
        candidate_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank candidate videos for a user.

        Args:
            user_features: User feature DataFrame (1 row)
            candidate_features: Candidate video features DataFrame (n rows)

        Returns:
            Ranked DataFrame with scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Expand user features to match candidates
        user_expanded = pd.concat([user_features] * len(candidate_features), ignore_index=True)

        # Combine features
        combined = pd.concat([
            user_expanded.reset_index(drop=True),
            candidate_features.reset_index(drop=True)
        ], axis=1)

        # Get scores
        scores = self.predict_proba(combined)

        # Add scores and rank
        result = candidate_features.copy()
        result["ranker_score"] = scores
        result = result.sort_values("ranker_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _compute_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            X: Feature DataFrame
            y: True labels
            prefix: Metric name prefix

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Basic classification metrics
        accuracy = np.mean(y_pred == y.values)
        auc = self._compute_auc(y.values, y_pred_proba)

        metrics = {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_auc": auc,
        }

        return metrics

    def _compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC-ROC score."""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.0

    def evaluate_ranking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        user_ids: pd.Series,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality metrics per user.

        Args:
            X: Feature DataFrame
            y: True labels
            user_ids: User ID series
            k_values: List of k values for metrics

        Returns:
            Dictionary of ranking metrics
        """
        # Get predictions
        y_pred_proba = self.predict_proba(X)

        # Group by user
        unique_users = user_ids.unique()
        metrics = {}

        mrr_scores = []
        ndcg_scores = {k: [] for k in k_values}
        map_scores = {k: [] for k in k_values}

        for user_id in unique_users:
            mask = user_ids == user_id
            user_true = y.values[mask]
            user_pred = y_pred_proba[mask]

            if len(user_true) < 2:
                continue

            # Reshape for metrics computation
            user_true_2d = user_true.reshape(1, -1)
            user_pred_2d = user_pred.reshape(1, -1)

            # MRR
            mrr_scores.append(compute_mrr(user_true_2d, user_pred_2d))

            # nDCG and mAP at different k values
            for k in k_values:
                if len(user_true) >= k:
                    ndcg_scores[k].append(compute_ndcg(user_true_2d, user_pred_2d, k))
                    map_scores[k].append(compute_map(user_true_2d, user_pred_2d, k))

        # Aggregate metrics
        metrics["mrr"] = np.mean(mrr_scores) if mrr_scores else 0.0

        for k in k_values:
            if ndcg_scores[k]:
                metrics[f"ndcg@{k}"] = np.mean(ndcg_scores[k])
            if map_scores[k]:
                metrics[f"map@{k}"] = np.mean(map_scores[k])

        return metrics

    def get_feature_importance(
        self,
        top_n: int = None
    ) -> Dict[str, float]:
        """
        Get feature importances.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary of feature importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        sorted_importance = dict(sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        if top_n:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])

        return sorted_importance

    def save(self, save_dir: str):
        """
        Save the model and configuration.

        Args:
            save_dir: Directory to save model files
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save CatBoost model
        model_path = os.path.join(save_dir, "ranker_model.cbm")
        self.model.save_model(model_path)

        # Save configuration
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "config": self.config.__dict__,
                "cat_features": self.cat_features,
                "feature_importances": self.feature_importances_,
                "training_history": self.training_history,
            }, f, indent=2)

        print(f"Model saved to {save_dir}")

    def load(self, save_dir: str):
        """
        Load the model and configuration.

        Args:
            save_dir: Directory containing saved model files
        """
        # Load CatBoost model
        model_path = os.path.join(save_dir, "ranker_model.cbm")
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        # Load configuration
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "r") as f:
            saved_data = json.load(f)

        self.cat_features = saved_data["cat_features"]
        self.feature_importances_ = saved_data.get("feature_importances")
        self.training_history = saved_data.get("training_history", {})

        print(f"Model loaded from {save_dir}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "config": self.config.__dict__,
            "cat_features": self.cat_features,
        }


class RankerEvaluator:
    """
    Evaluator class for ranking model evaluation.

    Provides utilities for computing ranking metrics on grouped data.
    """

    def __init__(self, model: RankerModel):
        self.model = model

    def evaluate_per_user(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        user_id_col: str = "user_id",
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate ranking metrics per user.

        Args:
            X: Feature DataFrame (must include user_id column)
            y: True labels
            user_id_col: Name of user ID column
            k_values: List of k values for metrics

        Returns:
            Dictionary of average metrics
        """
        if user_id_col not in X.columns:
            raise ValueError(f"Column {user_id_col} not found in DataFrame")

        return self.model.evaluate_ranking(
            X.drop(columns=[user_id_col]),
            y,
            X[user_id_col],
            k_values
        )

    def compute_lift_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Compute lift analysis for model scores.

        Args:
            X: Feature DataFrame
            y: True labels
            n_bins: Number of bins for analysis

        Returns:
            DataFrame with lift analysis
        """
        scores = self.model.predict_proba(X)

        df = pd.DataFrame({
            "score": scores,
            "label": y.values
        })

        # Create decile bins
        df["bin"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop")

        # Aggregate by bin
        lift_df = df.groupby("bin").agg({
            "score": ["mean", "min", "max"],
            "label": ["sum", "count", "mean"]
        }).reset_index()

        lift_df.columns = [
            "bin", "score_mean", "score_min", "score_max",
            "positive_count", "total_count", "conversion_rate"
        ]

        # Calculate lift
        baseline_rate = y.mean()
        lift_df["lift"] = lift_df["conversion_rate"] / baseline_rate

        return lift_df
