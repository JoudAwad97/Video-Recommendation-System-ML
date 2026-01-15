"""
Unit tests for Ranker model.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.models.ranker import RankerModel, RankerEvaluator
from src.models.model_config import RankerModelConfig


class TestRankerModel:
    """Tests for RankerModel."""

    @pytest.fixture
    def config(self):
        return RankerModelConfig(
            iterations=50,  # Small for fast tests
            learning_rate=0.1,
            depth=4,
            early_stopping_rounds=10,
            verbose=0,
        )

    @pytest.fixture
    def cat_features(self):
        return ["country", "device"]

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 200

        data = pd.DataFrame({
            "country": np.random.choice(["US", "UK", "DE", "FR"], n_samples),
            "device": np.random.choice(["Mobile", "Desktop"], n_samples),
            "age": np.random.randint(18, 65, n_samples),
            "video_duration": np.random.uniform(60, 3600, n_samples),
            "view_count": np.random.randint(100, 1000000, n_samples),
        })

        # Create target based on features
        data["label"] = (
            (data["age"] > 30).astype(int) +
            (data["video_duration"] > 1000).astype(int) +
            np.random.randint(0, 2, n_samples)
        ) > 1

        return data

    def test_model_initialization(self, config, cat_features):
        """Test model initialization."""
        model = RankerModel(config, cat_features)

        assert model.config == config
        assert model.cat_features == cat_features
        assert model.model is None

    def test_model_training(self, config, cat_features, sample_data):
        """Test model training."""
        model = RankerModel(config, cat_features)

        # Split data
        train_data = sample_data.iloc[:160]
        val_data = sample_data.iloc[160:]

        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"]
        X_val = val_data.drop(columns=["label"])
        y_val = val_data["label"]

        # Train
        history = model.fit(X_train, y_train, X_val, y_val)

        assert model.model is not None
        assert "train" in history
        assert "val" in history

    def test_model_prediction(self, config, cat_features, sample_data):
        """Test model prediction."""
        model = RankerModel(config, cat_features)

        train_data = sample_data.iloc[:160]
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"]

        model.fit(X_train, y_train)

        # Predict
        test_data = sample_data.iloc[160:]
        X_test = test_data.drop(columns=["label"])

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert all(p in [0, 1] or isinstance(p, (bool, np.bool_)) for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_feature_importance(self, config, cat_features, sample_data):
        """Test feature importance retrieval."""
        model = RankerModel(config, cat_features)

        train_data = sample_data.iloc[:160]
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"]

        model.fit(X_train, y_train)

        # Get feature importance
        importance = model.get_feature_importance()

        assert len(importance) == len(X_train.columns)
        assert all(v >= 0 for v in importance.values())

        # Top features
        top_importance = model.get_feature_importance(top_n=3)
        assert len(top_importance) == 3

    def test_model_save_load(self, config, cat_features, sample_data):
        """Test model save and load."""
        model = RankerModel(config, cat_features)

        train_data = sample_data.iloc[:160]
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"]

        model.fit(X_train, y_train)

        # Get predictions before saving
        test_data = sample_data.iloc[160:]
        X_test = test_data.drop(columns=["label"])
        preds_before = model.predict_proba(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save(tmpdir)

            # Load into new model
            model2 = RankerModel(config, cat_features)
            model2.load(tmpdir)

            # Get predictions after loading
            preds_after = model2.predict_proba(X_test)

            np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_rank_candidates(self, config, cat_features, sample_data):
        """Test ranking candidates for a user."""
        model = RankerModel(config, cat_features)

        train_data = sample_data.iloc[:160]
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"]

        model.fit(X_train, y_train)

        # Create user features
        user_features = pd.DataFrame({
            "country": ["US"],
            "device": ["Mobile"],
            "age": [25],
        })

        # Create candidate features
        candidates = pd.DataFrame({
            "video_duration": [100, 500, 1000, 2000, 3000],
            "view_count": [1000, 5000, 10000, 50000, 100000],
        })

        # Rank candidates
        ranked = model.rank_candidates(user_features, candidates)

        assert "ranker_score" in ranked.columns
        assert "rank" in ranked.columns
        assert list(ranked["rank"]) == [1, 2, 3, 4, 5]
        # Scores should be in descending order
        assert all(ranked["ranker_score"].iloc[i] >= ranked["ranker_score"].iloc[i+1]
                   for i in range(len(ranked)-1))


class TestRankerMetrics:
    """Tests for ranking metrics computation."""

    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions for metrics testing."""
        return {
            "y_true": np.array([
                [1, 0, 1, 0, 0],  # User 1: items 0 and 2 are relevant
                [0, 1, 0, 0, 1],  # User 2: items 1 and 4 are relevant
                [1, 1, 0, 0, 0],  # User 3: items 0 and 1 are relevant
            ]),
            "y_pred": np.array([
                [0.9, 0.5, 0.8, 0.2, 0.1],  # User 1 predictions
                [0.3, 0.9, 0.1, 0.2, 0.7],  # User 2 predictions
                [0.6, 0.9, 0.3, 0.4, 0.5],  # User 3 predictions
            ]),
        }

    def test_compute_mrr(self, sample_predictions):
        """Test MRR computation."""
        from src.models.metrics import compute_mrr

        y_true = sample_predictions["y_true"]
        y_pred = sample_predictions["y_pred"]

        mrr = compute_mrr(y_true, y_pred)

        # User 1: rank of first relevant = 1, RR = 1.0
        # User 2: rank of first relevant = 1, RR = 1.0
        # User 3: rank of first relevant = 1, RR = 1.0
        # MRR = (1 + 1 + 1) / 3 = 1.0
        assert mrr == pytest.approx(1.0, rel=0.01)

    def test_compute_ndcg(self, sample_predictions):
        """Test nDCG computation."""
        from src.models.metrics import compute_ndcg

        y_true = sample_predictions["y_true"]
        y_pred = sample_predictions["y_pred"]

        ndcg = compute_ndcg(y_true, y_pred, k=3)

        # nDCG should be between 0 and 1
        assert 0 <= ndcg <= 1

    def test_compute_map(self, sample_predictions):
        """Test mAP computation."""
        from src.models.metrics import compute_map

        y_true = sample_predictions["y_true"]
        y_pred = sample_predictions["y_pred"]

        map_score = compute_map(y_true, y_pred)

        # mAP should be between 0 and 1
        assert 0 <= map_score <= 1


class TestRankerEvaluator:
    """Tests for RankerEvaluator."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for evaluation."""
        config = RankerModelConfig(
            iterations=30,
            verbose=0,
        )
        model = RankerModel(config, cat_features=["category"])

        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "score": np.random.uniform(0, 1, n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        model.fit(X, y)
        return model

    def test_lift_analysis(self, trained_model):
        """Test lift analysis computation."""
        evaluator = RankerEvaluator(trained_model)

        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "score": np.random.uniform(0, 1, n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        lift_df = evaluator.compute_lift_analysis(X, y, n_bins=5)

        assert "bin" in lift_df.columns
        assert "lift" in lift_df.columns
        assert "conversion_rate" in lift_df.columns
        assert len(lift_df) <= 5
