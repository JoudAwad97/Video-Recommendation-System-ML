"""
Unit tests for data collection components.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.data_collection.collection_config import (
    DataCollectionConfig,
    LabelingRulesConfig,
    MergeJobConfig,
    InteractionType,
    LabelValue,
)
from src.data_collection.inference_tracker import (
    InferenceTracker,
    TrackedPrediction,
)
from src.data_collection.ground_truth_collector import (
    GroundTruthCollector,
    UserFeedback,
    LabeledInteraction,
)
from src.data_collection.merge_job import (
    PredictionLabelMerger,
    MergedRecord,
)
from src.data_collection.feedback_loop import (
    FeedbackLoopPipeline,
    RetrainingDataset,
)


class TestLabelingRulesConfig:
    """Tests for labeling rules configuration."""

    def test_get_label_for_like(self):
        """Test like interaction gives positive label."""
        config = LabelingRulesConfig()
        label = config.get_label_for_interaction(InteractionType.LIKE)
        assert label == LabelValue.POSITIVE

    def test_get_label_for_dislike(self):
        """Test dislike interaction gives negative label."""
        config = LabelingRulesConfig()
        label = config.get_label_for_interaction(InteractionType.DISLIKE)
        assert label == LabelValue.NEGATIVE

    def test_get_label_for_watch_high_percentage(self):
        """Test high watch percentage gives positive label."""
        config = LabelingRulesConfig(
            min_watch_percentage_positive=0.4,
            min_watch_seconds_positive=30,
        )
        label = config.get_label_for_interaction(
            InteractionType.WATCH,
            watch_percentage=0.5,
            watch_seconds=60,
        )
        assert label == LabelValue.POSITIVE

    def test_get_label_for_watch_low_percentage(self):
        """Test low watch percentage gives negative label."""
        config = LabelingRulesConfig(min_watch_percentage_negative=0.1)
        label = config.get_label_for_interaction(
            InteractionType.WATCH,
            watch_percentage=0.05,
            watch_seconds=10,
        )
        assert label == LabelValue.NEGATIVE

    def test_get_label_for_watch_medium_percentage(self):
        """Test medium watch percentage gives unknown label."""
        config = LabelingRulesConfig()
        label = config.get_label_for_interaction(
            InteractionType.WATCH,
            watch_percentage=0.25,  # Between 0.1 and 0.4
            watch_seconds=20,
        )
        assert label == LabelValue.UNKNOWN


class TestTrackedPrediction:
    """Tests for TrackedPrediction."""

    def test_create_prediction(self):
        """Test creating a tracked prediction."""
        pred = TrackedPrediction(
            inference_id="inf_123",
            request_id="req_456",
            user_id=1,
            recommended_video_ids=[100, 101, 102],
            video_scores={100: 0.9, 101: 0.8, 102: 0.7},
        )

        assert pred.inference_id == "inf_123"
        assert pred.user_id == 1
        assert len(pred.recommended_video_ids) == 3

    def test_to_json_and_back(self):
        """Test JSON serialization."""
        pred = TrackedPrediction(
            inference_id="inf_123",
            request_id="req_456",
            user_id=1,
            recommended_video_ids=[100, 101],
        )

        json_str = pred.to_json()
        restored = TrackedPrediction.from_json(json_str)

        assert restored.inference_id == pred.inference_id
        assert restored.user_id == pred.user_id


class TestInferenceTracker:
    """Tests for InferenceTracker."""

    @pytest.fixture
    def config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataCollectionConfig(local_storage_path=tmpdir)

    def test_track_inference(self, config):
        """Test tracking a new inference."""
        tracker = InferenceTracker(config)

        pred = tracker.track_inference(
            user_id=1,
            video_ids=[100, 101, 102],
            scores={100: 0.9, 101: 0.8, 102: 0.7},
            model_version="v1",
        )

        assert pred.inference_id is not None
        assert pred.user_id == 1
        assert len(pred.recommended_video_ids) == 3

    def test_get_prediction(self, config):
        """Test retrieving a prediction."""
        tracker = InferenceTracker(config)

        pred = tracker.track_inference(user_id=1, video_ids=[100])
        retrieved = tracker.get_prediction(pred.inference_id)

        assert retrieved is not None
        assert retrieved.inference_id == pred.inference_id

    def test_get_predictions_for_user(self, config):
        """Test getting predictions for a user."""
        tracker = InferenceTracker(config)

        tracker.track_inference(user_id=1, video_ids=[100])
        tracker.track_inference(user_id=1, video_ids=[101])
        tracker.track_inference(user_id=2, video_ids=[102])

        user_preds = tracker.get_predictions_for_user(user_id=1)
        assert len(user_preds) == 2

    def test_find_inference_for_feedback(self, config):
        """Test finding inference for feedback."""
        tracker = InferenceTracker(config)

        pred = tracker.track_inference(user_id=1, video_ids=[100, 101, 102])

        found = tracker.find_inference_for_feedback(
            user_id=1,
            video_id=101,
        )

        assert found is not None
        assert found.inference_id == pred.inference_id

    def test_find_inference_wrong_video(self, config):
        """Test not finding inference for wrong video."""
        tracker = InferenceTracker(config)

        tracker.track_inference(user_id=1, video_ids=[100, 101])

        found = tracker.find_inference_for_feedback(
            user_id=1,
            video_id=999,  # Not in recommendations
        )

        assert found is None


class TestGroundTruthCollector:
    """Tests for GroundTruthCollector."""

    @pytest.fixture
    def config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataCollectionConfig(local_storage_path=tmpdir)

    @pytest.fixture
    def tracker(self, config):
        return InferenceTracker(config)

    def test_record_feedback(self, config, tracker):
        """Test recording feedback."""
        collector = GroundTruthCollector(config, tracker)

        feedback = collector.record_feedback(
            user_id=1,
            video_id=100,
            interaction_type="click",
        )

        assert feedback.feedback_id is not None
        assert feedback.user_id == 1
        assert feedback.video_id == 100

    def test_record_feedback_with_inference_linking(self, config, tracker):
        """Test feedback links to inference."""
        collector = GroundTruthCollector(config, tracker)

        # Create prediction
        pred = tracker.track_inference(user_id=1, video_ids=[100, 101])

        # Record feedback without explicit inference_id
        feedback = collector.record_feedback(
            user_id=1,
            video_id=100,
            interaction_type="click",
        )

        # Should auto-link
        assert feedback.inference_id == pred.inference_id

    def test_process_feedback_to_labels(self, config, tracker):
        """Test converting feedback to labels."""
        collector = GroundTruthCollector(config, tracker)

        # Create prediction and feedback
        pred = tracker.track_inference(user_id=1, video_ids=[100])
        collector.record_feedback(
            user_id=1,
            video_id=100,
            interaction_type="like",
            inference_id=pred.inference_id,
        )

        labels = collector.process_feedback_to_labels()

        assert len(labels) == 1
        assert labels[0].label == 1  # Positive
        assert labels[0].label_source == "like"

    def test_watch_labeling(self, config, tracker):
        """Test watch time based labeling."""
        collector = GroundTruthCollector(config, tracker)

        pred = tracker.track_inference(user_id=1, video_ids=[100])

        # High watch time -> positive
        collector.record_feedback(
            user_id=1,
            video_id=100,
            interaction_type="watch",
            inference_id=pred.inference_id,
            watch_time_seconds=120,
            video_duration_seconds=200,  # 60%
        )

        labels = collector.process_feedback_to_labels()
        assert labels[0].label == 1


class TestPredictionLabelMerger:
    """Tests for PredictionLabelMerger."""

    @pytest.fixture
    def config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MergeJobConfig(merged_output_path=tmpdir)

    @pytest.fixture
    def sample_predictions(self):
        return [
            TrackedPrediction(
                inference_id=f"inf_{i}",
                request_id=f"req_{i}",
                user_id=i,
                recommended_video_ids=[100, 101, 102],
                video_scores={100: 0.9, 101: 0.8, 102: 0.7},
                model_version="v1",
            )
            for i in range(10)
        ]

    @pytest.fixture
    def sample_labels(self, sample_predictions):
        labels = []
        for i, pred in enumerate(sample_predictions[:5]):
            label = LabeledInteraction(
                interaction_id=f"labeled_{i}",
                inference_id=pred.inference_id,
                user_id=pred.user_id,
                video_id=100,
                label=1 if i % 2 == 0 else 0,
                label_source="click" if i % 2 == 0 else "skip",
                feedback_timestamp=pred.timestamp,
            )
            labels.append(label)
        return labels

    def test_merge(self, config, sample_predictions, sample_labels):
        """Test merging predictions with labels."""
        merger = PredictionLabelMerger(config)

        merged, result = merger.merge(sample_predictions, sample_labels)

        assert result.records_merged == 5
        assert result.positive_count == 3
        assert result.negative_count == 2

    def test_to_dataframe(self, config, sample_predictions, sample_labels):
        """Test converting to DataFrame."""
        merger = PredictionLabelMerger(config)

        merged, _ = merger.merge(sample_predictions, sample_labels)
        df = merger.to_dataframe(merged)

        assert len(df) == 5
        assert "user_id" in df.columns
        assert "label" in df.columns
        assert "predicted_score" in df.columns

    def test_save_and_load(self, config, sample_predictions, sample_labels):
        """Test saving merged records."""
        merger = PredictionLabelMerger(config)

        merged, _ = merger.merge(sample_predictions, sample_labels)
        output_path = merger.save_merged_records(merged, format="jsonl")

        assert Path(output_path).exists()


class TestFeedbackLoopPipeline:
    """Tests for FeedbackLoopPipeline."""

    @pytest.fixture
    def config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataCollectionConfig(
                local_storage_path=tmpdir,
                retraining_data_path=f"{tmpdir}/retraining",
                min_samples_for_retraining=5,  # Low for testing
            )

    def test_full_pipeline(self, config):
        """Test the full feedback loop pipeline."""
        pipeline = FeedbackLoopPipeline(config)

        # Track inferences
        for i in range(10):
            pred = pipeline.track_inference(
                user_id=i,
                video_ids=[100, 101, 102],
                scores={100: 0.9, 101: 0.8, 102: 0.7},
                model_version="v1",
            )

            # Record feedback for each
            pipeline.record_feedback(
                user_id=i,
                video_id=100,
                interaction_type="like" if i % 2 == 0 else "dislike",
                inference_id=pred.inference_id,
            )

        # Process labels
        labels = pipeline.process_labels()
        assert len(labels) == 10

        # Check stats
        stats = pipeline.get_stats()
        assert stats["inference_tracker"]["total_predictions"] == 10
        assert stats["ground_truth_collector"]["total_labeled"] == 10

    def test_generate_retraining_dataset(self, config):
        """Test generating retraining dataset."""
        pipeline = FeedbackLoopPipeline(config)

        # Create enough data
        for i in range(20):
            pred = pipeline.track_inference(
                user_id=i,
                video_ids=[100, 101],
                scores={100: 0.9, 101: 0.8},
            )

            pipeline.record_feedback(
                user_id=i,
                video_id=100,
                interaction_type="like" if i % 2 == 0 else "dislike",
                inference_id=pred.inference_id,
            )

        # Generate dataset
        dataset = pipeline.generate_retraining_dataset()

        assert dataset is not None
        assert dataset.total_samples == 20
        assert dataset.positive_samples == 10
        assert dataset.negative_samples == 10
        assert Path(dataset.train_path).exists()

    def test_save_and_load_state(self, config):
        """Test saving and loading pipeline state."""
        pipeline = FeedbackLoopPipeline(config)

        # Add some data
        for i in range(5):
            pred = pipeline.track_inference(user_id=i, video_ids=[100])
            pipeline.record_feedback(
                user_id=i,
                video_id=100,
                interaction_type="click",
                inference_id=pred.inference_id,
            )

        # Save state
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save_state(tmpdir)

            # Create new pipeline and load
            pipeline2 = FeedbackLoopPipeline(config)
            pipeline2.load_state(tmpdir)

            # Verify data loaded
            stats = pipeline2.get_stats()
            assert stats["inference_tracker"]["total_predictions"] == 5


class TestDataCollectionConfig:
    """Tests for DataCollectionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DataCollectionConfig()

        assert config.inference_ttl_hours == 48
        assert config.min_samples_for_retraining == 10000
        assert config.labeling_rules is not None

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataCollectionConfig(
            inference_ttl_hours=24,
            batch_size=500,
            positive_negative_ratio=2.0,
        )

        assert config.inference_ttl_hours == 24
        assert config.batch_size == 500
        assert config.positive_negative_ratio == 2.0
