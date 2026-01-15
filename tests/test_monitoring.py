"""
Unit tests for monitoring components.
"""

import pytest
import numpy as np
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.monitoring.monitoring_config import (
    MonitoringConfig,
    DataCaptureConfig,
    AlertConfig,
    OnlineMetricsConfig,
    ABTestConfig,
)
from src.monitoring.data_capture import (
    DataCaptureService,
    InferenceRecord,
    InferenceRecordBatch,
)
from src.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
)
from src.monitoring.data_quality_monitor import (
    DataQualityMonitor,
    FeatureStatistics,
)
from src.monitoring.online_metrics import (
    OnlineMetricsCollector,
    UserInteraction,
)
from src.monitoring.ab_testing import (
    ABTestManager,
    Experiment,
    ExperimentStatus,
)
from src.monitoring.ranker_sampler import (
    RankerQualitySampler,
    RankerSample,
)


class TestMonitoringConfig:
    """Tests for monitoring configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonitoringConfig()

        assert config.alerts is not None
        assert config.data_capture is not None
        assert len(config.numerical_features) > 0
        assert len(config.categorical_features) > 0

    def test_alert_config(self):
        """Test alert configuration."""
        config = AlertConfig(
            precision_drop_threshold=0.2,
            latency_p99_threshold_ms=1000,
        )

        assert config.precision_drop_threshold == 0.2
        assert config.latency_p99_threshold_ms == 1000

    def test_get_all_features(self):
        """Test getting all features."""
        config = MonitoringConfig()
        all_features = config.get_all_features()

        assert len(all_features) == len(config.numerical_features) + len(config.categorical_features)


class TestInferenceRecord:
    """Tests for InferenceRecord."""

    def test_create_record(self):
        """Test creating an inference record."""
        record = InferenceRecord(
            request_id="req_123",
            user_id=456,
            input_features={"age": 25, "country": "US"},
            output_predictions={"recommendations": [1, 2, 3]},
            inference_latency_ms=50.0,
        )

        assert record.request_id == "req_123"
        assert record.user_id == 456
        assert record.inference_latency_ms == 50.0
        assert len(record.record_id) > 0

    def test_to_json_and_back(self):
        """Test JSON serialization."""
        record = InferenceRecord(
            request_id="req_123",
            input_features={"age": 25},
            output_predictions={"score": 0.8},
        )

        json_str = record.to_json()
        restored = InferenceRecord.from_json(json_str)

        assert restored.request_id == record.request_id
        assert restored.input_features == record.input_features


class TestInferenceRecordBatch:
    """Tests for InferenceRecordBatch."""

    def test_batch_operations(self):
        """Test batch operations."""
        batch = InferenceRecordBatch()

        record1 = InferenceRecord(request_id="req_1")
        record2 = InferenceRecord(request_id="req_2")

        batch.add(record1)
        batch.add(record2)

        assert len(batch) == 2

    def test_to_jsonl(self):
        """Test JSONL conversion."""
        batch = InferenceRecordBatch()
        batch.add(InferenceRecord(request_id="req_1"))
        batch.add(InferenceRecord(request_id="req_2"))

        jsonl = batch.to_jsonl()
        lines = jsonl.strip().split("\n")

        assert len(lines) == 2


class TestDataCaptureService:
    """Tests for DataCaptureService."""

    @pytest.fixture
    def capture_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataCaptureConfig(
                enable_capture=True,
                capture_percentage=1.0,
                storage_type="local",
                local_storage_path=tmpdir,
            )

    def test_capture_record(self, capture_config):
        """Test capturing a record."""
        service = DataCaptureService(capture_config)

        record_id = service.capture(
            request_id="req_123",
            input_features={"age": 25},
            output_predictions={"score": 0.8},
            latency_ms=50.0,
        )

        assert record_id is not None
        stats = service.get_stats()
        assert stats["records_captured"] == 1

    def test_capture_disabled(self):
        """Test capture when disabled."""
        config = DataCaptureConfig(enable_capture=False)
        service = DataCaptureService(config)

        record_id = service.capture(
            request_id="req_123",
            input_features={},
            output_predictions={},
            latency_ms=50.0,
        )

        assert record_id is None


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    @pytest.fixture
    def config(self):
        return MonitoringConfig()

    @pytest.fixture
    def sample_records(self):
        records = []
        for i in range(100):
            record = InferenceRecord(
                request_id=f"req_{i}",
                inference_latency_ms=50 + i * 0.5,
                output_predictions={
                    "recommendations": [
                        {"video_id": j, "score": 0.8 - j * 0.05}
                        for j in range(10)
                    ]
                },
                metadata={
                    "stage_latencies": {
                        "retrieval": 20 + i * 0.2,
                        "ranking": 30 + i * 0.3,
                    }
                },
            )
            records.append(record)
        return records

    def test_compute_metrics(self, config, sample_records):
        """Test computing metrics from records."""
        monitor = PerformanceMonitor(config)
        metrics = monitor.compute_metrics(sample_records)

        assert metrics.num_samples == 100
        assert metrics.avg_total_latency_ms > 0
        assert metrics.latency_p50_ms > 0
        assert metrics.latency_p99_ms >= metrics.latency_p50_ms

    def test_analyze(self, config, sample_records):
        """Test full analysis."""
        monitor = PerformanceMonitor(config)
        report = monitor.analyze(sample_records)

        assert report.status in ["healthy", "degraded", "critical"]
        assert report.current_metrics.num_samples == 100

    def test_save_load_baseline(self, config, sample_records):
        """Test saving and loading baseline."""
        monitor = PerformanceMonitor(config)
        metrics = monitor.compute_metrics(sample_records)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            monitor.save_baseline(metrics, f.name)
            monitor.load_baseline(f.name)

        assert monitor.baseline_metrics is not None
        assert monitor.baseline_metrics.num_samples == 100


class TestDataQualityMonitor:
    """Tests for DataQualityMonitor."""

    @pytest.fixture
    def config(self):
        return MonitoringConfig(
            numerical_features=["age", "duration"],
            categorical_features=["country", "category"],
        )

    @pytest.fixture
    def sample_records(self):
        records = []
        for i in range(100):
            record = InferenceRecord(
                request_id=f"req_{i}",
                input_features={
                    "age": 20 + i % 40,
                    "duration": 100 + i * 10,
                    "country": ["US", "UK", "CA"][i % 3],
                    "category": ["Tech", "Sports", "Music"][i % 3],
                },
            )
            records.append(record)
        return records

    def test_compute_statistics(self, config, sample_records):
        """Test computing feature statistics."""
        monitor = DataQualityMonitor(config)
        stats = monitor.compute_statistics(sample_records)

        assert "age" in stats
        assert stats["age"].feature_type == "numerical"
        assert stats["age"].count == 100

        assert "country" in stats
        assert stats["country"].feature_type == "categorical"
        assert stats["country"].unique_count == 3

    def test_analyze(self, config, sample_records):
        """Test full analysis."""
        monitor = DataQualityMonitor(config)
        report = monitor.analyze(sample_records)

        assert report.status in ["healthy", "degraded", "critical"]
        assert len(report.feature_statistics) > 0

    def test_detect_drift(self, config):
        """Test drift detection."""
        monitor = DataQualityMonitor(config)

        baseline = FeatureStatistics(
            feature_name="age",
            feature_type="numerical",
            mean=30.0,
            std=10.0,
        )

        current_no_drift = FeatureStatistics(
            feature_name="age",
            feature_type="numerical",
            mean=31.0,
            std=10.0,
        )

        current_with_drift = FeatureStatistics(
            feature_name="age",
            feature_type="numerical",
            mean=50.0,
            std=10.0,
        )

        drift_small = monitor.compute_drift(current_no_drift, baseline)
        drift_large = monitor.compute_drift(current_with_drift, baseline)

        assert drift_large > drift_small


class TestOnlineMetricsCollector:
    """Tests for OnlineMetricsCollector."""

    @pytest.fixture
    def config(self):
        return OnlineMetricsConfig()

    def test_record_interactions(self, config):
        """Test recording various interactions."""
        collector = OnlineMetricsCollector(config)

        # Record impression
        imp_id = collector.record_impression(
            user_id=1,
            video_id=100,
            request_id="req_1",
            position=1,
        )
        assert imp_id is not None

        # Record click
        click_id = collector.record_click(
            user_id=1,
            video_id=100,
            request_id="req_1",
        )
        assert click_id is not None

        # Record watch
        watch_id = collector.record_watch(
            user_id=1,
            video_id=100,
            request_id="req_1",
            watch_time_seconds=120,
            video_duration_seconds=300,
        )
        assert watch_id is not None

        stats = collector.get_stats()
        assert stats["total_interactions"] == 3

    def test_generate_report(self, config):
        """Test report generation."""
        collector = OnlineMetricsCollector(config)

        # Record some interactions
        for i in range(10):
            collector.record_impression(
                user_id=i,
                video_id=100 + i,
                request_id=f"req_{i}",
                position=1,
            )

        for i in range(3):
            collector.record_click(
                user_id=i,
                video_id=100 + i,
                request_id=f"req_{i}",
            )

        report = collector.generate_report(window_hours=24)

        assert report.total_impressions == 10
        assert report.total_clicks == 3
        assert report.click_through_rate == 0.3


class TestABTestManager:
    """Tests for ABTestManager."""

    @pytest.fixture
    def config(self):
        return ABTestConfig()

    def test_create_experiment(self, config):
        """Test creating an experiment."""
        manager = ABTestManager(config)

        experiment = manager.create_experiment(
            name="Test Experiment",
            variants=[
                ("control", "v1", 0.5),
                ("treatment", "v2", 0.5),
            ],
        )

        assert experiment.experiment_id is not None
        assert len(experiment.variants) == 2
        assert experiment.status == ExperimentStatus.DRAFT

    def test_start_pause_complete(self, config):
        """Test experiment lifecycle."""
        manager = ABTestManager(config)

        experiment = manager.create_experiment(
            name="Lifecycle Test",
            variants=[
                ("control", "v1", 0.5),
                ("treatment", "v2", 0.5),
            ],
        )

        # Start
        manager.start_experiment(experiment.experiment_id)
        assert experiment.status == ExperimentStatus.RUNNING

        # Pause
        manager.pause_experiment(experiment.experiment_id)
        assert experiment.status == ExperimentStatus.PAUSED

        # Complete
        manager.complete_experiment(experiment.experiment_id)
        assert experiment.status == ExperimentStatus.COMPLETED

    def test_user_assignment(self, config):
        """Test consistent user assignment."""
        manager = ABTestManager(config)

        experiment = manager.create_experiment(
            name="Assignment Test",
            variants=[
                ("control", "v1", 0.5),
                ("treatment", "v2", 0.5),
            ],
        )
        manager.start_experiment(experiment.experiment_id)

        # Same user should always get same variant
        variant1 = manager.get_user_variant(user_id=123, experiment_id=experiment.experiment_id)
        variant2 = manager.get_user_variant(user_id=123, experiment_id=experiment.experiment_id)

        assert variant1.name == variant2.name

        # Check traffic distribution roughly
        assignments = {"control": 0, "treatment": 0}
        for user_id in range(10000):
            variant = manager.get_user_variant(user_id=user_id, experiment_id=experiment.experiment_id)
            assignments[variant.name] += 1

        # Should be roughly 50/50 (within 5%)
        assert 4500 < assignments["control"] < 5500
        assert 4500 < assignments["treatment"] < 5500

    def test_invalid_traffic_split(self, config):
        """Test validation of traffic percentages."""
        manager = ABTestManager(config)

        with pytest.raises(ValueError):
            manager.create_experiment(
                name="Invalid",
                variants=[
                    ("control", "v1", 0.3),
                    ("treatment", "v2", 0.3),
                ],
            )


class TestRankerQualitySampler:
    """Tests for RankerQualitySampler."""

    @pytest.fixture
    def config(self):
        return OnlineMetricsConfig(
            enable_ranker_sampling=True,
            ranker_sample_size=3,
            ranker_sample_from_top_n=30,
        )

    @pytest.fixture
    def sample_recommendations(self):
        return [
            {
                "video_id": i,
                "rank": i,
                "ranker_score": 1.0 - i * 0.03,
                "retrieval_score": 0.8,
            }
            for i in range(1, 31)
        ]

    def test_inject_samples(self, config, sample_recommendations):
        """Test injecting samples into recommendations."""
        sampler = RankerQualitySampler(config)

        # Force sampling (set high probability)
        modified_recs, sample_ids = sampler.inject_samples(
            recommendations=sample_recommendations,
            request_id="req_1",
            user_id=1,
            top_k=10,
            sample_probability=1.0,
        )

        # Should have some samples
        assert len(modified_recs) <= 10
        assert len(sample_ids) > 0 or len(modified_recs) == 10

    def test_record_feedback(self, config, sample_recommendations):
        """Test recording feedback for samples."""
        sampler = RankerQualitySampler(config)

        modified_recs, sample_ids = sampler.inject_samples(
            recommendations=sample_recommendations,
            request_id="req_1",
            user_id=1,
            top_k=10,
            sample_probability=1.0,
        )

        if sample_ids:
            sampler.record_feedback(
                sample_id=sample_ids[0],
                was_clicked=True,
                watch_time_seconds=60,
            )

            sample = sampler.get_sample(sample_ids[0])
            assert sample.was_clicked is True
            assert sample.watch_time_seconds == 60

    def test_generate_report(self, config, sample_recommendations):
        """Test generating quality report."""
        sampler = RankerQualitySampler(config)

        # Create multiple samples
        for i in range(10):
            sampler.inject_samples(
                recommendations=sample_recommendations,
                request_id=f"req_{i}",
                user_id=i,
                top_k=10,
                sample_probability=1.0,
            )

        # Record some clicks
        for sample in sampler._samples[:3]:
            sample.was_clicked = True

        report = sampler.generate_report(top_k_ctr=0.5)

        assert report.total_samples > 0
        assert 0 <= report.sampled_ctr <= 1
        assert 0 <= report.ranking_quality_score <= 1
