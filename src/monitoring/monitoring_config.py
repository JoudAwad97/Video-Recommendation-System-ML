"""
Configuration for monitoring components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MonitoringType(Enum):
    """Types of monitoring."""

    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    MODEL_BIAS = "model_bias"
    FEATURE_ATTRIBUTION = "feature_attribution"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configuration for monitoring alerts."""

    # Performance thresholds
    precision_drop_threshold: float = 0.1  # Alert if precision drops by 10%
    recall_drop_threshold: float = 0.1
    latency_p99_threshold_ms: float = 500.0
    error_rate_threshold: float = 0.01  # 1% error rate

    # Data quality thresholds
    missing_value_threshold: float = 0.05  # 5% missing values
    feature_drift_threshold: float = 0.3  # KS statistic threshold
    cardinality_change_threshold: float = 0.2  # 20% change in unique values

    # Alerting destinations
    enable_cloudwatch_alerts: bool = True
    enable_sns_alerts: bool = False
    sns_topic_arn: str = ""

    # Alert cooldown
    alert_cooldown_minutes: int = 30


@dataclass
class DataCaptureConfig:
    """Configuration for inference data capture."""

    # Capture settings
    enable_capture: bool = True
    capture_percentage: float = 1.0  # Capture 100% by default
    capture_inputs: bool = True
    capture_outputs: bool = True

    # Storage settings
    storage_type: str = "local"  # Options: "local", "s3", "kinesis"
    local_storage_path: str = "data/inference_logs"
    s3_bucket: str = ""
    s3_prefix: str = "inference-data"
    kinesis_stream_name: str = ""

    # Batching settings
    batch_size: int = 100
    flush_interval_seconds: int = 60

    # Data format
    output_format: str = "jsonl"  # Options: "jsonl", "parquet"


@dataclass
class BaselineConfig:
    """Configuration for baseline statistics."""

    # Baseline data paths
    baseline_data_path: str = "data/baseline"
    baseline_statistics_path: str = "data/baseline/statistics.json"
    baseline_constraints_path: str = "data/baseline/constraints.json"

    # Baseline computation
    sample_size: int = 10000
    compute_percentiles: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )


@dataclass
class MonitoringScheduleConfig:
    """Configuration for monitoring schedule."""

    # Schedule settings
    enable_scheduled_monitoring: bool = True
    monitoring_interval_hours: int = 2

    # Data window
    data_window_hours: int = 24  # Analyze last 24 hours of data
    min_samples_required: int = 100


@dataclass
class OnlineMetricsConfig:
    """Configuration for online metrics collection."""

    # Metrics to track
    track_ctr: bool = True
    track_watch_time: bool = True
    track_completions: bool = True
    track_explicit_feedback: bool = True

    # Aggregation settings
    aggregation_window_minutes: int = 15
    report_interval_minutes: int = 60

    # Storage
    metrics_storage_path: str = "data/online_metrics"

    # Ranker sampling for quality measurement
    enable_ranker_sampling: bool = True
    ranker_sample_size: int = 5  # Sample 5 videos outside top-k
    ranker_sample_from_top_n: int = 30  # Sample from top-30


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    # Test settings
    enable_ab_testing: bool = False
    default_traffic_split: Dict[str, float] = field(
        default_factory=lambda: {"control": 0.5, "treatment": 0.5}
    )

    # Statistical settings
    confidence_level: float = 0.95
    min_sample_size_per_variant: int = 1000
    max_test_duration_days: int = 14

    # Storage
    experiment_storage_path: str = "data/experiments"
    results_storage_path: str = "data/experiment_results"


@dataclass
class MonitoringConfig:
    """Main configuration for the monitoring system."""

    # Sub-configurations
    alerts: AlertConfig = field(default_factory=AlertConfig)
    data_capture: DataCaptureConfig = field(default_factory=DataCaptureConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    schedule: MonitoringScheduleConfig = field(default_factory=MonitoringScheduleConfig)
    online_metrics: OnlineMetricsConfig = field(default_factory=OnlineMetricsConfig)
    ab_testing: ABTestConfig = field(default_factory=ABTestConfig)

    # Feature configuration for monitoring
    numerical_features: List[str] = field(
        default_factory=lambda: [
            "age", "video_duration", "view_count", "like_count",
            "comment_count", "channel_subscriber_count",
        ]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "country", "user_language", "category", "video_language",
            "device", "popularity",
        ]
    )

    # Model versions
    production_model_version: str = "v1"
    shadow_model_version: Optional[str] = None

    def get_all_features(self) -> List[str]:
        """Get all monitored features."""
        return self.numerical_features + self.categorical_features
