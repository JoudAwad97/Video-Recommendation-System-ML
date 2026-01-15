"""
Data quality monitoring.

Monitors the quality of production data compared to training data,
detecting feature drift, missing values, and distribution changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import Counter

from ..utils.logging_utils import get_logger
from .monitoring_config import MonitoringConfig, AlertSeverity
from .data_capture import InferenceRecord

logger = get_logger(__name__)


@dataclass
class FeatureStatistics:
    """Statistics for a single feature."""

    feature_name: str
    feature_type: str  # "numerical" or "categorical"

    # Common stats
    count: int = 0
    missing_count: int = 0
    missing_rate: float = 0.0

    # Numerical stats
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    percentiles: Dict[str, float] = field(default_factory=dict)

    # Categorical stats
    unique_count: int = 0
    value_counts: Dict[str, int] = field(default_factory=dict)
    top_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_rate": self.missing_rate,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "percentiles": self.percentiles,
            "unique_count": self.unique_count,
            "value_counts": self.value_counts,
            "top_values": self.top_values,
        }


@dataclass
class DataQualityAlert:
    """Alert for data quality issues."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    feature_name: str
    message: str
    current_value: float
    baseline_value: Optional[float] = None
    threshold: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class DataQualityReport:
    """Data quality monitoring report."""

    report_id: str
    generated_at: str
    num_records: int

    # Feature statistics
    feature_statistics: Dict[str, FeatureStatistics] = field(default_factory=dict)
    baseline_statistics: Optional[Dict[str, FeatureStatistics]] = None

    # Drift scores
    drift_scores: Dict[str, float] = field(default_factory=dict)

    # Alerts
    alerts: List[DataQualityAlert] = field(default_factory=list)

    # Status
    status: str = "healthy"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "num_records": self.num_records,
            "feature_statistics": {
                k: v.to_dict() for k, v in self.feature_statistics.items()
            },
            "drift_scores": self.drift_scores,
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity.value,
                    "feature_name": a.feature_name,
                    "message": a.message,
                }
                for a in self.alerts
            ],
            "status": self.status,
        }


class DataQualityMonitor:
    """Monitor for data quality issues.

    Detects:
    - Missing values
    - Feature drift (distribution changes)
    - Cardinality changes
    - Out-of-range values

    Example:
        >>> monitor = DataQualityMonitor(config)
        >>> monitor.load_baseline("baseline/statistics.json")
        >>> report = monitor.analyze(inference_records)
    """

    def __init__(self, config: MonitoringConfig):
        """Initialize the data quality monitor.

        Args:
            config: Monitoring configuration.
        """
        self.config = config
        self.baseline_statistics: Dict[str, FeatureStatistics] = {}
        self._alert_counter = 0

    def load_baseline(self, baseline_path: str) -> None:
        """Load baseline statistics.

        Args:
            baseline_path: Path to baseline statistics JSON file.
        """
        baseline_path = Path(baseline_path)
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_path}")
            return

        with open(baseline_path) as f:
            data = json.load(f)

        self.baseline_statistics = {
            name: FeatureStatistics(**stats)
            for name, stats in data.items()
        }
        logger.info(f"Loaded baseline statistics for {len(self.baseline_statistics)} features")

    def save_baseline(
        self,
        statistics: Dict[str, FeatureStatistics],
        output_path: str,
    ) -> None:
        """Save statistics as baseline.

        Args:
            statistics: Feature statistics to save.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {name: stats.to_dict() for name, stats in statistics.items()}
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved baseline statistics to {output_path}")

    def compute_statistics(
        self,
        records: List[InferenceRecord],
    ) -> Dict[str, FeatureStatistics]:
        """Compute statistics for all features.

        Args:
            records: List of inference records.

        Returns:
            Dictionary of feature statistics.
        """
        if not records:
            return {}

        statistics = {}

        # Extract all features from records
        feature_values: Dict[str, List[Any]] = {}
        for record in records:
            for name, value in record.input_features.items():
                if name not in feature_values:
                    feature_values[name] = []
                feature_values[name].append(value)

        # Compute statistics for numerical features
        for feature_name in self.config.numerical_features:
            if feature_name in feature_values:
                values = feature_values[feature_name]
                statistics[feature_name] = self._compute_numerical_stats(
                    feature_name, values
                )

        # Compute statistics for categorical features
        for feature_name in self.config.categorical_features:
            if feature_name in feature_values:
                values = feature_values[feature_name]
                statistics[feature_name] = self._compute_categorical_stats(
                    feature_name, values
                )

        return statistics

    def _compute_numerical_stats(
        self,
        feature_name: str,
        values: List[Any],
    ) -> FeatureStatistics:
        """Compute statistics for a numerical feature.

        Args:
            feature_name: Name of the feature.
            values: List of feature values.

        Returns:
            Feature statistics.
        """
        # Convert to numpy, handling None values
        valid_values = [v for v in values if v is not None and not np.isnan(float(v))]
        missing_count = len(values) - len(valid_values)

        if not valid_values:
            return FeatureStatistics(
                feature_name=feature_name,
                feature_type="numerical",
                count=len(values),
                missing_count=missing_count,
                missing_rate=1.0,
            )

        arr = np.array(valid_values, dtype=float)

        # Compute percentiles
        percentile_values = [1, 5, 25, 50, 75, 95, 99]
        percentiles = {
            f"p{p}": float(np.percentile(arr, p))
            for p in percentile_values
        }

        return FeatureStatistics(
            feature_name=feature_name,
            feature_type="numerical",
            count=len(values),
            missing_count=missing_count,
            missing_rate=missing_count / len(values),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min_value=float(np.min(arr)),
            max_value=float(np.max(arr)),
            percentiles=percentiles,
        )

    def _compute_categorical_stats(
        self,
        feature_name: str,
        values: List[Any],
    ) -> FeatureStatistics:
        """Compute statistics for a categorical feature.

        Args:
            feature_name: Name of the feature.
            values: List of feature values.

        Returns:
            Feature statistics.
        """
        # Handle missing values
        valid_values = [str(v) for v in values if v is not None]
        missing_count = len(values) - len(valid_values)

        # Count values
        value_counts = Counter(valid_values)
        top_values = [v for v, _ in value_counts.most_common(10)]

        return FeatureStatistics(
            feature_name=feature_name,
            feature_type="categorical",
            count=len(values),
            missing_count=missing_count,
            missing_rate=missing_count / len(values) if values else 0,
            unique_count=len(value_counts),
            value_counts=dict(value_counts),
            top_values=top_values,
        )

    def compute_drift(
        self,
        current_stats: FeatureStatistics,
        baseline_stats: FeatureStatistics,
    ) -> float:
        """Compute drift score between current and baseline statistics.

        Uses KS statistic for numerical features and
        population stability index (PSI) for categorical.

        Args:
            current_stats: Current feature statistics.
            baseline_stats: Baseline feature statistics.

        Returns:
            Drift score (0-1, higher means more drift).
        """
        if current_stats.feature_type == "numerical":
            return self._compute_numerical_drift(current_stats, baseline_stats)
        else:
            return self._compute_categorical_drift(current_stats, baseline_stats)

    def _compute_numerical_drift(
        self,
        current: FeatureStatistics,
        baseline: FeatureStatistics,
    ) -> float:
        """Compute drift for numerical features using normalized difference.

        Args:
            current: Current statistics.
            baseline: Baseline statistics.

        Returns:
            Drift score.
        """
        if baseline.std == 0:
            return 0.0

        # Use normalized mean difference as a simple drift metric
        mean_diff = abs(current.mean - baseline.mean) / baseline.std

        # Scale to 0-1 range (sigmoid-like)
        drift_score = 1 - (1 / (1 + mean_diff))

        return min(drift_score, 1.0)

    def _compute_categorical_drift(
        self,
        current: FeatureStatistics,
        baseline: FeatureStatistics,
    ) -> float:
        """Compute drift for categorical features using PSI-like metric.

        Args:
            current: Current statistics.
            baseline: Baseline statistics.

        Returns:
            Drift score.
        """
        if not baseline.value_counts or not current.value_counts:
            return 0.0

        # Get all categories
        all_categories = set(baseline.value_counts.keys()) | set(current.value_counts.keys())

        # Compute distributions
        baseline_total = sum(baseline.value_counts.values())
        current_total = sum(current.value_counts.values())

        if baseline_total == 0 or current_total == 0:
            return 0.0

        psi = 0.0
        for category in all_categories:
            baseline_pct = baseline.value_counts.get(category, 0) / baseline_total
            current_pct = current.value_counts.get(category, 0) / current_total

            # Add small epsilon to avoid log(0)
            eps = 0.0001
            baseline_pct = max(baseline_pct, eps)
            current_pct = max(current_pct, eps)

            psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

        # Normalize PSI to 0-1 range
        # PSI > 0.25 typically indicates significant drift
        drift_score = min(psi / 0.5, 1.0)

        return drift_score

    def analyze(
        self,
        records: List[InferenceRecord],
    ) -> DataQualityReport:
        """Analyze data quality and generate report.

        Args:
            records: Inference records to analyze.

        Returns:
            Data quality report.
        """
        import uuid

        report = DataQualityReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow().isoformat(),
            num_records=len(records),
        )

        if not records:
            logger.warning("No records to analyze")
            return report

        # Compute current statistics
        report.feature_statistics = self.compute_statistics(records)
        report.baseline_statistics = self.baseline_statistics

        # Compute drift scores if baseline exists
        if self.baseline_statistics:
            for feature_name, current_stats in report.feature_statistics.items():
                if feature_name in self.baseline_statistics:
                    baseline_stats = self.baseline_statistics[feature_name]
                    drift = self.compute_drift(current_stats, baseline_stats)
                    report.drift_scores[feature_name] = drift

        # Generate alerts
        report.alerts = self._check_alerts(report.feature_statistics, report.drift_scores)

        # Determine status
        report.status = self._determine_status(report.alerts)

        logger.info(
            f"Data quality analysis complete: {len(records)} records, "
            f"{len(report.alerts)} alerts, status={report.status}"
        )

        return report

    def _check_alerts(
        self,
        statistics: Dict[str, FeatureStatistics],
        drift_scores: Dict[str, float],
    ) -> List[DataQualityAlert]:
        """Check for data quality issues and generate alerts.

        Args:
            statistics: Current feature statistics.
            drift_scores: Drift scores by feature.

        Returns:
            List of data quality alerts.
        """
        alerts = []

        # Check missing values
        for name, stats in statistics.items():
            if stats.missing_rate > self.config.alerts.missing_value_threshold:
                alerts.append(self._create_alert(
                    alert_type="high_missing_rate",
                    severity=AlertSeverity.WARNING,
                    feature_name=name,
                    message=f"Missing rate ({stats.missing_rate:.1%}) exceeds threshold",
                    current_value=stats.missing_rate,
                    threshold=self.config.alerts.missing_value_threshold,
                ))

        # Check drift
        for name, drift in drift_scores.items():
            if drift > self.config.alerts.feature_drift_threshold:
                alerts.append(self._create_alert(
                    alert_type="feature_drift",
                    severity=AlertSeverity.WARNING,
                    feature_name=name,
                    message=f"Feature drift ({drift:.2f}) exceeds threshold",
                    current_value=drift,
                    threshold=self.config.alerts.feature_drift_threshold,
                ))

        # Check cardinality changes
        for name, stats in statistics.items():
            if name in self.baseline_statistics:
                baseline = self.baseline_statistics[name]
                if baseline.unique_count > 0:
                    cardinality_change = abs(
                        stats.unique_count - baseline.unique_count
                    ) / baseline.unique_count

                    if cardinality_change > self.config.alerts.cardinality_change_threshold:
                        alerts.append(self._create_alert(
                            alert_type="cardinality_change",
                            severity=AlertSeverity.INFO,
                            feature_name=name,
                            message=f"Cardinality changed by {cardinality_change:.1%}",
                            current_value=stats.unique_count,
                            baseline_value=baseline.unique_count,
                            threshold=self.config.alerts.cardinality_change_threshold,
                        ))

        return alerts

    def _create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        feature_name: str,
        message: str,
        current_value: float,
        baseline_value: Optional[float] = None,
        threshold: float = 0.0,
    ) -> DataQualityAlert:
        """Create a data quality alert.

        Args:
            alert_type: Type of alert.
            severity: Alert severity.
            feature_name: Name of the affected feature.
            message: Alert message.
            current_value: Current value.
            baseline_value: Baseline value for comparison.
            threshold: Threshold that was exceeded.

        Returns:
            Data quality alert.
        """
        self._alert_counter += 1
        return DataQualityAlert(
            alert_id=f"dq_alert_{self._alert_counter}",
            alert_type=alert_type,
            severity=severity,
            feature_name=feature_name,
            message=message,
            current_value=current_value,
            baseline_value=baseline_value,
            threshold=threshold,
        )

    def _determine_status(self, alerts: List[DataQualityAlert]) -> str:
        """Determine overall status from alerts.

        Args:
            alerts: List of alerts.

        Returns:
            Status string.
        """
        if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
            return "critical"
        elif any(a.severity == AlertSeverity.WARNING for a in alerts):
            return "degraded"
        return "healthy"
