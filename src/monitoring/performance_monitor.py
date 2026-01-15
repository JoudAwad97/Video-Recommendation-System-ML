"""
Model performance monitoring.

Compares model predictions on production data vs training data
to detect performance degradation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from ..utils.logging_utils import get_logger
from .monitoring_config import MonitoringConfig, AlertSeverity
from .data_capture import InferenceRecord

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a time window."""

    # Time window
    start_time: str = ""
    end_time: str = ""
    num_samples: int = 0

    # Recommendation metrics
    avg_num_recommendations: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_ranking_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0

    # Latency percentiles
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0

    # Score distributions
    avg_retrieval_score: float = 0.0
    avg_ranker_score: float = 0.0
    avg_final_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_samples": self.num_samples,
            "avg_num_recommendations": self.avg_num_recommendations,
            "avg_retrieval_latency_ms": self.avg_retrieval_latency_ms,
            "avg_ranking_latency_ms": self.avg_ranking_latency_ms,
            "avg_total_latency_ms": self.avg_total_latency_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "avg_retrieval_score": self.avg_retrieval_score,
            "avg_ranker_score": self.avg_ranker_score,
            "avg_final_score": self.avg_final_score,
        }


@dataclass
class PerformanceAlert:
    """An alert for performance degradation."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    baseline_value: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class PerformanceReport:
    """Performance monitoring report."""

    report_id: str
    generated_at: str
    monitoring_window_hours: int

    # Metrics
    current_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    baseline_metrics: Optional[PerformanceMetrics] = None

    # Comparisons
    metric_changes: Dict[str, float] = field(default_factory=dict)

    # Alerts
    alerts: List[PerformanceAlert] = field(default_factory=list)

    # Status
    status: str = "healthy"  # healthy, degraded, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "monitoring_window_hours": self.monitoring_window_hours,
            "current_metrics": self.current_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "metric_changes": self.metric_changes,
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity.value,
                    "message": a.message,
                    "metric_name": a.metric_name,
                    "current_value": a.current_value,
                    "threshold_value": a.threshold_value,
                }
                for a in self.alerts
            ],
            "status": self.status,
        }


class PerformanceMonitor:
    """Monitor for model performance degradation.

    Tracks inference latencies, prediction distributions, and error rates.
    Compares current performance against baseline to detect degradation.

    Example:
        >>> monitor = PerformanceMonitor(config)
        >>> monitor.load_baseline("baseline/metrics.json")
        >>> report = monitor.analyze(inference_records)
        >>> if report.status == "critical":
        ...     send_alert(report.alerts)
    """

    def __init__(self, config: MonitoringConfig):
        """Initialize the performance monitor.

        Args:
            config: Monitoring configuration.
        """
        self.config = config
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self._alert_counter = 0

    def load_baseline(self, baseline_path: str) -> None:
        """Load baseline performance metrics.

        Args:
            baseline_path: Path to baseline metrics JSON file.
        """
        baseline_path = Path(baseline_path)
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_path}")
            return

        with open(baseline_path) as f:
            data = json.load(f)

        self.baseline_metrics = PerformanceMetrics(**data)
        logger.info(f"Loaded baseline metrics from {baseline_path}")

    def save_baseline(self, metrics: PerformanceMetrics, output_path: str) -> None:
        """Save metrics as baseline.

        Args:
            metrics: Performance metrics to save as baseline.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Saved baseline metrics to {output_path}")

    def compute_metrics(
        self,
        records: List[InferenceRecord],
    ) -> PerformanceMetrics:
        """Compute performance metrics from inference records.

        Args:
            records: List of inference records.

        Returns:
            Computed performance metrics.
        """
        if not records:
            return PerformanceMetrics()

        # Extract timestamps
        timestamps = [r.timestamp for r in records]
        start_time = min(timestamps)
        end_time = max(timestamps)

        # Extract latencies
        latencies = [r.inference_latency_ms for r in records]

        # Extract stage latencies
        retrieval_latencies = []
        ranking_latencies = []
        for r in records:
            stage_lat = r.metadata.get("stage_latencies", {})
            if "retrieval" in stage_lat:
                retrieval_latencies.append(stage_lat["retrieval"])
            if "ranking" in stage_lat:
                ranking_latencies.append(stage_lat["ranking"])

        # Extract prediction scores
        retrieval_scores = []
        ranker_scores = []
        final_scores = []
        num_recs = []

        for r in records:
            recs = r.output_predictions.get("recommendations", [])
            num_recs.append(len(recs))

            for rec in recs:
                if "retrieval_score" in rec:
                    retrieval_scores.append(rec["retrieval_score"])
                if "ranker_score" in rec:
                    ranker_scores.append(rec["ranker_score"])
                if "score" in rec:
                    final_scores.append(rec["score"])

        # Count errors (records with empty predictions)
        error_count = sum(
            1 for r in records
            if not r.output_predictions.get("recommendations")
        )

        return PerformanceMetrics(
            start_time=start_time,
            end_time=end_time,
            num_samples=len(records),
            avg_num_recommendations=np.mean(num_recs) if num_recs else 0,
            avg_retrieval_latency_ms=np.mean(retrieval_latencies) if retrieval_latencies else 0,
            avg_ranking_latency_ms=np.mean(ranking_latencies) if ranking_latencies else 0,
            avg_total_latency_ms=np.mean(latencies),
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            error_count=error_count,
            error_rate=error_count / len(records),
            avg_retrieval_score=np.mean(retrieval_scores) if retrieval_scores else 0,
            avg_ranker_score=np.mean(ranker_scores) if ranker_scores else 0,
            avg_final_score=np.mean(final_scores) if final_scores else 0,
        )

    def analyze(
        self,
        records: List[InferenceRecord],
        monitoring_window_hours: int = 24,
    ) -> PerformanceReport:
        """Analyze performance and generate report.

        Args:
            records: Inference records to analyze.
            monitoring_window_hours: Window size in hours.

        Returns:
            Performance report with metrics and alerts.
        """
        import uuid

        report = PerformanceReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow().isoformat(),
            monitoring_window_hours=monitoring_window_hours,
        )

        if not records:
            logger.warning("No records to analyze")
            return report

        # Compute current metrics
        current_metrics = self.compute_metrics(records)
        report.current_metrics = current_metrics
        report.baseline_metrics = self.baseline_metrics

        # Compare with baseline if available
        if self.baseline_metrics:
            report.metric_changes = self._compute_changes(
                current_metrics,
                self.baseline_metrics,
            )

        # Generate alerts
        report.alerts = self._check_alerts(current_metrics)

        # Determine overall status
        report.status = self._determine_status(report.alerts)

        logger.info(
            f"Performance analysis complete: {len(records)} records, "
            f"{len(report.alerts)} alerts, status={report.status}"
        )

        return report

    def _compute_changes(
        self,
        current: PerformanceMetrics,
        baseline: PerformanceMetrics,
    ) -> Dict[str, float]:
        """Compute percentage changes from baseline.

        Args:
            current: Current metrics.
            baseline: Baseline metrics.

        Returns:
            Dictionary of metric changes (as percentages).
        """
        changes = {}

        def safe_change(curr, base):
            if base == 0:
                return 0.0 if curr == 0 else float("inf")
            return (curr - base) / base

        changes["latency_change"] = safe_change(
            current.avg_total_latency_ms,
            baseline.avg_total_latency_ms,
        )
        changes["latency_p99_change"] = safe_change(
            current.latency_p99_ms,
            baseline.latency_p99_ms,
        )
        changes["error_rate_change"] = safe_change(
            current.error_rate,
            baseline.error_rate,
        )
        changes["avg_score_change"] = safe_change(
            current.avg_final_score,
            baseline.avg_final_score,
        )

        return changes

    def _check_alerts(
        self,
        metrics: PerformanceMetrics,
    ) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts.

        Args:
            metrics: Current performance metrics.

        Returns:
            List of performance alerts.
        """
        alerts = []

        # Check latency P99
        if metrics.latency_p99_ms > self.config.alerts.latency_p99_threshold_ms:
            alerts.append(self._create_alert(
                alert_type="high_latency",
                severity=AlertSeverity.WARNING,
                message=f"P99 latency ({metrics.latency_p99_ms:.2f}ms) exceeds threshold",
                metric_name="latency_p99_ms",
                current_value=metrics.latency_p99_ms,
                threshold_value=self.config.alerts.latency_p99_threshold_ms,
            ))

        # Check error rate
        if metrics.error_rate > self.config.alerts.error_rate_threshold:
            alerts.append(self._create_alert(
                alert_type="high_error_rate",
                severity=AlertSeverity.CRITICAL,
                message=f"Error rate ({metrics.error_rate:.2%}) exceeds threshold",
                metric_name="error_rate",
                current_value=metrics.error_rate,
                threshold_value=self.config.alerts.error_rate_threshold,
            ))

        # Check baseline comparisons
        if self.baseline_metrics:
            # Latency increase
            latency_increase = (
                (metrics.avg_total_latency_ms - self.baseline_metrics.avg_total_latency_ms)
                / self.baseline_metrics.avg_total_latency_ms
                if self.baseline_metrics.avg_total_latency_ms > 0 else 0
            )
            if latency_increase > 0.5:  # 50% increase
                alerts.append(self._create_alert(
                    alert_type="latency_regression",
                    severity=AlertSeverity.WARNING,
                    message=f"Latency increased by {latency_increase:.1%} from baseline",
                    metric_name="avg_total_latency_ms",
                    current_value=metrics.avg_total_latency_ms,
                    threshold_value=self.baseline_metrics.avg_total_latency_ms * 1.5,
                    baseline_value=self.baseline_metrics.avg_total_latency_ms,
                ))

            # Score drop
            if self.baseline_metrics.avg_final_score > 0:
                score_drop = (
                    self.baseline_metrics.avg_final_score - metrics.avg_final_score
                ) / self.baseline_metrics.avg_final_score

                if score_drop > 0.2:  # 20% drop
                    alerts.append(self._create_alert(
                        alert_type="score_degradation",
                        severity=AlertSeverity.WARNING,
                        message=f"Average prediction score dropped by {score_drop:.1%}",
                        metric_name="avg_final_score",
                        current_value=metrics.avg_final_score,
                        threshold_value=self.baseline_metrics.avg_final_score * 0.8,
                        baseline_value=self.baseline_metrics.avg_final_score,
                    ))

        return alerts

    def _create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        baseline_value: Optional[float] = None,
    ) -> PerformanceAlert:
        """Create a performance alert.

        Args:
            alert_type: Type of alert.
            severity: Alert severity.
            message: Alert message.
            metric_name: Name of the metric.
            current_value: Current metric value.
            threshold_value: Threshold that was exceeded.
            baseline_value: Optional baseline value.

        Returns:
            Performance alert.
        """
        self._alert_counter += 1
        return PerformanceAlert(
            alert_id=f"perf_alert_{self._alert_counter}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            baseline_value=baseline_value,
        )

    def _determine_status(self, alerts: List[PerformanceAlert]) -> str:
        """Determine overall status from alerts.

        Args:
            alerts: List of alerts.

        Returns:
            Status string: "healthy", "degraded", or "critical".
        """
        if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
            return "critical"
        elif any(a.severity == AlertSeverity.WARNING for a in alerts):
            return "degraded"
        return "healthy"
