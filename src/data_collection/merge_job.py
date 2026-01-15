"""
Prediction-label merge job for combining predictions with ground truth.

Joins inference predictions with collected ground truth labels
to create datasets for model evaluation and retraining.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np

from ..utils.logging_utils import get_logger
from .collection_config import DataCollectionConfig, MergeJobConfig
from .inference_tracker import TrackedPrediction
from .ground_truth_collector import LabeledInteraction

logger = get_logger(__name__)


@dataclass
class MergedRecord:
    """A merged prediction-label record."""

    # Identifiers
    record_id: str
    inference_id: str
    user_id: int
    video_id: int

    # Timestamps
    inference_timestamp: str
    feedback_timestamp: str
    merge_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Prediction info
    predicted_score: float = 0.0
    predicted_rank: int = 0
    model_version: str = ""
    experiment_id: Optional[str] = None

    # Label info
    label: int = 0  # 1 = positive, 0 = negative
    label_source: str = ""
    label_confidence: float = 1.0

    # Features
    user_features: Dict[str, Any] = field(default_factory=dict)
    video_features: Dict[str, Any] = field(default_factory=dict)
    context_features: Dict[str, Any] = field(default_factory=dict)

    # Engagement metrics
    watch_time_seconds: float = 0.0
    watch_percentage: float = 0.0
    position_shown: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "inference_id": self.inference_id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "inference_timestamp": self.inference_timestamp,
            "feedback_timestamp": self.feedback_timestamp,
            "merge_timestamp": self.merge_timestamp,
            "predicted_score": self.predicted_score,
            "predicted_rank": self.predicted_rank,
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
            "label": self.label,
            "label_source": self.label_source,
            "label_confidence": self.label_confidence,
            "user_features": self.user_features,
            "video_features": self.video_features,
            "context_features": self.context_features,
            "watch_time_seconds": self.watch_time_seconds,
            "watch_percentage": self.watch_percentage,
            "position_shown": self.position_shown,
        }


@dataclass
class MergeResult:
    """Result of a merge job."""

    job_id: str
    started_at: str
    completed_at: str

    # Counts
    predictions_processed: int = 0
    labels_processed: int = 0
    records_merged: int = 0
    records_unmatched: int = 0

    # Output
    output_path: str = ""
    output_format: str = ""

    # Statistics
    positive_count: int = 0
    negative_count: int = 0
    avg_label_delay_hours: float = 0.0

    # By model version
    counts_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "predictions_processed": self.predictions_processed,
            "labels_processed": self.labels_processed,
            "records_merged": self.records_merged,
            "records_unmatched": self.records_unmatched,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "avg_label_delay_hours": self.avg_label_delay_hours,
            "counts_by_model": self.counts_by_model,
        }


class PredictionLabelMerger:
    """Merges predictions with ground truth labels.

    Performs the join between:
    1. Model predictions (with inference_id, user features, scores)
    2. Ground truth labels (with inference_id, label, engagement metrics)

    This is used in the AWS architecture with:
    - Athena for SQL-based joins on S3 data
    - Or in-memory joining for local development

    Example:
        >>> merger = PredictionLabelMerger(config)
        >>> result = merger.merge(predictions, labeled_interactions)
        >>> merged_df = merger.to_dataframe(result.records)
    """

    def __init__(self, config: MergeJobConfig):
        """Initialize the merger.

        Args:
            config: Merge job configuration.
        """
        self.config = config
        self._record_counter = 0
        self._job_counter = 0

        # Output path
        self._output_path = Path(config.merged_output_path)
        self._output_path.mkdir(parents=True, exist_ok=True)

    def merge(
        self,
        predictions: List[TrackedPrediction],
        labeled_interactions: List[LabeledInteraction],
    ) -> Tuple[List[MergedRecord], MergeResult]:
        """Merge predictions with labeled interactions.

        Args:
            predictions: List of tracked predictions.
            labeled_interactions: List of labeled interactions.

        Returns:
            Tuple of (merged records, merge result).
        """
        import uuid

        self._job_counter += 1
        job_id = f"merge_job_{self._job_counter}"
        started_at = datetime.utcnow().isoformat()

        # Create index of predictions by inference_id
        predictions_by_id: Dict[str, TrackedPrediction] = {
            p.inference_id: p for p in predictions
        }

        # Create index of labels by inference_id and video_id
        labels_by_key: Dict[Tuple[str, int], LabeledInteraction] = {}
        for label in labeled_interactions:
            key = (label.inference_id, label.video_id)
            # Keep the most recent label if duplicates
            if key not in labels_by_key or label.feedback_timestamp > labels_by_key[key].feedback_timestamp:
                labels_by_key[key] = label

        # Perform merge
        merged_records = []
        label_delays = []
        counts_by_model: Dict[str, Dict[str, int]] = {}

        for (inference_id, video_id), label in labels_by_key.items():
            prediction = predictions_by_id.get(inference_id)
            if not prediction:
                continue

            # Check if video was in recommendations
            if video_id not in prediction.recommended_video_ids:
                continue

            # Calculate rank
            try:
                rank = prediction.recommended_video_ids.index(video_id) + 1
            except ValueError:
                rank = 0

            # Create merged record
            self._record_counter += 1
            record = MergedRecord(
                record_id=f"merged_{self._record_counter}",
                inference_id=inference_id,
                user_id=label.user_id,
                video_id=video_id,
                inference_timestamp=prediction.timestamp,
                feedback_timestamp=label.feedback_timestamp,
                predicted_score=prediction.video_scores.get(video_id, 0),
                predicted_rank=rank,
                model_version=prediction.model_version,
                experiment_id=prediction.experiment_id,
                label=label.label,
                label_source=label.label_source,
                label_confidence=label.label_confidence,
                user_features=prediction.user_features,
                context_features=prediction.context_features,
                watch_time_seconds=label.watch_time_seconds,
                watch_percentage=label.watch_percentage,
                position_shown=label.position_shown,
            )
            merged_records.append(record)

            # Track delay
            inf_time = datetime.fromisoformat(prediction.timestamp)
            fb_time = datetime.fromisoformat(label.feedback_timestamp)
            delay_hours = (fb_time - inf_time).total_seconds() / 3600
            label_delays.append(delay_hours)

            # Track by model version
            model = prediction.model_version or "unknown"
            if model not in counts_by_model:
                counts_by_model[model] = {"positive": 0, "negative": 0, "total": 0}
            counts_by_model[model]["total"] += 1
            if label.label == 1:
                counts_by_model[model]["positive"] += 1
            else:
                counts_by_model[model]["negative"] += 1

        # Create result
        completed_at = datetime.utcnow().isoformat()

        positive_count = sum(1 for r in merged_records if r.label == 1)
        negative_count = sum(1 for r in merged_records if r.label == 0)

        result = MergeResult(
            job_id=job_id,
            started_at=started_at,
            completed_at=completed_at,
            predictions_processed=len(predictions),
            labels_processed=len(labeled_interactions),
            records_merged=len(merged_records),
            records_unmatched=len(labeled_interactions) - len(merged_records),
            positive_count=positive_count,
            negative_count=negative_count,
            avg_label_delay_hours=np.mean(label_delays) if label_delays else 0,
            counts_by_model=counts_by_model,
        )

        logger.info(
            f"Merge complete: {result.records_merged} records "
            f"(+{positive_count}/-{negative_count})"
        )

        return merged_records, result

    def to_dataframe(self, records: List[MergedRecord]) -> pd.DataFrame:
        """Convert merged records to DataFrame.

        Args:
            records: List of merged records.

        Returns:
            DataFrame with flattened features.
        """
        rows = []
        for record in records:
            row = {
                "record_id": record.record_id,
                "inference_id": record.inference_id,
                "user_id": record.user_id,
                "video_id": record.video_id,
                "inference_timestamp": record.inference_timestamp,
                "feedback_timestamp": record.feedback_timestamp,
                "predicted_score": record.predicted_score,
                "predicted_rank": record.predicted_rank,
                "model_version": record.model_version,
                "experiment_id": record.experiment_id,
                "label": record.label,
                "label_source": record.label_source,
                "watch_time_seconds": record.watch_time_seconds,
                "watch_percentage": record.watch_percentage,
                "position_shown": record.position_shown,
            }

            # Flatten user features
            for key, value in record.user_features.items():
                row[f"user_{key}"] = value

            # Flatten context features
            for key, value in record.context_features.items():
                row[f"context_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def save_merged_records(
        self,
        records: List[MergedRecord],
        output_path: Optional[str] = None,
        format: Optional[str] = None,
    ) -> str:
        """Save merged records to file.

        Args:
            records: List of merged records.
            output_path: Output file path.
            format: Output format (parquet, jsonl, csv).

        Returns:
            Path to saved file.
        """
        format = format or self.config.output_format
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if not output_path:
            output_path = str(self._output_path / f"merged_{timestamp}.{format}")

        if format == "parquet":
            df = self.to_dataframe(records)
            df.to_parquet(output_path, index=False)

        elif format == "jsonl":
            with open(output_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record.to_dict()) + "\n")

        elif format == "csv":
            df = self.to_dataframe(records)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(records)} merged records to {output_path}")
        return output_path

    def load_merged_records(self, input_path: str) -> List[MergedRecord]:
        """Load merged records from file.

        Args:
            input_path: Input file path.

        Returns:
            List of merged records.
        """
        records = []
        input_path = Path(input_path)

        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
            # Would need to reconstruct nested dicts
            raise NotImplementedError("Loading from parquet not fully implemented")

        elif input_path.suffix in [".jsonl", ".json"]:
            with open(input_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = MergedRecord(**data)
                        records.append(record)

        logger.info(f"Loaded {len(records)} merged records from {input_path}")
        return records

    def generate_athena_query(
        self,
        predictions_table: str,
        labels_table: str,
        output_table: str,
    ) -> str:
        """Generate Athena SQL query for merging.

        Args:
            predictions_table: Name of predictions table.
            labels_table: Name of labels table.
            output_table: Name of output table.

        Returns:
            SQL query string.
        """
        return f"""
        CREATE TABLE {output_table} AS
        SELECT
            p.inference_id,
            p.user_id,
            l.video_id,
            p.timestamp as inference_timestamp,
            l.timestamp as feedback_timestamp,
            p.video_scores[l.video_id] as predicted_score,
            p.model_version,
            p.experiment_id,
            l.label,
            l.label_source,
            l.watch_time_seconds,
            l.watch_percentage,
            p.user_features,
            p.context_features
        FROM {predictions_table} p
        INNER JOIN {labels_table} l
            ON p.inference_id = l.inference_id
        WHERE
            date_diff('hour', from_iso8601_timestamp(p.timestamp),
                      from_iso8601_timestamp(l.timestamp)) <= {self.config.join_window_hours}
            AND contains(p.recommended_video_ids, l.video_id)
        """


class AthenaJobRunner:
    """Runner for Athena-based merge jobs.

    Uses AWS Athena for large-scale joins on S3 data.
    """

    def __init__(self, config: MergeJobConfig):
        """Initialize the Athena runner.

        Args:
            config: Merge job configuration.
        """
        self.config = config
        self._boto3_available = False

        try:
            import boto3
            self._boto3_available = True
            self._athena_client = boto3.client("athena")
        except ImportError:
            logger.warning("boto3 not available. Athena jobs will not work.")

    def run_query(self, query: str) -> str:
        """Run an Athena query.

        Args:
            query: SQL query to run.

        Returns:
            Query execution ID.
        """
        if not self._boto3_available:
            raise ImportError("boto3 is required for Athena jobs")

        response = self._athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={
                "Database": self.config.athena_database,
            },
            ResultConfiguration={
                "OutputLocation": self.config.s3_output_location,
            },
            WorkGroup=self.config.athena_workgroup,
        )

        execution_id = response["QueryExecutionId"]
        logger.info(f"Started Athena query: {execution_id}")
        return execution_id

    def wait_for_query(self, execution_id: str, timeout_seconds: int = 300) -> str:
        """Wait for a query to complete.

        Args:
            execution_id: Query execution ID.
            timeout_seconds: Maximum wait time.

        Returns:
            Query state (SUCCEEDED, FAILED, CANCELLED).
        """
        import time

        start_time = time.time()

        while True:
            response = self._athena_client.get_query_execution(
                QueryExecutionId=execution_id
            )
            state = response["QueryExecution"]["Status"]["State"]

            if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                return state

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Query {execution_id} timed out")

            time.sleep(5)
