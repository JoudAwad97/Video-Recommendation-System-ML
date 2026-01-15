"""
Data capture service for serverless endpoints.

Captures inference requests and responses for monitoring purposes.
Supports local storage, S3, and Kinesis Firehose destinations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import uuid
import time
import threading
import queue

from ..utils.logging_utils import get_logger
from .monitoring_config import DataCaptureConfig

logger = get_logger(__name__)


@dataclass
class InferenceRecord:
    """A single inference record capturing input, output, and metadata."""

    # Unique identifiers
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""

    # Timestamps
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    inference_latency_ms: float = 0.0

    # User context
    user_id: Optional[int] = None
    session_id: Optional[str] = None

    # Model info
    model_name: str = ""
    model_version: str = ""
    endpoint_name: str = ""

    # Input features
    input_features: Dict[str, Any] = field(default_factory=dict)

    # Output predictions
    output_predictions: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRecord":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "InferenceRecord":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class InferenceRecordBatch:
    """A batch of inference records."""

    records: List[InferenceRecord] = field(default_factory=list)
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def add(self, record: InferenceRecord) -> None:
        """Add a record to the batch."""
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def to_jsonl(self) -> str:
        """Convert to JSON Lines format."""
        return "\n".join(r.to_json() for r in self.records)


class DataCaptureService:
    """Service for capturing inference data.

    Supports buffering and async writing to reduce latency impact.

    Example:
        >>> service = DataCaptureService(config)
        >>> service.start()
        >>> service.capture(inference_record)
        >>> service.stop()
    """

    def __init__(self, config: DataCaptureConfig):
        """Initialize the data capture service.

        Args:
            config: Data capture configuration.
        """
        self.config = config
        self._buffer: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._records_captured = 0
        self._records_written = 0

        # Initialize storage based on config
        if config.storage_type == "local":
            self._init_local_storage()
        elif config.storage_type == "s3":
            self._init_s3_storage()
        elif config.storage_type == "kinesis":
            self._init_kinesis_storage()

    def _init_local_storage(self) -> None:
        """Initialize local file storage."""
        storage_path = Path(self.config.local_storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        self._storage_path = storage_path

    def _init_s3_storage(self) -> None:
        """Initialize S3 storage."""
        try:
            import boto3
            self._s3_client = boto3.client("s3")
        except ImportError:
            logger.warning("boto3 not available, falling back to local storage")
            self.config.storage_type = "local"
            self._init_local_storage()

    def _init_kinesis_storage(self) -> None:
        """Initialize Kinesis Firehose storage."""
        try:
            import boto3
            self._firehose_client = boto3.client("firehose")
        except ImportError:
            logger.warning("boto3 not available, falling back to local storage")
            self.config.storage_type = "local"
            self._init_local_storage()

    def start(self) -> None:
        """Start the data capture service."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop)
        self._writer_thread.daemon = True
        self._writer_thread.start()
        self._is_running = True

        logger.info("Data capture service started")

    def stop(self) -> None:
        """Stop the data capture service and flush remaining records."""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=10)
        self._is_running = False

        # Flush remaining records
        self._flush_buffer()

        logger.info(
            f"Data capture service stopped. "
            f"Captured: {self._records_captured}, Written: {self._records_written}"
        )

    def capture(
        self,
        request_id: str,
        input_features: Dict[str, Any],
        output_predictions: Dict[str, Any],
        latency_ms: float,
        user_id: Optional[int] = None,
        model_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Capture an inference record.

        Args:
            request_id: Request identifier.
            input_features: Input features dict.
            output_predictions: Model predictions dict.
            latency_ms: Inference latency in milliseconds.
            user_id: Optional user identifier.
            model_version: Model version string.
            metadata: Additional metadata.

        Returns:
            Record ID if captured, None if skipped.
        """
        if not self.config.enable_capture:
            return None

        # Apply sampling
        import random
        if random.random() > self.config.capture_percentage:
            return None

        record = InferenceRecord(
            request_id=request_id,
            inference_latency_ms=latency_ms,
            user_id=user_id,
            model_version=model_version,
            input_features=input_features if self.config.capture_inputs else {},
            output_predictions=output_predictions if self.config.capture_outputs else {},
            metadata=metadata or {},
        )

        self._buffer.put(record)
        self._records_captured += 1

        return record.record_id

    def capture_from_response(
        self,
        request_id: str,
        user_data: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        latency_ms: float,
        model_version: str = "",
        stage_latencies: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Capture from a recommendation response.

        Convenience method for capturing recommendation pipeline outputs.

        Args:
            request_id: Request identifier.
            user_data: User features.
            recommendations: List of recommendation dicts.
            latency_ms: Total latency.
            model_version: Model version.
            stage_latencies: Per-stage latencies.

        Returns:
            Record ID if captured.
        """
        return self.capture(
            request_id=request_id,
            input_features=user_data,
            output_predictions={
                "recommendations": recommendations,
                "num_recommendations": len(recommendations),
            },
            latency_ms=latency_ms,
            user_id=user_data.get("user_id") or user_data.get("id"),
            model_version=model_version,
            metadata={
                "stage_latencies": stage_latencies or {},
            },
        )

    def _writer_loop(self) -> None:
        """Background thread for writing records."""
        batch = InferenceRecordBatch()
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                # Get record with timeout
                record = self._buffer.get(timeout=1.0)
                batch.add(record)

                # Check if we should flush
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    time.time() - last_flush >= self.config.flush_interval_seconds
                )

                if should_flush and len(batch) > 0:
                    self._write_batch(batch)
                    batch = InferenceRecordBatch()
                    last_flush = time.time()

            except queue.Empty:
                # Check time-based flush
                if time.time() - last_flush >= self.config.flush_interval_seconds:
                    if len(batch) > 0:
                        self._write_batch(batch)
                        batch = InferenceRecordBatch()
                        last_flush = time.time()

    def _flush_buffer(self) -> None:
        """Flush all remaining records from buffer."""
        batch = InferenceRecordBatch()

        while not self._buffer.empty():
            try:
                record = self._buffer.get_nowait()
                batch.add(record)
            except queue.Empty:
                break

        if len(batch) > 0:
            self._write_batch(batch)

    def _write_batch(self, batch: InferenceRecordBatch) -> None:
        """Write a batch of records to storage.

        Args:
            batch: Batch of inference records.
        """
        try:
            if self.config.storage_type == "local":
                self._write_local(batch)
            elif self.config.storage_type == "s3":
                self._write_s3(batch)
            elif self.config.storage_type == "kinesis":
                self._write_kinesis(batch)

            self._records_written += len(batch)

        except Exception as e:
            logger.error(f"Failed to write batch: {e}")

    def _write_local(self, batch: InferenceRecordBatch) -> None:
        """Write batch to local storage."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_{timestamp}_{batch.batch_id[:8]}.jsonl"
        filepath = self._storage_path / filename

        with open(filepath, "w") as f:
            f.write(batch.to_jsonl())

        logger.debug(f"Wrote {len(batch)} records to {filepath}")

    def _write_s3(self, batch: InferenceRecordBatch) -> None:
        """Write batch to S3."""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H")
        key = f"{self.config.s3_prefix}/{timestamp}/{batch.batch_id}.jsonl"

        self._s3_client.put_object(
            Bucket=self.config.s3_bucket,
            Key=key,
            Body=batch.to_jsonl().encode("utf-8"),
        )

        logger.debug(f"Wrote {len(batch)} records to s3://{self.config.s3_bucket}/{key}")

    def _write_kinesis(self, batch: InferenceRecordBatch) -> None:
        """Write batch to Kinesis Firehose."""
        records = [
            {"Data": (r.to_json() + "\n").encode("utf-8")}
            for r in batch.records
        ]

        self._firehose_client.put_record_batch(
            DeliveryStreamName=self.config.kinesis_stream_name,
            Records=records,
        )

        logger.debug(f"Wrote {len(batch)} records to Kinesis Firehose")

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics.

        Returns:
            Dictionary with capture stats.
        """
        return {
            "is_running": self._is_running,
            "records_captured": self._records_captured,
            "records_written": self._records_written,
            "buffer_size": self._buffer.qsize(),
            "storage_type": self.config.storage_type,
        }

    def load_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[InferenceRecord]:
        """Load captured records from storage.

        Args:
            start_time: Filter records after this time.
            end_time: Filter records before this time.
            limit: Maximum number of records to load.

        Returns:
            List of inference records.
        """
        if self.config.storage_type != "local":
            raise NotImplementedError("Only local storage loading is implemented")

        records = []

        for filepath in sorted(self._storage_path.glob("*.jsonl")):
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        record = InferenceRecord.from_json(line)

                        # Apply time filters
                        record_time = datetime.fromisoformat(record.timestamp)
                        if start_time and record_time < start_time:
                            continue
                        if end_time and record_time > end_time:
                            continue

                        records.append(record)

                        if limit and len(records) >= limit:
                            return records

        return records
