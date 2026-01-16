"""
Lambda handler for prediction-label merge jobs.

Merges model predictions with ground truth labels for model evaluation and retraining.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")
INFERENCE_TABLE = os.environ.get("INFERENCE_TABLE", "")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for merge job.

    Merges predictions with ground truth labels using:
    - inference_id as join key
    - Configurable time window for label collection

    Args:
        event: Lambda event with merge job parameters
        context: Lambda context

    Returns:
        Merge result with statistics
    """
    from ..utils.logging_utils import get_logger

    logger = get_logger(__name__)

    try:
        # Get merge parameters
        start_date = event.get("start_date")
        end_date = event.get("end_date")
        join_window_hours = event.get("join_window_hours", 24)
        output_format = event.get("output_format", "parquet")

        logger.info(f"Starting merge job: {start_date} to {end_date}")

        result = _run_merge_job(
            start_date=start_date,
            end_date=end_date,
            join_window_hours=join_window_hours,
            output_format=output_format,
            logger=logger
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        logger.error(f"Merge job failed: {traceback.format_exc()}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def _run_merge_job(
    start_date: str,
    end_date: str,
    join_window_hours: int,
    output_format: str,
    logger
) -> Dict[str, Any]:
    """Run the prediction-label merge job."""
    import boto3
    from datetime import datetime

    s3 = boto3.client("s3")
    dynamodb = boto3.resource("dynamodb")

    # Load predictions from S3
    predictions = _load_predictions(s3, start_date, end_date, logger)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Load labels from DynamoDB
    labels = _load_labels(dynamodb, start_date, end_date, logger)
    logger.info(f"Loaded {len(labels)} labels")

    # Perform merge
    merged_records = _merge_predictions_labels(
        predictions, labels, join_window_hours, logger
    )
    logger.info(f"Merged {len(merged_records)} records")

    # Save merged data
    output_key = _save_merged_data(s3, merged_records, output_format, logger)

    # Calculate statistics
    positive_count = sum(1 for r in merged_records if r.get("label") == 1)
    negative_count = len(merged_records) - positive_count

    return {
        "predictions_processed": len(predictions),
        "labels_processed": len(labels),
        "records_merged": len(merged_records),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "output_path": f"s3://{DATA_BUCKET}/{output_key}",
        "completed_at": datetime.utcnow().isoformat()
    }


def _load_predictions(s3, start_date: str, end_date: str, logger) -> list:
    """Load predictions from S3."""
    predictions = []

    # List prediction files in date range
    prefix = "predictions/"
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=DATA_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Parse date from key and filter
            try:
                response = s3.get_object(Bucket=DATA_BUCKET, Key=key)
                content = response["Body"].read().decode("utf-8")
                for line in content.strip().split("\n"):
                    if line:
                        predictions.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")

    return predictions


def _load_labels(dynamodb, start_date: str, end_date: str, logger) -> list:
    """Load labels from DynamoDB."""
    table = dynamodb.Table(INFERENCE_TABLE)
    labels = []

    # Scan for labels in date range (in production, use GSI with date)
    response = table.scan(
        FilterExpression="attribute_exists(label)"
    )

    labels.extend(response.get("Items", []))

    while "LastEvaluatedKey" in response:
        response = table.scan(
            FilterExpression="attribute_exists(label)",
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        labels.extend(response.get("Items", []))

    return labels


def _merge_predictions_labels(
    predictions: list,
    labels: list,
    join_window_hours: int,
    logger
) -> list:
    """Merge predictions with labels on inference_id."""
    from datetime import datetime, timedelta

    # Index predictions by inference_id
    predictions_by_id = {}
    for pred in predictions:
        inf_id = pred.get("inference_id")
        if inf_id:
            predictions_by_id[inf_id] = pred

    # Index labels by inference_id + video_id
    labels_by_key = {}
    for label in labels:
        inf_id = label.get("inference_id")
        video_id = label.get("video_id")
        if inf_id and video_id:
            key = (inf_id, video_id)
            labels_by_key[key] = label

    # Merge
    merged = []
    for (inf_id, video_id), label in labels_by_key.items():
        pred = predictions_by_id.get(inf_id)
        if not pred:
            continue

        # Check time window
        try:
            pred_time = datetime.fromisoformat(pred.get("timestamp", ""))
            label_time = datetime.fromisoformat(label.get("timestamp", ""))
            if (label_time - pred_time) > timedelta(hours=join_window_hours):
                continue
        except (ValueError, TypeError):
            pass

        # Create merged record
        merged_record = {
            "inference_id": inf_id,
            "user_id": label.get("user_id"),
            "video_id": video_id,
            "inference_timestamp": pred.get("timestamp"),
            "feedback_timestamp": label.get("timestamp"),
            "predicted_score": pred.get("video_scores", {}).get(str(video_id), 0),
            "model_version": pred.get("model_version"),
            "label": int(label.get("label", 0)),
            "label_source": label.get("label_source", ""),
            "watch_time_seconds": float(label.get("watch_time_seconds", 0)),
            "watch_percentage": float(label.get("watch_percentage", 0)),
            "user_features": pred.get("user_features", {}),
            "context_features": pred.get("context_features", {}),
        }
        merged.append(merged_record)

    return merged


def _save_merged_data(s3, records: list, output_format: str, logger) -> str:
    """Save merged data to S3."""
    from datetime import datetime
    import io

    timestamp = datetime.utcnow()
    base_key = f"merged/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}"

    if output_format == "parquet":
        import pandas as pd
        df = pd.DataFrame(records)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        key = f"{base_key}/merged_{timestamp.strftime('%H%M%S')}.parquet"
        s3.put_object(
            Bucket=DATA_BUCKET,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream"
        )
    else:
        body = "\n".join(json.dumps(r) for r in records)
        key = f"{base_key}/merged_{timestamp.strftime('%H%M%S')}.jsonl"
        s3.put_object(
            Bucket=DATA_BUCKET,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/x-ndjson"
        )

    logger.info(f"Saved {len(records)} merged records to s3://{DATA_BUCKET}/{key}")
    return key
