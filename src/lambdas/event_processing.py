"""
Lambda handler for event processing (feedback loop).

Processes user interaction events from Kinesis and stores them for model retraining.
"""

from __future__ import annotations

import base64
import json
import os
import traceback
from typing import Any, Dict, List

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")
INFERENCE_TABLE = os.environ.get("INFERENCE_TABLE", "")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for Kinesis event processing.

    Processes user interaction events and stores them for:
    - Ground truth label collection
    - Model evaluation
    - Retraining data generation

    Args:
        event: Kinesis event with base64-encoded records
        context: Lambda context

    Returns:
        Processing result with success/failure counts
    """
    from ..utils.logging_utils import get_logger

    logger = get_logger(__name__)

    records = event.get("Records", [])
    logger.info(f"Processing {len(records)} Kinesis records")

    success_count = 0
    failure_count = 0
    processed_events = []

    for record in records:
        try:
            # Decode Kinesis record
            payload = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
            interaction = json.loads(payload)

            # Process the interaction
            processed = _process_interaction(interaction, logger)
            if processed:
                processed_events.append(processed)
                success_count += 1
            else:
                failure_count += 1

        except Exception as e:
            logger.error(f"Failed to process record: {e}")
            failure_count += 1

    # Batch store processed events
    if processed_events:
        _store_events(processed_events, logger)

    logger.info(f"Processed {success_count} events, {failure_count} failures")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": success_count,
            "failed": failure_count
        })
    }


def _process_interaction(interaction: Dict[str, Any], logger) -> Dict[str, Any] | None:
    """Process a single user interaction event.

    Extracts relevant fields and determines the label based on interaction type.
    """
    from datetime import datetime

    try:
        event_type = interaction.get("event_type", "unknown")
        user_id = interaction.get("user_id")
        video_id = interaction.get("video_id")
        inference_id = interaction.get("inference_id")

        if not all([user_id, video_id]):
            logger.warning(f"Missing required fields in interaction: {interaction}")
            return None

        # Determine label based on event type
        label = 0
        label_source = event_type

        if event_type == "like":
            label = 1
        elif event_type == "comment":
            label = 1
        elif event_type == "watch":
            watch_percentage = interaction.get("watch_percentage", 0)
            if watch_percentage >= 0.4:  # 40% watch threshold
                label = 1
        elif event_type == "impression":
            label = 1
        elif event_type == "skip":
            label = 0
        elif event_type == "dislike":
            label = 0

        return {
            "inference_id": inference_id,
            "user_id": user_id,
            "video_id": video_id,
            "event_type": event_type,
            "label": label,
            "label_source": label_source,
            "watch_time_seconds": interaction.get("watch_time_seconds", 0),
            "watch_percentage": interaction.get("watch_percentage", 0),
            "position_shown": interaction.get("position", 0),
            "timestamp": interaction.get("timestamp", datetime.utcnow().isoformat()),
            "device": interaction.get("device", "unknown"),
            "processed_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        return None


def _store_events(events: List[Dict[str, Any]], logger) -> None:
    """Store processed events to DynamoDB and S3."""
    import boto3
    from datetime import datetime
    from decimal import Decimal

    # Store to DynamoDB for quick lookup
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(INFERENCE_TABLE)

    def convert_floats(obj):
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(i) for i in obj]
        return obj

    with table.batch_writer() as batch:
        for event in events:
            item = convert_floats(event)
            # Use composite key for DynamoDB
            item["pk"] = f"USER#{event['user_id']}"
            item["sk"] = f"EVENT#{event['timestamp']}#{event['video_id']}"
            batch.put_item(Item=item)

    # Also store to S3 for batch processing
    s3 = boto3.client("s3")
    timestamp = datetime.utcnow()
    key = f"feedback/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/events_{timestamp.strftime('%H%M%S')}.jsonl"

    body = "\n".join(json.dumps(e) for e in events)
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/x-ndjson"
    )

    logger.info(f"Stored {len(events)} events to DynamoDB and S3")
