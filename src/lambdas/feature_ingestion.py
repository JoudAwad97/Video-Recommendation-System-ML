"""
Lambda handler for feature store ingestion.

Handles batch ingestion of user and video features into DynamoDB tables.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")
USER_FEATURES_TABLE = os.environ.get("USER_FEATURES_TABLE", "")
VIDEO_FEATURES_TABLE = os.environ.get("VIDEO_FEATURES_TABLE", "")
USER_ACTIVITY_TABLE = os.environ.get("USER_ACTIVITY_TABLE", "")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for feature ingestion.

    Supports:
    - User feature batch ingestion
    - Video feature batch ingestion
    - User activity updates

    Args:
        event: Lambda event with ingestion parameters
        context: Lambda context

    Returns:
        Ingestion result with status and counts
    """
    from ..utils.logging_utils import get_logger

    logger = get_logger(__name__)

    try:
        action = event.get("action", "ingest_features")
        feature_type = event.get("feature_type", "user")

        logger.info(f"Starting feature ingestion: action={action}, type={feature_type}")

        if action == "ingest_features":
            return _ingest_features(event, feature_type, logger)
        elif action == "update_activity":
            return _update_user_activity(event, logger)
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unknown action: {action}"})
            }

    except Exception as e:
        logger.error(f"Feature ingestion failed: {traceback.format_exc()}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def _ingest_features(event: Dict[str, Any], feature_type: str, logger) -> Dict[str, Any]:
    """Ingest features into DynamoDB."""
    import boto3
    from decimal import Decimal

    records = event.get("records", [])
    if not records:
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No records to ingest", "count": 0})
        }

    dynamodb = boto3.resource("dynamodb")

    if feature_type == "user":
        table = dynamodb.Table(USER_FEATURES_TABLE)
        key_field = "user_id"
    else:
        table = dynamodb.Table(VIDEO_FEATURES_TABLE)
        key_field = "video_id"

    # Convert floats to Decimal for DynamoDB
    def convert_floats(obj):
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(i) for i in obj]
        return obj

    # Batch write
    with table.batch_writer() as batch:
        for record in records:
            item = convert_floats(record)
            batch.put_item(Item=item)

    logger.info(f"Ingested {len(records)} {feature_type} features")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Successfully ingested {feature_type} features",
            "count": len(records)
        })
    }


def _update_user_activity(event: Dict[str, Any], logger) -> Dict[str, Any]:
    """Update user activity in DynamoDB."""
    import boto3
    from datetime import datetime

    user_id = event.get("user_id")
    activity = event.get("activity", {})

    if not user_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "user_id is required"})
        }

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(USER_ACTIVITY_TABLE)

    item = {
        "user_id": str(user_id),
        "updated_at": datetime.utcnow().isoformat(),
        **activity
    }

    table.put_item(Item=item)

    logger.info(f"Updated activity for user {user_id}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Updated activity for user {user_id}"
        })
    }
