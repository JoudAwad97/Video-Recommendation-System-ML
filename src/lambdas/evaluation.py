"""
Lambda handler for model evaluation.

This module provides the entry point for the evaluation Lambda function
that runs as part of the ML training pipeline.

Handler mapping for CDK: src.lambdas.evaluation.handler
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict

# Lazy import logger
_logger = None


def _get_logger():
    """Get logger with lazy import."""
    global _logger
    if _logger is None:
        from ..utils.logging_utils import get_logger
        _logger = get_logger(__name__)
    return _logger


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for model evaluation.

    This handler evaluates trained models and determines if they
    should be deployed based on quality metrics.

    Args:
        event: Step Functions event containing:
            - training_results: Results from training steps (array with Two-Tower and Ranker)
            - preprocessing: Preprocessing results
            - thresholds: Metric thresholds for deployment decision
        context: Lambda context

    Returns:
        Dictionary with evaluation results and deployment decision
    """
    logger = _get_logger()
    logger.info(f"Starting model evaluation with event: {json.dumps(event, default=str)}")

    start_time = datetime.utcnow()

    try:
        # Extract training results (from parallel execution)
        training_results = event.get("training_results", [])
        preprocessing = event.get("preprocessing", {})

        # Parse training results - handle nested structure from parallel execution
        two_tower_result = None
        ranker_result = None

        for result in training_results:
            # Check for nested training results (from Step Functions parallel execution)
            if "two_tower_training" in result:
                two_tower_result = result["two_tower_training"]
            if "ranker_training" in result:
                ranker_result = result["ranker_training"]

            # Also check for direct model_type (backwards compatibility)
            if result.get("model_type") == "two_tower":
                two_tower_result = result
            elif result.get("model_type") == "ranker":
                ranker_result = result

        logger.info(f"Parsed training results: two_tower={two_tower_result is not None}, ranker={ranker_result is not None}")

        # Get thresholds
        thresholds = event.get("thresholds", {})
        min_recall = thresholds.get("min_recall_at_k", 0.1)
        min_ndcg = thresholds.get("min_ndcg", 0.3)
        min_auc = thresholds.get("min_auc", 0.6)

        # Evaluate models
        evaluation_results = {
            "two_tower": _evaluate_two_tower(two_tower_result, min_recall, min_ndcg),
            "ranker": _evaluate_ranker(ranker_result, min_auc, min_ndcg),
        }

        # Determine deployment decision
        should_deploy = (
            evaluation_results["two_tower"]["passed"] and
            evaluation_results["ranker"]["passed"]
        )

        # If any model failed training, don't deploy
        if two_tower_result and two_tower_result.get("status") == "failed":
            should_deploy = False
            evaluation_results["two_tower"]["reason"] = "Training failed"

        if ranker_result and ranker_result.get("status") == "failed":
            should_deploy = False
            evaluation_results["ranker"]["reason"] = "Training failed"

        duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        response = {
            "status": "success",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "evaluation": evaluation_results,
            "should_deploy": should_deploy,
            "thresholds": thresholds,
            "model_versions": {
                "two_tower": two_tower_result.get("model_artifacts", {}).get("version") if two_tower_result else None,
                "ranker": ranker_result.get("model_artifacts", {}).get("version") if ranker_result else None,
            },
        }

        logger.info(
            f"Evaluation completed. Deploy decision: {should_deploy}. "
            f"Two-Tower passed: {evaluation_results['two_tower']['passed']}, "
            f"Ranker passed: {evaluation_results['ranker']['passed']}"
        )

        return response

    except Exception as e:
        logger.error(f"Evaluation failed: {traceback.format_exc()}")

        return {
            "status": "failed",
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "should_deploy": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            }
        }


def _evaluate_two_tower(
    training_result: Dict[str, Any],
    min_recall: float,
    min_ndcg: float
) -> Dict[str, Any]:
    """Evaluate Two-Tower model metrics."""
    if not training_result or training_result.get("status") == "failed":
        return {
            "passed": False,
            "reason": "Training failed or no results",
            "metrics": {},
        }

    metrics = training_result.get("metrics", {})
    recall_at_k = metrics.get("recall_at_100", metrics.get("recall_at_k", 0))
    ndcg = metrics.get("ndcg", 0)

    passed = recall_at_k >= min_recall and ndcg >= min_ndcg
    reason = None

    if not passed:
        reasons = []
        if recall_at_k < min_recall:
            reasons.append(f"recall@k ({recall_at_k:.4f}) < {min_recall}")
        if ndcg < min_ndcg:
            reasons.append(f"ndcg ({ndcg:.4f}) < {min_ndcg}")
        reason = ", ".join(reasons)

    return {
        "passed": passed,
        "reason": reason,
        "metrics": {
            "recall_at_k": recall_at_k,
            "ndcg": ndcg,
        },
        "thresholds": {
            "min_recall": min_recall,
            "min_ndcg": min_ndcg,
        },
    }


def _evaluate_ranker(
    training_result: Dict[str, Any],
    min_auc: float,
    min_ndcg: float
) -> Dict[str, Any]:
    """Evaluate Ranker model metrics."""
    if not training_result or training_result.get("status") == "failed":
        return {
            "passed": False,
            "reason": "Training failed or no results",
            "metrics": {},
        }

    metrics = training_result.get("metrics", {})
    auc = metrics.get("auc", metrics.get("roc_auc", 0))
    ndcg = metrics.get("ndcg", 0)

    passed = auc >= min_auc and ndcg >= min_ndcg
    reason = None

    if not passed:
        reasons = []
        if auc < min_auc:
            reasons.append(f"auc ({auc:.4f}) < {min_auc}")
        if ndcg < min_ndcg:
            reasons.append(f"ndcg ({ndcg:.4f}) < {min_ndcg}")
        reason = ", ".join(reasons)

    return {
        "passed": passed,
        "reason": reason,
        "metrics": {
            "auc": auc,
            "ndcg": ndcg,
        },
        "thresholds": {
            "min_auc": min_auc,
            "min_ndcg": min_ndcg,
        },
    }
