#!/usr/bin/env python
"""
Training script for CatBoost Ranker model.

Usage:
    python scripts/train_ranker.py --data-dir processed_data/ranker

Options:
    --data-dir: Directory containing processed train/val/test parquet files
    --iterations: Number of boosting iterations (default: 1000)
    --learning-rate: Learning rate (default: 0.1)
    --depth: Tree depth (default: 6)
    --model-dir: Directory to save model
    --eval-k: Comma-separated k values for evaluation (default: 5,10,20)
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.trainers import RankerTrainer
from src.models.model_config import TrainingConfig, RankerModelConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CatBoost Ranker model"
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="processed_data/ranker",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/checkpoints/ranker",
        help="Directory to save model"
    )

    # Training hyperparameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of boosting iterations"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Tree depth"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="Early stopping rounds"
    )

    # Evaluation
    parser.add_argument(
        "--eval-k",
        type=str,
        default="5,10,20",
        help="Comma-separated k values for evaluation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CatBoost Ranker Model Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Iterations: {args.iterations}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Tree depth: {args.depth}")
    print(f"Early stopping rounds: {args.early_stopping}")
    print("=" * 60)

    # Parse k values
    k_values = [int(k) for k in args.eval_k.split(",")]

    # Create configuration
    ranker_config = RankerModelConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        early_stopping_rounds=args.early_stopping,
        model_dir=args.model_dir,
    )

    config = TrainingConfig(
        ranker=ranker_config,
        processed_data_dir=os.path.dirname(args.data_dir),
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = RankerTrainer(config=config)

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = trainer.load_data(args.data_dir)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Print label distribution
    print(f"\nLabel distribution (training):")
    print(train_df["label"].value_counts())

    # Build model
    print("\nBuilding model...")
    model = trainer.build_model()
    print(f"Categorical features: {trainer.model.cat_features}")

    # Train model
    print("\nTraining model...")
    history = trainer.train(train_df, val_df)

    print("\nTraining completed!")
    print(f"Best iteration: {history.get('best_iteration', 'N/A')}")
    if 'train' in history:
        for metric, value in history['train'].items():
            print(f"  Train {metric}: {value:.4f}")
    if 'val' in history:
        for metric, value in history['val'].items():
            print(f"  Val {metric}: {value:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(
        test_df,
        k_values=k_values
    )
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Get feature importance
    print("\nTop 10 Feature Importances:")
    importance = trainer.model.get_feature_importance(top_n=10)
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")

    # Save model
    print(f"\nSaving model to {args.model_dir}...")
    trainer.save_model(args.model_dir)

    # Save training summary
    summary = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "early_stopping_rounds": args.early_stopping,
        },
        "data_stats": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_positive_rate": float(train_df["label"].mean()),
        },
        "training_history": history,
        "test_metrics": test_metrics,
        "feature_importance": importance,
    }

    summary_path = os.path.join(args.model_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to {summary_path}")
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
