#!/usr/bin/env python
"""
Training script for Two-Tower recommendation model.

Usage:
    python scripts/train_two_tower.py --data-dir processed_data/two_tower --artifacts-dir artifacts

Options:
    --data-dir: Directory containing processed train/val/test parquet files
    --artifacts-dir: Directory containing vocabularies and normalizers
    --epochs: Number of training epochs (default: 10)
    --batch-size: Training batch size (default: 256)
    --learning-rate: Learning rate (default: 0.001)
    --embedding-dim: Output embedding dimension (default: 16)
    --checkpoint-dir: Directory to save model checkpoints
    --no-embeddings: Skip loading pre-computed embeddings (faster for testing)
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.trainers import TwoTowerTrainer
from src.models.model_config import TrainingConfig, TwoTowerModelConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Two-Tower recommendation model"
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="processed_data/two_tower",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing vocabularies and normalizers"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/two_tower",
        help="Directory to save model checkpoints"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Output embedding dimension"
    )

    # Options
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip loading pre-computed embeddings"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Two-Tower Model Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Artifacts directory: {args.artifacts_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Use embeddings: {not args.no_embeddings}")
    print("=" * 60)

    # Create configuration
    two_tower_config = TwoTowerModelConfig(
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    config = TrainingConfig(
        two_tower=two_tower_config,
        processed_data_dir=os.path.dirname(args.data_dir),
        artifacts_dir=args.artifacts_dir,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = TwoTowerTrainer(
        config=config,
        artifacts_dir=args.artifacts_dir,
    )

    print(f"Vocabulary sizes: {trainer.vocab_sizes}")

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = trainer.load_data(args.data_dir)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Build model
    print("\nBuilding model...")
    model = trainer.build_model()
    print("Model architecture:")
    print(f"  User Tower hidden dims: {config.two_tower.user_tower_hidden_dims}")
    print(f"  Video Tower hidden dims: {config.two_tower.video_tower_hidden_dims}")
    print(f"  Output embedding dim: {config.two_tower.embedding_dim}")

    # Train model
    print("\nTraining model...")
    history = trainer.train(
        train_df,
        val_df,
        include_embeddings=not args.no_embeddings,
    )

    print("\nTraining completed!")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(
        test_df,
        include_embeddings=not args.no_embeddings,
    )
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save model
    print(f"\nSaving model to {args.checkpoint_dir}...")
    trainer.save_model(args.checkpoint_dir)

    # Save training summary
    summary = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        "data_stats": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
        "vocab_sizes": trainer.vocab_sizes,
        "final_metrics": {
            "train_loss": float(history['loss'][-1]),
            "val_loss": float(history['val_loss'][-1]),
            "train_accuracy": float(history['accuracy'][-1]),
            "val_accuracy": float(history['val_accuracy'][-1]),
        },
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }

    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to {summary_path}")
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
