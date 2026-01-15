#!/usr/bin/env python3
"""
Entry point script for running the data processing pipeline.

Usage:
    # Run with synthetic data (for testing):
    python scripts/run_processing.py --synthetic

    # Run with real data:
    python scripts/run_processing.py --data-dir data/raw

    # Custom configuration:
    python scripts/run_processing.py --synthetic --users 5000 --videos 2000
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.processing_pipeline import ProcessingPipeline
from src.pipeline.pipeline_config import PipelineConfig
from src.data.data_loader import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the video recommendation data processing pipeline."
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate and use synthetic data for testing"
    )
    data_group.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing raw data files (users.parquet, videos.parquet, etc.)"
    )

    # Synthetic data options
    parser.add_argument(
        "--users",
        type=int,
        default=1000,
        help="Number of synthetic users (default: 1000)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=100,
        help="Number of synthetic channels (default: 100)"
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=500,
        help="Number of synthetic videos (default: 500)"
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=10000,
        help="Number of synthetic interactions (default: 10000)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Directory for processed output (default: processed_data)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory for artifacts (default: artifacts)"
    )

    # Processing options
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=3,
        help="Negative samples per positive for ranker (default: 3)"
    )
    parser.add_argument(
        "--min-watch-ratio",
        type=float,
        default=0.4,
        help="Minimum watch ratio for positive interaction (default: 0.4)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip pre-computing text embeddings"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create pipeline config
    config = PipelineConfig(
        raw_data_dir=args.data_dir or "data/raw",
        artifacts_dir=args.artifacts_dir,
        processed_data_dir=args.output_dir,
        negative_ratio=args.negative_ratio,
        min_watch_ratio=args.min_watch_ratio,
        compute_embeddings=not args.no_embeddings,
        random_seed=args.seed,
    )

    # Create pipeline
    pipeline = ProcessingPipeline(pipeline_config=config)

    # Run pipeline
    if args.synthetic:
        print(f"\nRunning pipeline with synthetic data:")
        print(f"  Users: {args.users}")
        print(f"  Channels: {args.channels}")
        print(f"  Videos: {args.videos}")
        print(f"  Interactions: {args.interactions}")
        print()

        results = pipeline.run_synthetic(
            num_users=args.users,
            num_channels=args.channels,
            num_videos=args.videos,
            num_interactions=args.interactions,
        )
    else:
        print(f"\nRunning pipeline with data from: {args.data_dir}")
        print()

        data_loader = DataLoader(args.data_dir)
        results = pipeline.run(data_loader)

    # Print summary
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)

    metadata = pipeline.get_metadata()
    print(f"\nPipeline Duration: {metadata.get('duration_seconds', 0):.2f} seconds")
    print(f"\nTwo-Tower Dataset:")
    print(f"  Train: {len(results['two_tower_train'])} samples")
    print(f"  Val:   {len(results['two_tower_val'])} samples")
    print(f"  Test:  {len(results['two_tower_test'])} samples")

    print(f"\nRanker Dataset:")
    print(f"  Train: {len(results['ranker_train'])} samples")
    print(f"  Val:   {len(results['ranker_val'])} samples")
    print(f"  Test:  {len(results['ranker_test'])} samples")

    print(f"\nOutputs saved to:")
    print(f"  Processed data: {config.processed_data_dir}/")
    print(f"  Artifacts:      {config.artifacts_dir}/")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
