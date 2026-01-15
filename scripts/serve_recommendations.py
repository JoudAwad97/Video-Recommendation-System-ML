#!/usr/bin/env python3
"""
Script for running the video recommendation serving pipeline.

This script demonstrates how to:
1. Initialize the recommendation orchestrator
2. Load models and video embeddings
3. Generate recommendations for users

Usage:
    python scripts/serve_recommendations.py --model-path models/two_tower --artifacts-path artifacts

    # With synthetic data for testing
    python scripts/serve_recommendations.py --generate-synthetic --num-videos 1000
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.serving import (
    ServingConfig,
    VectorDBConfig,
    RecommendationOrchestrator,
    RecommendationRequest,
    OfflineInferencePipeline,
)
from src.data.synthetic_generator import SyntheticDataGenerator
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the video recommendation serving pipeline"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/two_tower",
        help="Path to Two-Tower model directory",
    )
    parser.add_argument(
        "--ranker-path",
        type=str,
        default="models/ranker",
        help="Path to Ranker model directory",
    )
    parser.add_argument(
        "--artifacts-path",
        type=str,
        default="artifacts",
        help="Path to feature engineering artifacts",
    )
    parser.add_argument(
        "--vector-index-path",
        type=str,
        default=None,
        help="Path to pre-built vector index",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic data for testing",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=500,
        help="Number of synthetic videos to generate",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=100,
        help="Number of synthetic users to generate",
    )
    parser.add_argument(
        "--num-recommendations",
        type=int,
        default=20,
        help="Number of recommendations to return",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode",
    )
    parser.add_argument(
        "--benchmark-queries",
        type=int,
        default=100,
        help="Number of queries for benchmarking",
    )

    return parser.parse_args()


def generate_synthetic_data(num_users: int, num_videos: int):
    """Generate synthetic data for testing.

    Args:
        num_users: Number of users to generate.
        num_videos: Number of videos to generate.

    Returns:
        Tuple of (users_df, videos_df, video_data_dict).
    """
    logger.info(f"Generating synthetic data: {num_users} users, {num_videos} videos")

    generator = SyntheticDataGenerator(seed=42)

    users_df = generator.generate_users(num_users)
    videos_df = generator.generate_videos(num_videos)

    # Convert videos to dictionary format for serving
    video_data = {}
    for _, row in videos_df.iterrows():
        video_data[row["id"]] = {
            "category": row["category"],
            "child_categories": row.get("child_categories", row["category"]),
            "language": row["language"],
            "duration": row["duration"],
            "popularity": row["popularity"],
            "view_count": row["view_count"],
            "like_count": row["like_count"],
            "comment_count": row["comment_count"],
            "channel_subscriber_count": row.get("channel_subscriber_count", 10000),
            "is_active": True,
        }

    return users_df, videos_df, video_data


def generate_random_embeddings(num_videos: int, embedding_dim: int):
    """Generate random embeddings for testing.

    Args:
        num_videos: Number of videos.
        embedding_dim: Embedding dimension.

    Returns:
        Tuple of (video_ids, embeddings).
    """
    video_ids = list(range(1, num_videos + 1))
    embeddings = np.random.randn(num_videos, embedding_dim).astype(np.float32)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return video_ids, embeddings


def run_demo(orchestrator, users_df, num_recommendations: int):
    """Run demo recommendations for sample users.

    Args:
        orchestrator: Initialized orchestrator.
        users_df: Users DataFrame.
        num_recommendations: Number of recommendations.
    """
    logger.info("Running demo recommendations...")

    # Sample a few users
    sample_users = users_df.sample(min(5, len(users_df)))

    for _, user_row in sample_users.iterrows():
        user_data = {
            "id": user_row["id"],
            "country_code": user_row["country_code"],
            "preferred_language": user_row["preferred_language"],
            "age": user_row["age"],
            "previously_watched_category": "Technology",
        }

        request = RecommendationRequest(
            user_data=user_data,
            num_recommendations=num_recommendations,
            enable_diversification=True,
        )

        response = orchestrator.recommend(request)

        print(f"\n{'='*60}")
        print(f"Recommendations for User {user_data['id']}:")
        print(f"  Country: {user_data['country_code']}")
        print(f"  Language: {user_data['preferred_language']}")
        print(f"  Age: {user_data['age']}")
        print(f"  Latency: {response.latency_ms:.2f}ms")
        print(f"  Stage latencies: {response.stage_latencies}")
        print(f"\nTop {len(response.recommendations)} recommendations:")

        for rec in response.recommendations[:10]:
            print(f"  {rec.rank}. Video {rec.video_id} (score: {rec.score:.4f})")


def run_benchmark(orchestrator, users_df, num_queries: int, num_recommendations: int):
    """Run benchmark to measure throughput and latency.

    Args:
        orchestrator: Initialized orchestrator.
        users_df: Users DataFrame.
        num_queries: Number of queries to run.
        num_recommendations: Number of recommendations per query.
    """
    logger.info(f"Running benchmark with {num_queries} queries...")

    # Sample users with replacement for benchmark
    sample_users = users_df.sample(num_queries, replace=True)

    latencies = []
    stage_latencies = {"retrieval": [], "ranking": [], "ordering": []}

    start_time = time.time()

    for _, user_row in sample_users.iterrows():
        user_data = {
            "id": user_row["id"],
            "country_code": user_row["country_code"],
            "preferred_language": user_row["preferred_language"],
            "age": user_row["age"],
            "previously_watched_category": "Technology",
        }

        request = RecommendationRequest(
            user_data=user_data,
            num_recommendations=num_recommendations,
        )

        response = orchestrator.recommend(request)

        latencies.append(response.latency_ms)
        for stage, latency in response.stage_latencies.items():
            if stage in stage_latencies:
                stage_latencies[stage].append(latency)

    total_time = time.time() - start_time

    # Compute statistics
    latencies = np.array(latencies)

    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total queries: {num_queries}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_queries / total_time:.2f} queries/s")
    print(f"\nLatency Statistics (ms):")
    print(f"  Mean: {np.mean(latencies):.2f}")
    print(f"  Median: {np.median(latencies):.2f}")
    print(f"  P95: {np.percentile(latencies, 95):.2f}")
    print(f"  P99: {np.percentile(latencies, 99):.2f}")
    print(f"  Min: {np.min(latencies):.2f}")
    print(f"  Max: {np.max(latencies):.2f}")

    print(f"\nStage Latencies (ms):")
    for stage, values in stage_latencies.items():
        if values:
            print(f"  {stage}: mean={np.mean(values):.2f}, p95={np.percentile(values, 95):.2f}")


def main():
    """Main function."""
    args = parse_args()

    # Create configuration
    vector_config = VectorDBConfig(
        store_type="memory",  # Use in-memory for testing
        embedding_dim=args.embedding_dim,
    )

    config = ServingConfig(
        two_tower_model_path=args.model_path,
        ranker_model_path=args.ranker_path,
        artifacts_path=args.artifacts_path,
        vector_db=vector_config,
        num_candidates=100,
        top_k_final=args.num_recommendations,
    )

    # Generate or load data
    if args.generate_synthetic:
        users_df, videos_df, video_data = generate_synthetic_data(
            args.num_users,
            args.num_videos,
        )
        video_ids, embeddings = generate_random_embeddings(
            args.num_videos,
            args.embedding_dim,
        )
    else:
        # In production, load real data
        logger.warning("No data specified. Use --generate-synthetic for testing.")
        return

    # Initialize orchestrator
    logger.info("Initializing recommendation orchestrator...")
    orchestrator = RecommendationOrchestrator(config)

    # For testing without trained models, we'll use a simplified setup
    # In production, you would call orchestrator.initialize()
    from src.serving.vector_store import InMemoryVectorStore
    from src.serving.query_encoder import QueryEncoderService
    from src.serving.candidate_retrieval import CandidateRetrievalService

    # Set up vector store
    vector_store = InMemoryVectorStore(vector_config)
    vector_store.add(video_ids, embeddings)

    # For demo purposes, create a mock query encoder that returns random embeddings
    class MockQueryEncoder:
        def encode_user(self, user_data):
            embedding = np.random.randn(args.embedding_dim).astype(np.float32)
            return embedding / (np.linalg.norm(embedding) + 1e-8)

        def encode_users_batch(self, users_data):
            embeddings = np.random.randn(len(users_data), args.embedding_dim).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / (norms + 1e-8)

    # Set up retrieval service
    retrieval_service = CandidateRetrievalService(config)
    retrieval_service.set_query_encoder(MockQueryEncoder())
    retrieval_service.set_vector_store(vector_store)
    retrieval_service.load_video_metadata(video_data)

    # Set orchestrator components
    orchestrator.vector_store = vector_store
    orchestrator.retrieval_service = retrieval_service
    orchestrator.video_data = video_data
    orchestrator._is_initialized = True

    logger.info(f"Loaded {len(video_ids)} videos into vector store")

    # Run demo or benchmark
    if args.benchmark:
        run_benchmark(
            orchestrator,
            users_df,
            args.benchmark_queries,
            args.num_recommendations,
        )
    else:
        run_demo(orchestrator, users_df, args.num_recommendations)

    # Print final stats
    print("\n" + "="*60)
    print("ORCHESTRATOR STATS")
    print("="*60)
    stats = orchestrator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
