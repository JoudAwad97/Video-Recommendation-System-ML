#!/usr/bin/env python3
"""
Bootstrap script for populating the recommendation system with synthetic data.

This script:
1. Generates synthetic users, videos, channels, and interactions
2. Populates DynamoDB feature tables (user and video features)
3. Generates video embeddings using a simple embedding model
4. Indexes embeddings in the vector store
5. Tests the end-to-end recommendation flow

Usage:
    python scripts/bootstrap_data.py --num-users 1000 --num-videos 500

For local testing:
    python scripts/bootstrap_data.py --local --num-users 100 --num-videos 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataBootstrapper:
    """Bootstraps the recommendation system with synthetic data."""

    def __init__(
        self,
        num_users: int = 1000,
        num_channels: int = 100,
        num_videos: int = 500,
        num_interactions: int = 10000,
        embedding_dim: int = 16,
        local_mode: bool = False,
        seed: int = 42,
    ):
        """Initialize the bootstrapper.

        Args:
            num_users: Number of users to generate
            num_channels: Number of channels to generate
            num_videos: Number of videos to generate
            num_interactions: Number of interactions to generate
            embedding_dim: Dimension of video embeddings
            local_mode: If True, use local storage instead of AWS
            seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.num_channels = num_channels
        self.num_videos = num_videos
        self.num_interactions = num_interactions
        self.embedding_dim = embedding_dim
        self.local_mode = local_mode
        self.seed = seed

        # Initialize generator
        self.generator = SyntheticDataGenerator(seed=seed)

        # Data storage
        self.users_df = None
        self.channels_df = None
        self.videos_df = None
        self.interactions_df = None
        self.video_embeddings = None

        # AWS clients (initialized if not local mode)
        self._dynamodb = None
        self._lambda_client = None
        self._cfn_outputs = None  # Cache for CloudFormation outputs

    def _get_cfn_output(self, output_key: str) -> Optional[str]:
        """Get a CloudFormation output value from the stack.

        Args:
            output_key: The key of the output to retrieve

        Returns:
            The output value, or None if not found
        """
        import boto3

        if self._cfn_outputs is None:
            try:
                cfn = boto3.client("cloudformation", region_name="us-east-2")
                stack_name = os.environ.get("STACK_NAME", "VideoRecSystem-dev")
                response = cfn.describe_stacks(StackName=stack_name)

                if response["Stacks"]:
                    outputs = response["Stacks"][0].get("Outputs", [])
                    self._cfn_outputs = {
                        o["OutputKey"]: o["OutputValue"] for o in outputs
                    }
                else:
                    self._cfn_outputs = {}
            except Exception as e:
                logger.warning(f"Could not fetch CloudFormation outputs: {e}")
                self._cfn_outputs = {}

        return self._cfn_outputs.get(output_key)

    def run(self) -> Dict[str, Any]:
        """Run the full bootstrap process.

        Returns:
            Dictionary with bootstrap results and statistics.
        """
        logger.info("=" * 60)
        logger.info("Starting data bootstrap process")
        logger.info("=" * 60)

        results = {
            "started_at": datetime.utcnow().isoformat(),
            "config": {
                "num_users": self.num_users,
                "num_videos": self.num_videos,
                "num_channels": self.num_channels,
                "num_interactions": self.num_interactions,
                "embedding_dim": self.embedding_dim,
                "local_mode": self.local_mode,
            },
            "steps": {},
        }

        try:
            # Step 1: Generate synthetic data
            logger.info("\n[Step 1/5] Generating synthetic data...")
            self._generate_data()
            results["steps"]["data_generation"] = {
                "status": "success",
                "users": len(self.users_df),
                "videos": len(self.videos_df),
                "channels": len(self.channels_df),
                "interactions": len(self.interactions_df),
            }

            # Step 2: Generate embeddings
            logger.info("\n[Step 2/5] Generating video embeddings...")
            self._generate_embeddings()
            results["steps"]["embedding_generation"] = {
                "status": "success",
                "num_embeddings": len(self.video_embeddings),
                "embedding_dim": self.embedding_dim,
            }

            # Step 3: Populate feature stores
            logger.info("\n[Step 3/5] Populating feature stores...")
            feature_result = self._populate_feature_stores()
            results["steps"]["feature_store_population"] = feature_result

            # Step 4: Index embeddings in vector store
            logger.info("\n[Step 4/5] Indexing embeddings in vector store...")
            vector_result = self._index_embeddings()
            results["steps"]["vector_store_indexing"] = vector_result

            # Step 5: Test recommendations
            logger.info("\n[Step 5/5] Testing recommendation endpoint...")
            test_result = self._test_recommendations()
            results["steps"]["recommendation_test"] = test_result

            results["status"] = "success"
            results["completed_at"] = datetime.utcnow().isoformat()

        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        logger.info("\n" + "=" * 60)
        logger.info("Bootstrap process completed!")
        logger.info("=" * 60)

        return results

    def _generate_data(self) -> None:
        """Generate synthetic data using the data generator."""
        data = self.generator.generate_all(
            num_users=self.num_users,
            num_channels=self.num_channels,
            num_videos=self.num_videos,
            num_interactions=self.num_interactions,
        )

        self.users_df = data["users"]
        self.channels_df = data["channels"]
        self.videos_df = data["videos"]
        self.interactions_df = data["interactions"]

        logger.info(f"  Generated {len(self.users_df)} users")
        logger.info(f"  Generated {len(self.channels_df)} channels")
        logger.info(f"  Generated {len(self.videos_df)} videos")
        logger.info(f"  Generated {len(self.interactions_df)} interactions")

        # Save to local files for reference
        output_dir = Path("data/synthetic")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.users_df.to_parquet(output_dir / "users.parquet", index=False)
        self.videos_df.to_parquet(output_dir / "videos.parquet", index=False)
        self.channels_df.to_parquet(output_dir / "channels.parquet", index=False)
        self.interactions_df.to_parquet(output_dir / "interactions.parquet", index=False)

        logger.info(f"  Saved data to {output_dir}")

    def _generate_embeddings(self) -> None:
        """Generate video embeddings.

        Uses a simple embedding approach based on video features:
        - Category is embedded
        - Numeric features (views, likes, duration) contribute to embedding
        - Tags contribute through simple hashing

        In production, this would use the trained Two-Tower model.
        """
        np.random.seed(self.seed)

        # Category embedding lookup
        categories = self.videos_df["category"].unique()
        category_embeddings = {
            cat: np.random.randn(self.embedding_dim // 2)
            for cat in categories
        }

        embeddings = []
        video_ids = []

        for _, video in self.videos_df.iterrows():
            video_id = video["id"]
            category = video["category"]

            # Start with category embedding
            cat_emb = category_embeddings[category]

            # Add feature-based component
            # Normalize numeric features
            duration_norm = np.log1p(video["duration"]) / 10
            views_norm = np.log1p(video["view_count"]) / 20
            likes_norm = np.log1p(video["like_count"]) / 15

            # Create feature embedding
            feature_emb = np.random.randn(self.embedding_dim // 2) * 0.1
            feature_emb[0] = duration_norm
            feature_emb[1] = views_norm
            feature_emb[2] = likes_norm

            # Combine category and feature embeddings
            embedding = np.concatenate([cat_emb, feature_emb])

            # Normalize to unit length
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            embeddings.append(embedding)
            video_ids.append(video_id)

        self.video_embeddings = {
            "video_ids": video_ids,
            "embeddings": np.array(embeddings),
        }

        logger.info(f"  Generated {len(embeddings)} embeddings")
        logger.info(f"  Embedding shape: {self.video_embeddings['embeddings'].shape}")

        # Save embeddings
        output_dir = Path("data/embeddings")
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_dir / "video_embeddings.npz",
            video_ids=np.array(video_ids),
            embeddings=self.video_embeddings["embeddings"],
        )
        logger.info(f"  Saved embeddings to {output_dir}/video_embeddings.npz")

    def _populate_feature_stores(self) -> Dict[str, Any]:
        """Populate DynamoDB feature tables with user and video features."""
        if self.local_mode:
            return self._populate_local_feature_store()
        else:
            return self._populate_dynamodb_feature_store()

    def _populate_local_feature_store(self) -> Dict[str, Any]:
        """Populate local feature store files."""
        output_dir = Path("data/feature_store")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare user features
        user_features = {}
        for _, user in self.users_df.iterrows():
            user_features[int(user["id"])] = {
                "user_id": int(user["id"]),
                "country": user["country_code"],
                "user_language": user["preferred_language"],
                "age": int(user["age"]),
            }

        # Prepare video features (merge with channel data)
        channel_subscribers = dict(zip(
            self.channels_df["id"],
            self.channels_df["subscriber_count"]
        ))

        video_features = {}
        for _, video in self.videos_df.iterrows():
            video_features[int(video["id"])] = {
                "video_id": int(video["id"]),
                "category": video["category"],
                "video_language": video["language"],
                "video_duration": int(video["duration"]),
                "popularity": video["popularity"],
                "view_count": int(video["view_count"]),
                "like_count": int(video["like_count"]),
                "comment_count": int(video["comment_count"]),
                "channel_subscriber_count": int(channel_subscribers.get(video["channel_id"], 0)),
            }

        # Save to JSON files
        with open(output_dir / "user_features.json", "w") as f:
            json.dump(user_features, f)

        with open(output_dir / "video_features.json", "w") as f:
            json.dump(video_features, f)

        logger.info(f"  Saved {len(user_features)} user features to local store")
        logger.info(f"  Saved {len(video_features)} video features to local store")

        return {
            "status": "success",
            "mode": "local",
            "users_ingested": len(user_features),
            "videos_ingested": len(video_features),
        }

    def _populate_dynamodb_feature_store(self) -> Dict[str, Any]:
        """Populate DynamoDB tables with features."""
        import boto3

        # Get table names from environment
        user_table_name = os.environ.get(
            "USER_FEATURES_TABLE",
            "VideoRecSystem-dev-user-features"
        )
        video_table_name = os.environ.get(
            "VIDEO_FEATURES_TABLE",
            "VideoRecSystem-dev-video-features"
        )

        dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
        user_table = dynamodb.Table(user_table_name)
        video_table = dynamodb.Table(video_table_name)

        # Prepare channel subscriber lookup
        channel_subscribers = dict(zip(
            self.channels_df["id"],
            self.channels_df["subscriber_count"]
        ))

        timestamp = datetime.utcnow().isoformat()

        # Ingest user features
        logger.info(f"  Ingesting {len(self.users_df)} users to {user_table_name}...")
        user_count = 0
        with user_table.batch_writer() as batch:
            for _, user in self.users_df.iterrows():
                item = {
                    "user_id": int(user["id"]),
                    "country": user["country_code"],
                    "user_language": user["preferred_language"],
                    "age": int(user["age"]),
                    "updated_at": timestamp,
                }
                batch.put_item(Item=item)
                user_count += 1

                if user_count % 100 == 0:
                    logger.info(f"    Ingested {user_count} users...")

        # Ingest video features
        logger.info(f"  Ingesting {len(self.videos_df)} videos to {video_table_name}...")
        video_count = 0
        with video_table.batch_writer() as batch:
            for _, video in self.videos_df.iterrows():
                item = {
                    "video_id": int(video["id"]),
                    "category": video["category"],
                    "video_language": video["language"],
                    "video_duration": int(video["duration"]),
                    "popularity": video["popularity"],
                    "view_count": int(video["view_count"]),
                    "like_count": int(video["like_count"]),
                    "comment_count": int(video["comment_count"]),
                    "channel_subscriber_count": int(channel_subscribers.get(video["channel_id"], 0)),
                    "updated_at": timestamp,
                }
                batch.put_item(Item=item)
                video_count += 1

                if video_count % 100 == 0:
                    logger.info(f"    Ingested {video_count} videos...")

        logger.info(f"  Successfully ingested {user_count} users and {video_count} videos")

        return {
            "status": "success",
            "mode": "dynamodb",
            "users_ingested": user_count,
            "videos_ingested": video_count,
            "user_table": user_table_name,
            "video_table": video_table_name,
        }

    def _index_embeddings(self) -> Dict[str, Any]:
        """Index video embeddings in the vector store."""
        if self.local_mode:
            return self._index_embeddings_local()
        else:
            return self._index_embeddings_s3()

    def _index_embeddings_local(self) -> Dict[str, Any]:
        """Index embeddings using local FAISS store."""
        from src.serving.vector_store import InMemoryVectorStore
        from src.serving.serving_config import VectorDBConfig

        config = VectorDBConfig(
            embedding_dim=self.embedding_dim,
            index_type="Flat",
        )

        store = InMemoryVectorStore(config)

        # Add embeddings
        video_ids = self.video_embeddings["video_ids"]
        embeddings = self.video_embeddings["embeddings"]

        store.add(
            ids=video_ids,
            embeddings=embeddings,
        )

        # Save to disk
        output_dir = Path("data/vector_store")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy for simple loading
        np.savez(
            output_dir / "vector_index.npz",
            video_ids=np.array(video_ids),
            embeddings=embeddings,
        )

        logger.info(f"  Indexed {len(video_ids)} videos in local vector store")
        logger.info(f"  Saved vector index to {output_dir}/vector_index.npz")

        # Test search
        test_query = embeddings[0:1]  # First video embedding
        results, scores = store.search(test_query[0], top_k=5)
        logger.info(f"  Test search returned {len(results)} results")

        return {
            "status": "success",
            "mode": "local",
            "videos_indexed": len(video_ids),
            "index_path": str(output_dir / "vector_index.npz"),
        }

    def _index_embeddings_s3(self) -> Dict[str, Any]:
        """Index embeddings and upload to S3 for Lambda to use."""
        import boto3

        # Get bucket name from environment or CloudFormation
        model_bucket = os.environ.get("MODEL_BUCKET")
        if not model_bucket:
            model_bucket = self._get_cfn_output("ModelBucketName")
        if not model_bucket:
            raise ValueError("MODEL_BUCKET not set and could not fetch from CloudFormation")

        # Create local index first
        output_dir = Path("data/vector_store")
        output_dir.mkdir(parents=True, exist_ok=True)

        video_ids = self.video_embeddings["video_ids"]
        embeddings = self.video_embeddings["embeddings"]

        # Save as numpy
        local_path = output_dir / "vector_index.npz"
        np.savez(
            local_path,
            video_ids=np.array(video_ids),
            embeddings=embeddings,
        )

        # Upload to S3
        s3 = boto3.client("s3", region_name="us-east-2")
        s3_key = "vector_store/vector_index.npz"

        logger.info(f"  Uploading vector index to s3://{model_bucket}/{s3_key}...")
        s3.upload_file(str(local_path), model_bucket, s3_key)

        # Also upload video features for the ranker
        features_path = output_dir / "video_features_for_ranking.json"

        # Prepare video features for ranking
        channel_subscribers = dict(zip(
            self.channels_df["id"],
            self.channels_df["subscriber_count"]
        ))

        video_features = {}
        for _, video in self.videos_df.iterrows():
            video_features[int(video["id"])] = {
                "video_id": int(video["id"]),
                "category": video["category"],
                "video_language": video["language"],
                "video_duration": int(video["duration"]),
                "popularity": video["popularity"],
                "view_count": int(video["view_count"]),
                "like_count": int(video["like_count"]),
                "comment_count": int(video["comment_count"]),
                "channel_subscriber_count": int(channel_subscribers.get(video["channel_id"], 0)),
            }

        with open(features_path, "w") as f:
            json.dump(video_features, f)

        s3.upload_file(str(features_path), model_bucket, "vector_store/video_features.json")

        logger.info(f"  Indexed {len(video_ids)} videos")
        logger.info(f"  Uploaded to s3://{model_bucket}/vector_store/")

        return {
            "status": "success",
            "mode": "s3",
            "videos_indexed": len(video_ids),
            "s3_bucket": model_bucket,
            "s3_key": s3_key,
        }

    def _test_recommendations(self) -> Dict[str, Any]:
        """Test the recommendation endpoint."""
        if self.local_mode:
            return self._test_local_recommendations()
        else:
            return self._test_api_recommendations()

    def _test_local_recommendations(self) -> Dict[str, Any]:
        """Test recommendations using local components."""
        from src.serving.vector_store import InMemoryVectorStore
        from src.serving.serving_config import VectorDBConfig

        logger.info("  Testing local recommendation flow...")

        # Load embeddings
        embeddings_path = Path("data/embeddings/video_embeddings.npz")
        data = np.load(embeddings_path)
        video_ids = data["video_ids"].tolist()
        embeddings = data["embeddings"]

        # Create vector store
        config = VectorDBConfig(embedding_dim=self.embedding_dim)
        store = InMemoryVectorStore(config)
        store.add(ids=video_ids, embeddings=embeddings)

        # Test with a few users
        test_users = [1, 5, 10]
        results = []

        for user_id in test_users:
            # Simulate user embedding (in production, this comes from user features)
            np.random.seed(user_id)
            user_embedding = np.random.randn(self.embedding_dim)
            user_embedding = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)

            # Search for similar videos
            candidates, scores = store.search(user_embedding, top_k=10)

            results.append({
                "user_id": user_id,
                "num_recommendations": len(candidates),
                "top_video_ids": candidates[:5],
                "top_scores": scores[:5].tolist() if hasattr(scores, 'tolist') else scores[:5],
            })

            logger.info(f"    User {user_id}: Got {len(candidates)} recommendations")

        return {
            "status": "success",
            "mode": "local",
            "test_results": results,
        }

    def _test_api_recommendations(self) -> Dict[str, Any]:
        """Test recommendations via the deployed API."""
        import requests

        # Get API URL from environment or CloudFormation
        api_url = os.environ.get("API_URL")
        if not api_url:
            api_url = self._get_cfn_output("ApiUrl")
        if not api_url:
            raise ValueError("API_URL not set and could not fetch from CloudFormation")

        logger.info(f"  Testing API at {api_url}...")

        # Test health endpoint first
        health_response = requests.get(f"{api_url}/health", timeout=30)
        logger.info(f"    Health check: {health_response.status_code}")

        # Test recommendations for a few users
        test_users = [1, 5, 10]
        results = []

        for user_id in test_users:
            try:
                response = requests.post(
                    f"{api_url}/recommendations",
                    json={"user_id": user_id, "num_recommendations": 10},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    num_recs = len(data.get("recommendations", []))
                    logger.info(f"    User {user_id}: Got {num_recs} recommendations")
                    results.append({
                        "user_id": user_id,
                        "status": "success",
                        "num_recommendations": num_recs,
                        "latency_ms": data.get("total_latency_ms"),
                    })
                else:
                    logger.warning(f"    User {user_id}: Error {response.status_code}")
                    results.append({
                        "user_id": user_id,
                        "status": "error",
                        "status_code": response.status_code,
                        "message": response.text[:200],
                    })

            except Exception as e:
                logger.error(f"    User {user_id}: Exception {e}")
                results.append({
                    "user_id": user_id,
                    "status": "error",
                    "message": str(e),
                })

        return {
            "status": "success",
            "mode": "api",
            "api_url": api_url,
            "health_status": health_response.status_code,
            "test_results": results,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap the recommendation system with synthetic data"
    )
    parser.add_argument(
        "--num-users", type=int, default=1000,
        help="Number of users to generate (default: 1000)"
    )
    parser.add_argument(
        "--num-videos", type=int, default=500,
        help="Number of videos to generate (default: 500)"
    )
    parser.add_argument(
        "--num-channels", type=int, default=100,
        help="Number of channels to generate (default: 100)"
    )
    parser.add_argument(
        "--num-interactions", type=int, default=10000,
        help="Number of interactions to generate (default: 10000)"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=16,
        help="Embedding dimension (default: 16)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run in local mode (no AWS services)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    bootstrapper = DataBootstrapper(
        num_users=args.num_users,
        num_channels=args.num_channels,
        num_videos=args.num_videos,
        num_interactions=args.num_interactions,
        embedding_dim=args.embedding_dim,
        local_mode=args.local,
        seed=args.seed,
    )

    results = bootstrapper.run()

    # Print summary
    print("\n" + "=" * 60)
    print("BOOTSTRAP SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))

    # Save results
    output_path = Path("data/bootstrap_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
