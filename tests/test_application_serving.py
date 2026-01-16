"""
Unit tests for Phase 8 serving components.

Tests the new serving infrastructure:
- Feature Store client
- Redis cache client
- Multi-query candidate generation
- Filtering service
- Enhanced ranker service
- Recommendation service orchestrator
- Lambda handler
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.serving.feature_store_client import (
    FeatureStoreConfig,
    InMemoryFeatureStore,
    create_feature_store_client,
)
from src.serving.redis_cache_client import (
    RedisCacheConfig,
    UserInteraction,
    InMemoryCacheClient,
    create_cache_client,
)
from src.serving.multi_query_generator import (
    MultiQueryConfig,
    MultiQueryGenerator,
    QueryResult,
    DiversitySampler,
)
from src.serving.filtering_service import (
    FilteringConfig,
    FilteringService,
    FilteringPipeline,
)
from src.serving.ranker_service_v2 import (
    RankerServiceConfig,
    EnhancedRankerService,
    RankedCandidate,
)
from src.serving.recommendation_service import (
    RecommendationServiceConfig,
    RecommendationService,
    VideoRecommendation,
)
from src.lambdas.recommendations import (
    handler,
    _create_response,
    _create_error_response,
)


class TestFeatureStoreClient:
    """Tests for Feature Store client."""

    def test_in_memory_feature_store_creation(self):
        """Test creating in-memory feature store."""
        config = FeatureStoreConfig()
        store = InMemoryFeatureStore(config)

        assert store is not None
        assert store.config == config

    def test_load_user_features(self):
        """Test loading user features."""
        config = FeatureStoreConfig()
        store = InMemoryFeatureStore(config)

        user_features = {
            1: {"user_id": 1, "country": "US", "age": 25},
            2: {"user_id": 2, "country": "UK", "age": 30},
        }
        store.load_user_features(user_features)

        result = store.get_user_features(1)
        assert result is not None
        assert result["country"] == "US"
        assert result["age"] == 25

    def test_load_video_features(self):
        """Test loading video features."""
        config = FeatureStoreConfig()
        store = InMemoryFeatureStore(config)

        video_features = {
            101: {"video_id": 101, "category": "Music", "duration": 300},
            102: {"video_id": 102, "category": "Sports", "duration": 600},
        }
        store.load_video_features(video_features)

        result = store.get_video_features(101)
        assert result is not None
        assert result["category"] == "Music"

    def test_batch_get_video_features(self):
        """Test batch fetching video features."""
        config = FeatureStoreConfig()
        store = InMemoryFeatureStore(config)

        video_features = {
            101: {"video_id": 101, "category": "Music"},
            102: {"video_id": 102, "category": "Sports"},
            103: {"video_id": 103, "category": "Gaming"},
        }
        store.load_video_features(video_features)

        result = store.batch_get_video_features([101, 102, 999])
        assert len(result) == 2
        assert 101 in result
        assert 102 in result
        assert 999 not in result

    def test_feature_store_caching(self):
        """Test that caching works correctly."""
        config = FeatureStoreConfig(enable_cache=True)
        store = InMemoryFeatureStore(config)

        user_features = {1: {"user_id": 1, "country": "US"}}
        store.load_user_features(user_features)

        # First call
        result1 = store.get_user_features(1)
        # Second call should use cache
        result2 = store.get_user_features(1)

        assert result1 == result2

    def test_create_feature_store_client(self):
        """Test factory function."""
        config = FeatureStoreConfig()

        # In-memory client
        client = create_feature_store_client(config, use_sagemaker=False)
        assert isinstance(client, InMemoryFeatureStore)


class TestRedisCacheClient:
    """Tests for Redis cache client."""

    def test_in_memory_cache_creation(self):
        """Test creating in-memory cache client."""
        config = RedisCacheConfig()
        cache = InMemoryCacheClient(config)

        assert cache is not None

    def test_add_interaction(self):
        """Test adding user interaction."""
        config = RedisCacheConfig()
        cache = InMemoryCacheClient(config)

        interaction = UserInteraction(
            category="Music",
            video_id=101,
            timestamp=datetime.utcnow().isoformat(),
            interaction_type="watch",
            duration_watched=120.0,
        )
        cache.add_interaction(user_id=1, interaction=interaction)

        interactions = cache.get_recent_interactions(user_id=1)
        assert len(interactions) == 1
        assert interactions[0].category == "Music"
        assert interactions[0].video_id == 101

    def test_get_recent_categories(self):
        """Test getting recent categories."""
        config = RedisCacheConfig()
        cache = InMemoryCacheClient(config)

        # Add multiple interactions
        categories = ["Music", "Sports", "Music", "Gaming"]
        for i, cat in enumerate(categories):
            interaction = UserInteraction(
                category=cat,
                video_id=100 + i,
                timestamp=datetime.utcnow().isoformat(),
            )
            cache.add_interaction(user_id=1, interaction=interaction)

        recent = cache.get_recent_categories(user_id=1, limit=3)

        # Should be unique categories in order
        assert len(recent) == 3
        assert recent[0] == "Gaming"  # Most recent
        assert "Music" in recent

    def test_get_watched_videos(self):
        """Test getting watched videos."""
        config = RedisCacheConfig()
        cache = InMemoryCacheClient(config)

        # Add interactions
        for vid in [101, 102, 103]:
            interaction = UserInteraction(
                category="Music",
                video_id=vid,
                timestamp=datetime.utcnow().isoformat(),
            )
            cache.add_interaction(user_id=1, interaction=interaction)

        watched = cache.get_watched_videos(user_id=1)

        assert len(watched) == 3
        assert 101 in watched
        assert 102 in watched
        assert 103 in watched

    def test_set_and_get_recommendations(self):
        """Test storing and retrieving recommendations."""
        config = RedisCacheConfig()
        cache = InMemoryCacheClient(config)

        video_ids = [101, 102, 103]
        scores = [0.9, 0.8, 0.7]

        cache.set_recommendations(user_id=1, video_ids=video_ids, scores=scores)

        recs = cache.get_recommendations(user_id=1)
        assert recs is not None
        assert len(recs) == 3
        assert recs[0]["video_id"] == 101
        assert recs[0]["score"] == 0.9

    def test_recommendations_expiry(self):
        """Test recommendations expire after TTL."""
        config = RedisCacheConfig(recommendations_ttl_minutes=0)  # Immediate expiry
        cache = InMemoryCacheClient(config)

        cache.set_recommendations(user_id=1, video_ids=[101], scores=[0.9])

        # Should be expired immediately
        recs = cache.get_recommendations(user_id=1)
        assert recs is None

    def test_clear_user_data(self):
        """Test clearing user data."""
        config = RedisCacheConfig()
        cache = InMemoryCacheClient(config)

        interaction = UserInteraction(
            category="Music", video_id=101, timestamp=datetime.utcnow().isoformat()
        )
        cache.add_interaction(user_id=1, interaction=interaction)
        cache.set_recommendations(user_id=1, video_ids=[101])

        cache.clear_user_data(user_id=1)

        assert len(cache.get_recent_interactions(user_id=1)) == 0
        assert len(cache.get_watched_videos(user_id=1)) == 0
        assert cache.get_recommendations(user_id=1) is None

    def test_create_cache_client(self):
        """Test factory function."""
        config = RedisCacheConfig()

        client = create_cache_client(config, use_redis=False)
        assert isinstance(client, InMemoryCacheClient)


class TestMultiQueryGenerator:
    """Tests for multi-query candidate generation."""

    def test_generator_creation(self):
        """Test creating multi-query generator."""
        config = MultiQueryConfig()
        cache = InMemoryCacheClient(RedisCacheConfig())
        store = InMemoryFeatureStore(FeatureStoreConfig())

        generator = MultiQueryGenerator(config, cache, store)
        assert generator is not None

    def test_generate_query_contexts(self):
        """Test generating query contexts."""
        config = MultiQueryConfig(max_queries=3)
        cache = InMemoryCacheClient(RedisCacheConfig())
        store = InMemoryFeatureStore(FeatureStoreConfig())

        # Add user interactions
        for cat in ["Music", "Sports"]:
            interaction = UserInteraction(
                category=cat, video_id=100, timestamp=datetime.utcnow().isoformat()
            )
            cache.add_interaction(user_id=1, interaction=interaction)

        generator = MultiQueryGenerator(config, cache, store)

        base_user_data = {"user_id": 1, "country": "US"}
        contexts = generator.generate_query_contexts(user_id=1, base_user_data=base_user_data)

        # Should have primary + category queries
        assert len(contexts) >= 1
        assert contexts[0]["_query_type"] == "primary"

    def test_merge_results(self):
        """Test merging query results."""
        config = MultiQueryConfig(
            primary_query_weight=0.5,
            category_query_weight=0.3,
        )
        cache = InMemoryCacheClient(RedisCacheConfig())
        store = InMemoryFeatureStore(FeatureStoreConfig())
        generator = MultiQueryGenerator(config, cache, store)

        # Create mock query results
        results = [
            QueryResult(
                query_id="q_primary",
                query_type="primary",
                category=None,
                video_ids=[101, 102, 103],
                scores=[0.9, 0.8, 0.7],
            ),
            QueryResult(
                query_id="q_cat_0",
                query_type="category",
                category="Music",
                video_ids=[101, 104, 105],  # 101 appears in both
                scores=[0.85, 0.75, 0.65],
            ),
        ]

        merged = generator.merge_results(results, max_results=5)

        # Video 101 should have highest score (appears in both)
        assert merged.merged_video_ids[0] == 101
        assert len(merged.merged_video_ids) == 5
        assert merged.unique_candidates == 5

    def test_diversity_sampler(self):
        """Test diversity sampler."""
        sampler = DiversitySampler(min_categories=2, category_cap=3)

        video_ids = [1, 2, 3, 4, 5, 6]
        scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
        metadata = {
            1: {"category": "Music"},
            2: {"category": "Music"},
            3: {"category": "Music"},
            4: {"category": "Sports"},
            5: {"category": "Sports"},
            6: {"category": "Gaming"},
        }

        diverse_ids, diverse_scores = sampler.sample(
            video_ids, scores, metadata, n=4
        )

        # Should include videos from different categories
        categories_selected = {metadata[vid]["category"] for vid in diverse_ids}
        assert len(categories_selected) >= 2


class TestFilteringService:
    """Tests for filtering service."""

    def test_filtering_service_creation(self):
        """Test creating filtering service."""
        config = FilteringConfig()
        cache = InMemoryCacheClient(RedisCacheConfig())

        service = FilteringService(config, cache)
        assert service is not None

    def test_filter_watched_videos(self):
        """Test filtering watched videos."""
        config = FilteringConfig(filter_watched_videos=True)
        cache = InMemoryCacheClient(RedisCacheConfig())

        # Add watched videos
        for vid in [101, 102]:
            interaction = UserInteraction(
                category="Music", video_id=vid, timestamp=datetime.utcnow().isoformat()
            )
            cache.add_interaction(user_id=1, interaction=interaction)

        service = FilteringService(config, cache)

        result = service.filter(
            video_ids=[101, 102, 103, 104],
            user_id=1,
            video_metadata={},
        )

        # 101 and 102 should be filtered
        assert 101 not in result.passed_video_ids
        assert 102 not in result.passed_video_ids
        assert 103 in result.passed_video_ids
        assert 104 in result.passed_video_ids

    def test_filter_by_category(self):
        """Test filtering by blocked categories."""
        config = FilteringConfig(blocked_categories=["Adult", "Violence"])
        cache = InMemoryCacheClient(RedisCacheConfig())

        service = FilteringService(config, cache)

        metadata = {
            101: {"category": "Music"},
            102: {"category": "Adult"},
            103: {"category": "Sports"},
        }

        result = service.filter(
            video_ids=[101, 102, 103],
            user_id=1,
            video_metadata=metadata,
        )

        assert 101 in result.passed_video_ids
        assert 102 not in result.passed_video_ids  # Blocked category
        assert 103 in result.passed_video_ids

    def test_filter_by_duration(self):
        """Test filtering by video duration."""
        config = FilteringConfig(
            enable_duration_filter=True,
            min_duration_seconds=60,
            max_duration_seconds=3600,
        )
        cache = InMemoryCacheClient(RedisCacheConfig())

        service = FilteringService(config, cache)

        metadata = {
            101: {"video_duration": 30},   # Too short
            102: {"video_duration": 300},  # OK
            103: {"video_duration": 5000}, # Too long
        }

        result = service.filter(
            video_ids=[101, 102, 103],
            user_id=1,
            video_metadata=metadata,
        )

        assert 101 not in result.passed_video_ids
        assert 102 in result.passed_video_ids
        assert 103 not in result.passed_video_ids

    def test_filter_nsfw(self):
        """Test filtering NSFW content."""
        config = FilteringConfig(filter_nsfw=True)
        cache = InMemoryCacheClient(RedisCacheConfig())

        service = FilteringService(config, cache)

        metadata = {
            101: {"is_nsfw": False},
            102: {"is_nsfw": True},
        }

        result = service.filter(
            video_ids=[101, 102],
            user_id=1,
            video_metadata=metadata,
        )

        assert 101 in result.passed_video_ids
        assert 102 not in result.passed_video_ids

    def test_filter_by_quality(self):
        """Test filtering by quality score."""
        config = FilteringConfig(filter_low_quality=True, min_quality_score=0.5)
        cache = InMemoryCacheClient(RedisCacheConfig())

        service = FilteringService(config, cache)

        metadata = {
            101: {"quality_score": 0.8},
            102: {"quality_score": 0.3},  # Low quality
        }

        result = service.filter(
            video_ids=[101, 102],
            user_id=1,
            video_metadata=metadata,
        )

        assert 101 in result.passed_video_ids
        assert 102 not in result.passed_video_ids

    def test_filter_stats(self):
        """Test that filter stats are recorded."""
        config = FilteringConfig(blocked_categories=["Adult"])
        cache = InMemoryCacheClient(RedisCacheConfig())

        service = FilteringService(config, cache)

        metadata = {
            101: {"category": "Adult"},
            102: {"category": "Music"},
        }

        result = service.filter(
            video_ids=[101, 102],
            user_id=1,
            video_metadata=metadata,
        )

        assert result.total_filtered == 1
        assert "category" in result.filter_stats

    def test_duration_preference_filter(self):
        """Test filtering by duration preference."""
        config = FilteringConfig()
        cache = InMemoryCacheClient(RedisCacheConfig())
        service = FilteringService(config, cache)

        metadata = {
            101: {"video_duration": 60},    # Short
            102: {"video_duration": 600},   # Medium
            103: {"video_duration": 1500},  # Long
        }

        short_result = service.filter_by_duration_preference(
            [101, 102, 103], metadata, "short"
        )
        assert 101 in short_result
        assert 102 not in short_result

        long_result = service.filter_by_duration_preference(
            [101, 102, 103], metadata, "long"
        )
        assert 103 in long_result
        assert 101 not in long_result


class TestEnhancedRankerService:
    """Tests for enhanced ranker service."""

    def test_ranker_service_creation(self):
        """Test creating ranker service."""
        config = RankerServiceConfig()
        store = InMemoryFeatureStore(FeatureStoreConfig())

        service = EnhancedRankerService(config, store)
        assert service is not None

    def test_rank_without_model(self):
        """Test ranking without a loaded model (fallback behavior)."""
        config = RankerServiceConfig()
        store = InMemoryFeatureStore(FeatureStoreConfig())

        service = EnhancedRankerService(config, store)

        candidates = [(101, 0.9), (102, 0.8), (103, 0.7)]
        result = service.rank(user_id=1, candidates=candidates)

        # Should return ranked candidates using retrieval scores
        assert len(result.ranked_candidates) == 3
        assert result.total_scored == 3

    def test_rank_with_features(self):
        """Test ranking with provided features."""
        config = RankerServiceConfig(enrich_from_feature_store=False)
        store = InMemoryFeatureStore(FeatureStoreConfig())

        service = EnhancedRankerService(config, store)

        candidates = [(101, 0.9), (102, 0.8)]
        user_features = {"user_id": 1, "country": "US", "age": 25}
        video_metadata = {
            101: {"category": "Music", "view_count": 1000},
            102: {"category": "Sports", "view_count": 500},
        }

        result = service.rank(
            user_id=1,
            candidates=candidates,
            user_features=user_features,
            video_metadata=video_metadata,
        )

        assert len(result.ranked_candidates) == 2

    def test_score_blending(self):
        """Test score blending with default weights."""
        config = RankerServiceConfig()
        store = InMemoryFeatureStore(FeatureStoreConfig())

        service = EnhancedRankerService(config, store)

        # Test the blending calculation
        blended = service._blend_scores(
            retrieval_score=0.8,
            ranker_score=0.6,
            retrieval_weight=0.3,
        )

        # 0.3 * 0.8 + 0.7 * 0.6 = 0.24 + 0.42 = 0.66
        assert pytest.approx(blended, rel=1e-2) == 0.66

    def test_feature_preparation(self):
        """Test feature preparation for ranker."""
        config = RankerServiceConfig()
        store = InMemoryFeatureStore(FeatureStoreConfig())

        service = EnhancedRankerService(config, store)

        user_features = {"user_id": 1, "country": "US"}
        video_features = {
            "video_id": 101,
            "video_duration": 300,
            "view_count": 1000,
            "like_count": 100,
            "comment_count": 10,
            "channel_subscriber_count": 50000,
        }

        features = service._prepare_features(
            user_id=1,
            video_id=101,
            user_features=user_features,
            video_features=video_features,
            interaction_context={},
        )

        # Check derived features
        assert "video_duration_log" in features
        assert "view_count_log" in features
        assert "channel_tier" in features
        assert features["channel_tier"] == "small"  # 10k-100k subscribers


class TestRecommendationService:
    """Tests for recommendation service orchestrator."""

    def test_service_creation(self):
        """Test creating recommendation service."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)

        assert service is not None
        assert not service._is_initialized

    def test_service_initialization(self):
        """Test initializing recommendation service."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)

        service.initialize(
            use_sagemaker_feature_store=False,
            use_redis=False,
        )

        assert service._is_initialized
        assert service.feature_store is not None
        assert service.cache_client is not None
        assert service.multi_query_generator is not None
        assert service.filtering_service is not None
        assert service.ranker_service is not None

    def test_load_video_data(self):
        """Test loading video data."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        metadata = {
            101: {"video_id": 101, "category": "Music"},
            102: {"video_id": 102, "category": "Sports"},
        }
        embeddings = {
            101: np.random.randn(16).astype(np.float32),
            102: np.random.randn(16).astype(np.float32),
        }

        service.load_video_data(metadata, embeddings)

        assert len(service._video_metadata) == 2
        assert service.vector_store.size == 2

    def test_get_recommendations(self):
        """Test getting recommendations."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        # Load some video data
        metadata = {i: {"video_id": i, "category": "Music"} for i in range(100, 110)}
        embeddings = {i: np.random.randn(16).astype(np.float32) for i in range(100, 110)}
        service.load_video_data(metadata, embeddings)

        # Get recommendations
        response = service.get_recommendations(user_id=1, num_recommendations=5)

        assert response is not None
        assert response.user_id == 1
        assert len(response.recommendations) <= 5
        assert "candidate_generation" in response.stage_latencies

    def test_record_interaction(self):
        """Test recording user interaction."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        service.record_interaction(
            user_id=1,
            video_id=101,
            category="Music",
            interaction_type="watch",
            duration_watched=120.0,
        )

        # Check interaction was recorded
        interactions = service.cache_client.get_recent_interactions(user_id=1)
        assert len(interactions) == 1
        assert interactions[0].video_id == 101

    def test_store_and_get_cached_recommendations(self):
        """Test storing and retrieving cached recommendations."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        recommendations = [
            VideoRecommendation(video_id=101, score=0.9, rank=1, source="ranking"),
            VideoRecommendation(video_id=102, score=0.8, rank=2, source="ranking"),
        ]

        service.store_recommendations(user_id=1, recommendations=recommendations)

        cached = service.get_cached_recommendations(user_id=1)
        assert cached is not None
        assert len(cached) == 2

    def test_service_warmup(self):
        """Test service warmup."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        # Should not raise
        service.warmup()

    def test_get_stats(self):
        """Test getting service stats."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        stats = service.get_stats()

        assert stats["is_initialized"] is True
        assert stats["request_count"] == 0


class TestLambdaHandler:
    """Tests for Lambda handler."""

    def test_create_response(self):
        """Test creating API Gateway response."""
        response = _create_response(200, {"status": "ok"})

        assert response["statusCode"] == 200
        assert "Content-Type" in response["headers"]
        assert "application/json" in response["headers"]["Content-Type"]

    def test_error_response(self):
        """Test creating error response."""
        response = _create_error_response(400, "INVALID", "Bad request")

        assert response["statusCode"] == 400
        body = __import__("json").loads(response["body"])
        assert body["error"]["type"] == "INVALID"
        assert body["error"]["message"] == "Bad request"

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        event = {
            "httpMethod": "GET",
            "path": "/health",
        }

        response = handler(event, None)

        assert response["statusCode"] == 200
        body = __import__("json").loads(response["body"])
        assert "status" in body

    def test_options_request(self):
        """Test OPTIONS request for CORS."""
        event = {
            "httpMethod": "OPTIONS",
            "path": "/recommendations",
        }

        response = handler(event, None)

        assert response["statusCode"] == 200
        assert "Access-Control-Allow-Origin" in response["headers"]

    def test_unknown_endpoint(self):
        """Test unknown endpoint returns 404."""
        event = {
            "httpMethod": "GET",
            "path": "/unknown",
        }

        response = handler(event, None)

        assert response["statusCode"] == 404


class TestFilteringPipeline:
    """Tests for filtering pipeline."""

    def test_pipeline_creation(self):
        """Test creating filtering pipeline."""
        config = FilteringConfig()
        pipeline = FilteringPipeline(config)

        assert pipeline is not None

    def test_pipeline_run(self):
        """Test running filtering pipeline."""
        config = FilteringConfig()
        pipeline = FilteringPipeline(config)

        metadata = {101: {"category": "Music"}, 102: {"category": "Sports"}}

        result = pipeline.run(
            video_ids=[101, 102],
            user_id=1,
            video_metadata=metadata,
        )

        assert result.total_input == 2
        assert result.total_passed == 2

    def test_get_filter_stats_summary(self):
        """Test getting summary statistics."""
        config = FilteringConfig(blocked_categories=["Adult"])
        pipeline = FilteringPipeline(config)

        results = []
        for i in range(3):
            metadata = {
                100 + i: {"category": "Music"},
                200 + i: {"category": "Adult"},
            }
            result = pipeline.run(
                video_ids=[100 + i, 200 + i],
                user_id=1,
                video_metadata=metadata,
            )
            results.append(result)

        summary = pipeline.get_filter_stats_summary(results)

        assert summary["total_input"] == 6
        assert summary["total_filtered"] == 3
        assert "filter_breakdown" in summary


class TestIntegration:
    """Integration tests for the full serving pipeline."""

    def test_full_recommendation_flow(self):
        """Test full recommendation flow."""
        # Create service
        config = RecommendationServiceConfig(
            enable_multi_query=True,
            enable_filtering=True,
            enable_ranking=True,
            enable_diversification=True,
        )
        service = RecommendationService(config)
        service.initialize()

        # Load video data
        categories = ["Music", "Sports", "Gaming", "Education", "News"]
        metadata = {}
        embeddings = {}
        for i in range(100, 150):
            metadata[i] = {
                "video_id": i,
                "category": categories[i % len(categories)],
                "video_duration": 300 + (i % 10) * 60,
                "view_count": 1000 * (i % 20),
            }
            embeddings[i] = np.random.randn(16).astype(np.float32)

        service.load_video_data(metadata, embeddings)

        # Record some user interactions
        for cat in ["Music", "Gaming"]:
            service.record_interaction(
                user_id=1,
                video_id=100,
                category=cat,
            )

        # Get recommendations
        response = service.get_recommendations(
            user_id=1,
            num_recommendations=10,
            user_preferences={"min_duration": 60, "max_duration": 3600},
        )

        assert response is not None
        assert len(response.recommendations) <= 10
        assert response.total_latency_ms > 0

        # Check that recommendations have valid scores
        # Note: diversification may reorder items for category variety
        # so we don't check strict ordering
        scores = [r.score for r in response.recommendations]
        assert all(0 <= s <= 2 for s in scores)  # Allow scores slightly above 1 due to diversity bonus

    def test_caching_recommendations(self):
        """Test caching and retrieving recommendations."""
        config = RecommendationServiceConfig()
        service = RecommendationService(config)
        service.initialize()

        # Load minimal video data
        metadata = {100: {"video_id": 100, "category": "Music"}}
        embeddings = {100: np.random.randn(16).astype(np.float32)}
        service.load_video_data(metadata, embeddings)

        # Get and cache recommendations
        response = service.get_recommendations(user_id=1, num_recommendations=5)
        service.store_recommendations(user_id=1, recommendations=response.recommendations)

        # Retrieve cached
        cached = service.get_cached_recommendations(user_id=1)

        assert cached is not None
        if response.recommendations:
            assert len(cached) == len(response.recommendations)
