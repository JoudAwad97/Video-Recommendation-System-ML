"""
Unit tests for serving components.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.serving.serving_config import ServingConfig, VectorDBConfig, SageMakerConfig
from src.serving.vector_store import InMemoryVectorStore
from src.serving.candidate_retrieval import (
    CandidateRetrievalService,
    VideoCandidate,
    RetrievalResult,
)
from src.serving.ranking_service import RankingService, RankedVideo
from src.serving.orchestrator import (
    RecommendationOrchestrator,
    RecommendationRequest,
    RecommendationResponse,
)


class TestVectorDBConfig:
    """Tests for VectorDBConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VectorDBConfig()

        assert config.store_type == "faiss"
        assert config.index_type == "IVFFlat"
        assert config.embedding_dim == 16
        assert config.default_top_k == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = VectorDBConfig(
            store_type="memory",
            embedding_dim=32,
            nlist=200,
            nprobe=20,
        )

        assert config.store_type == "memory"
        assert config.embedding_dim == 32
        assert config.nlist == 200
        assert config.nprobe == 20


class TestServingConfig:
    """Tests for ServingConfig."""

    def test_default_config(self):
        """Test default serving config."""
        config = ServingConfig()

        assert config.num_candidates == 100
        assert config.top_k_final == 20
        assert config.batch_size == 256
        assert config.enable_business_rules is True

    def test_get_model_paths(self):
        """Test get_model_paths method."""
        config = ServingConfig(
            two_tower_model_path="path/to/two_tower",
            ranker_model_path="path/to/ranker",
        )

        paths = config.get_model_paths()

        assert paths["two_tower"] == "path/to/two_tower"
        assert paths["ranker"] == "path/to/ranker"


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    @pytest.fixture
    def config(self):
        return VectorDBConfig(embedding_dim=8)

    @pytest.fixture
    def sample_embeddings(self):
        np.random.seed(42)
        embeddings = np.random.randn(100, 8).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    @pytest.fixture
    def sample_ids(self):
        return list(range(1, 101))

    def test_add_embeddings(self, config, sample_ids, sample_embeddings):
        """Test adding embeddings to store."""
        store = InMemoryVectorStore(config)
        store.add(sample_ids, sample_embeddings)

        assert store.size == 100

    def test_search(self, config, sample_ids, sample_embeddings):
        """Test search functionality."""
        store = InMemoryVectorStore(config)
        store.add(sample_ids, sample_embeddings)

        # Search with first embedding
        query = sample_embeddings[0]
        video_ids, scores = store.search(query, top_k=10)

        assert len(video_ids) == 10
        assert len(scores) == 10
        # First result should be the query itself (highest similarity)
        assert video_ids[0] == sample_ids[0]
        assert scores[0] == pytest.approx(1.0, abs=0.01)

    def test_batch_search(self, config, sample_ids, sample_embeddings):
        """Test batch search."""
        store = InMemoryVectorStore(config)
        store.add(sample_ids, sample_embeddings)

        # Search with multiple queries
        queries = sample_embeddings[:5]
        video_ids, scores = store.batch_search(queries, top_k=10)

        assert video_ids.shape == (5, 10)
        assert scores.shape == (5, 10)
        # Each query should find itself as best match
        for i in range(5):
            assert video_ids[i, 0] == sample_ids[i]

    def test_save_load(self, config, sample_ids, sample_embeddings):
        """Test save and load."""
        store = InMemoryVectorStore(config)
        store.add(sample_ids, sample_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            # Load into new store
            store2 = InMemoryVectorStore(config)
            store2.load(tmpdir)

            assert store2.size == 100

            # Verify search works after loading
            query = sample_embeddings[0]
            video_ids, scores = store2.search(query, top_k=5)
            assert video_ids[0] == sample_ids[0]


class TestCandidateRetrievalService:
    """Tests for CandidateRetrievalService."""

    @pytest.fixture
    def config(self):
        return ServingConfig(
            num_candidates=50,
            enable_business_rules=False,
        )

    @pytest.fixture
    def mock_query_encoder(self):
        """Create mock query encoder."""
        class MockEncoder:
            def encode_user(self, user_data):
                np.random.seed(user_data.get("id", 0))
                emb = np.random.randn(8).astype(np.float32)
                return emb / (np.linalg.norm(emb) + 1e-8)

            def encode_users_batch(self, users_data):
                embeddings = []
                for user in users_data:
                    embeddings.append(self.encode_user(user))
                return np.array(embeddings)

        return MockEncoder()

    @pytest.fixture
    def vector_store(self):
        config = VectorDBConfig(embedding_dim=8)
        store = InMemoryVectorStore(config)

        np.random.seed(42)
        embeddings = np.random.randn(100, 8).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        store.add(list(range(1, 101)), embeddings)
        return store

    def test_retrieve_candidates(self, config, mock_query_encoder, vector_store):
        """Test candidate retrieval."""
        service = CandidateRetrievalService(config)
        service.set_query_encoder(mock_query_encoder)
        service.set_vector_store(vector_store)

        user_data = {"id": 1, "country_code": "US", "age": 25}
        result = service.retrieve(user_data, num_candidates=20)

        assert isinstance(result, RetrievalResult)
        assert len(result.candidates) == 20
        assert result.user_embedding is not None
        assert result.retrieval_time_ms > 0

        # Check candidates
        for candidate in result.candidates:
            assert isinstance(candidate, VideoCandidate)
            assert candidate.video_id > 0
            assert 0 <= candidate.similarity_score <= 1

    def test_retrieve_with_exclusions(self, config, mock_query_encoder, vector_store):
        """Test retrieval with excluded videos."""
        service = CandidateRetrievalService(config)
        service.set_query_encoder(mock_query_encoder)
        service.set_vector_store(vector_store)

        excluded = {1, 2, 3, 4, 5}
        user_data = {"id": 1}
        result = service.retrieve(
            user_data,
            num_candidates=20,
            excluded_video_ids=excluded,
        )

        # Excluded videos should not be in results
        result_ids = {c.video_id for c in result.candidates}
        assert result_ids.isdisjoint(excluded)

    def test_retrieve_with_blocked_videos(self, config, mock_query_encoder, vector_store):
        """Test retrieval with blocked videos."""
        service = CandidateRetrievalService(config)
        service.set_query_encoder(mock_query_encoder)
        service.set_vector_store(vector_store)
        service.set_blocked_videos({10, 20, 30})

        user_data = {"id": 1}
        result = service.retrieve(user_data, num_candidates=50)

        result_ids = {c.video_id for c in result.candidates}
        assert 10 not in result_ids
        assert 20 not in result_ids
        assert 30 not in result_ids


class TestRankedVideo:
    """Tests for RankedVideo dataclass."""

    def test_ranked_video_creation(self):
        """Test RankedVideo creation."""
        video = RankedVideo(
            video_id=123,
            retrieval_score=0.8,
            ranker_score=0.75,
            final_score=0.77,
            rank=1,
        )

        assert video.video_id == 123
        assert video.retrieval_score == 0.8
        assert video.ranker_score == 0.75
        assert video.final_score == 0.77
        assert video.rank == 1


class TestRecommendationOrchestrator:
    """Tests for RecommendationOrchestrator."""

    @pytest.fixture
    def config(self):
        return ServingConfig(
            num_candidates=50,
            top_k_final=10,
            enable_business_rules=False,
        )

    @pytest.fixture
    def mock_query_encoder(self):
        class MockEncoder:
            def encode_user(self, user_data):
                np.random.seed(user_data.get("id", 0))
                emb = np.random.randn(8).astype(np.float32)
                return emb / (np.linalg.norm(emb) + 1e-8)

            def encode_users_batch(self, users_data):
                return np.array([self.encode_user(u) for u in users_data])

            def warmup(self):
                pass

        return MockEncoder()

    @pytest.fixture
    def setup_orchestrator(self, config, mock_query_encoder):
        """Set up orchestrator with mock components."""
        orchestrator = RecommendationOrchestrator(config)

        # Create vector store
        vector_config = VectorDBConfig(embedding_dim=8)
        vector_store = InMemoryVectorStore(vector_config)

        np.random.seed(42)
        embeddings = np.random.randn(100, 8).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        video_ids = list(range(1, 101))
        vector_store.add(video_ids, embeddings)

        # Create retrieval service
        from src.serving.candidate_retrieval import CandidateRetrievalService
        retrieval_service = CandidateRetrievalService(config)
        retrieval_service.set_query_encoder(mock_query_encoder)
        retrieval_service.set_vector_store(vector_store)

        # Create video data
        video_data = {
            i: {"category": f"Cat{i % 5}", "is_active": True}
            for i in range(1, 101)
        }
        retrieval_service.load_video_metadata(video_data)

        # Set up orchestrator
        orchestrator.query_encoder = mock_query_encoder
        orchestrator.vector_store = vector_store
        orchestrator.retrieval_service = retrieval_service
        orchestrator.video_data = video_data
        orchestrator._is_initialized = True

        return orchestrator

    def test_recommend(self, setup_orchestrator):
        """Test recommendation generation."""
        orchestrator = setup_orchestrator

        request = RecommendationRequest(
            user_data={"id": 1, "country_code": "US", "age": 30},
            num_recommendations=10,
        )

        response = orchestrator.recommend(request)

        assert isinstance(response, RecommendationResponse)
        assert len(response.recommendations) == 10
        assert response.latency_ms > 0
        assert "retrieval" in response.stage_latencies

        # Check recommendations
        for i, rec in enumerate(response.recommendations, start=1):
            assert rec.rank == i
            assert rec.video_id > 0
            assert rec.score >= 0

    def test_recommend_simple(self, setup_orchestrator):
        """Test simple recommendation interface."""
        orchestrator = setup_orchestrator

        recs = orchestrator.recommend_simple(
            user_data={"id": 1},
            num_recommendations=5,
        )

        assert len(recs) == 5
        assert all("video_id" in r for r in recs)
        assert all("score" in r for r in recs)
        assert all("rank" in r for r in recs)

    def test_recommend_with_exclusions(self, setup_orchestrator):
        """Test recommendations with exclusions."""
        orchestrator = setup_orchestrator

        excluded = {1, 2, 3, 4, 5}
        request = RecommendationRequest(
            user_data={"id": 1},
            num_recommendations=10,
            excluded_video_ids=excluded,
        )

        response = orchestrator.recommend(request)

        result_ids = {r.video_id for r in response.recommendations}
        assert result_ids.isdisjoint(excluded)

    def test_recommend_with_diversification(self, setup_orchestrator):
        """Test recommendation diversification."""
        orchestrator = setup_orchestrator

        # Without diversification
        request_no_div = RecommendationRequest(
            user_data={"id": 1},
            num_recommendations=10,
            enable_diversification=False,
        )
        response_no_div = orchestrator.recommend(request_no_div)

        # With diversification
        request_div = RecommendationRequest(
            user_data={"id": 1},
            num_recommendations=10,
            enable_diversification=True,
            diversity_weight=0.5,
        )
        response_div = orchestrator.recommend(request_div)

        # Both should return results
        assert len(response_no_div.recommendations) == 10
        assert len(response_div.recommendations) == 10

    def test_get_stats(self, setup_orchestrator):
        """Test get_stats method."""
        orchestrator = setup_orchestrator

        stats = orchestrator.get_stats()

        assert stats["is_initialized"] is True
        assert stats["vector_store_size"] == 100
        assert stats["video_data_count"] == 100


class TestRecommendationRequest:
    """Tests for RecommendationRequest."""

    def test_default_request(self):
        """Test default request values."""
        request = RecommendationRequest(
            user_data={"id": 1},
        )

        assert request.num_recommendations == 20
        assert request.enable_diversification is True
        assert request.diversity_weight == 0.1

    def test_custom_request(self):
        """Test custom request."""
        request = RecommendationRequest(
            user_data={"id": 1, "country": "US"},
            num_recommendations=50,
            excluded_video_ids={1, 2, 3},
            filters={"category": "Technology"},
            enable_diversification=False,
        )

        assert request.num_recommendations == 50
        assert request.excluded_video_ids == {1, 2, 3}
        assert request.filters == {"category": "Technology"}
        assert request.enable_diversification is False


class TestSageMakerConfig:
    """Tests for SageMakerConfig."""

    def test_default_config(self):
        """Test default SageMaker config."""
        config = SageMakerConfig()

        assert config.use_serverless is True
        assert config.memory_size_mb == 2048
        assert config.max_concurrency == 10

    def test_custom_config(self):
        """Test custom SageMaker config."""
        config = SageMakerConfig(
            endpoint_name="my-endpoint",
            use_serverless=False,
            instance_type="ml.g4dn.xlarge",
            min_instances=2,
            max_instances=10,
        )

        assert config.endpoint_name == "my-endpoint"
        assert config.use_serverless is False
        assert config.instance_type == "ml.g4dn.xlarge"
        assert config.min_instances == 2
        assert config.max_instances == 10
