"""
Ranker quality sampling for measuring ranking model effectiveness.

Samples videos outside the top-k recommendations to collect user feedback
and measure the quality of the ranking model's decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random
import numpy as np

from ..utils.logging_utils import get_logger
from .monitoring_config import OnlineMetricsConfig

logger = get_logger(__name__)


@dataclass
class RankerSample:
    """A sampled video for quality measurement."""

    video_id: int
    original_rank: int  # Rank from the ranker (e.g., 15 out of 30)
    displayed_position: int  # Position shown to user
    ranker_score: float
    retrieval_score: float

    # User feedback
    was_clicked: bool = False
    was_completed: bool = False
    watch_time_seconds: float = 0.0
    explicit_feedback: Optional[str] = None  # "like", "dislike", None

    # Metadata
    sample_id: str = ""
    request_id: str = ""
    user_id: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class RankerQualityReport:
    """Report on ranker quality based on sampled feedback."""

    report_id: str
    generated_at: str
    total_samples: int

    # Stratified metrics by rank tier
    metrics_by_tier: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Overall metrics
    sampled_ctr: float = 0.0
    sampled_completion_rate: float = 0.0
    avg_watch_time: float = 0.0

    # Comparison with top-k
    top_k_ctr: float = 0.0
    ctr_gap: float = 0.0  # Expected drop from top-k to sampled

    # Quality score
    ranking_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "total_samples": self.total_samples,
            "metrics_by_tier": self.metrics_by_tier,
            "sampled_ctr": self.sampled_ctr,
            "sampled_completion_rate": self.sampled_completion_rate,
            "avg_watch_time": self.avg_watch_time,
            "top_k_ctr": self.top_k_ctr,
            "ctr_gap": self.ctr_gap,
            "ranking_quality_score": self.ranking_quality_score,
        }


class RankerQualitySampler:
    """Sampler for measuring ranker quality.

    The idea is to show users some videos outside the top-k results
    to measure whether the ranker is correctly ordering videos.

    If videos ranked 20-30 perform similarly to videos ranked 1-10,
    the ranker may not be effective.

    Example:
        >>> sampler = RankerQualitySampler(config)
        >>> modified_recs = sampler.inject_samples(
        ...     recommendations=ranked_videos,
        ...     request_id="req_123",
        ...     user_id=456,
        ... )
        >>> # Later, record feedback
        >>> sampler.record_click(sample_id, was_clicked=True)
        >>> report = sampler.generate_report()
    """

    def __init__(self, config: OnlineMetricsConfig):
        """Initialize the ranker quality sampler.

        Args:
            config: Online metrics configuration.
        """
        self.config = config
        self._samples: List[RankerSample] = []
        self._sample_counter = 0
        self._samples_by_id: Dict[str, RankerSample] = {}

    def inject_samples(
        self,
        recommendations: List[Dict[str, Any]],
        request_id: str,
        user_id: int,
        top_k: int = 10,
        sample_probability: float = 0.2,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Inject sampled videos into recommendations.

        Takes the full ranked list and injects some lower-ranked videos
        into the top-k results for quality measurement.

        Args:
            recommendations: Full list of ranked recommendations.
            request_id: Request identifier.
            user_id: User identifier.
            top_k: Number of top recommendations to show.
            sample_probability: Probability of injecting samples.

        Returns:
            Tuple of (modified recommendations, list of sample IDs).
        """
        if not self.config.enable_ranker_sampling:
            return recommendations[:top_k], []

        # Decide whether to sample this request
        if random.random() > sample_probability:
            return recommendations[:top_k], []

        # Get videos to potentially sample from (outside top-k)
        sample_from_top_n = min(
            self.config.ranker_sample_from_top_n,
            len(recommendations),
        )

        if sample_from_top_n <= top_k:
            return recommendations[:top_k], []

        # Pool of candidates for sampling (ranks top_k+1 to sample_from_top_n)
        sample_pool = recommendations[top_k:sample_from_top_n]

        if not sample_pool:
            return recommendations[:top_k], []

        # Number of samples to inject
        num_samples = min(
            self.config.ranker_sample_size,
            len(sample_pool),
            top_k // 3,  # Don't inject more than 1/3 of the list
        )

        # Select samples
        sampled_recs = random.sample(sample_pool, num_samples)

        # Create sample records
        sample_ids = []
        for rec in sampled_recs:
            sample = self._create_sample(rec, request_id, user_id)
            sample_ids.append(sample.sample_id)

        # Build modified recommendations
        # Remove sampled videos from their original positions
        sampled_video_ids = {rec["video_id"] for rec in sampled_recs}

        # Take top-k minus num_samples from original list
        top_recs = [
            r for r in recommendations[:top_k]
            if r["video_id"] not in sampled_video_ids
        ][:top_k - num_samples]

        # Inject samples at random positions within top-k
        modified_recs = top_recs.copy()
        for rec in sampled_recs:
            insert_pos = random.randint(len(modified_recs) // 2, len(modified_recs))
            modified_recs.insert(insert_pos, rec)

        # Update displayed positions in samples
        for i, rec in enumerate(modified_recs):
            video_id = rec["video_id"]
            for sample in self._samples:
                if sample.video_id == video_id and sample.request_id == request_id:
                    sample.displayed_position = i + 1

        return modified_recs, sample_ids

    def _create_sample(
        self,
        recommendation: Dict[str, Any],
        request_id: str,
        user_id: int,
    ) -> RankerSample:
        """Create a sample record.

        Args:
            recommendation: Recommendation dict.
            request_id: Request identifier.
            user_id: User identifier.

        Returns:
            Ranker sample.
        """
        self._sample_counter += 1
        sample_id = f"sample_{self._sample_counter}"

        sample = RankerSample(
            sample_id=sample_id,
            video_id=recommendation["video_id"],
            original_rank=recommendation.get("rank", 0),
            displayed_position=0,  # Set later
            ranker_score=recommendation.get("ranker_score", 0),
            retrieval_score=recommendation.get("retrieval_score", 0),
            request_id=request_id,
            user_id=user_id,
        )

        self._samples.append(sample)
        self._samples_by_id[sample_id] = sample

        return sample

    def record_feedback(
        self,
        sample_id: str,
        was_clicked: bool = False,
        was_completed: bool = False,
        watch_time_seconds: float = 0.0,
        explicit_feedback: Optional[str] = None,
    ) -> None:
        """Record user feedback for a sample.

        Args:
            sample_id: Sample identifier.
            was_clicked: Whether the video was clicked.
            was_completed: Whether the video was completed.
            watch_time_seconds: Time spent watching.
            explicit_feedback: Explicit feedback ("like" or "dislike").
        """
        sample = self._samples_by_id.get(sample_id)
        if not sample:
            logger.warning(f"Sample not found: {sample_id}")
            return

        sample.was_clicked = was_clicked
        sample.was_completed = was_completed
        sample.watch_time_seconds = watch_time_seconds
        sample.explicit_feedback = explicit_feedback

    def record_click(self, video_id: int, request_id: str) -> None:
        """Record a click on a sampled video.

        Args:
            video_id: Video identifier.
            request_id: Request identifier.
        """
        for sample in self._samples:
            if sample.video_id == video_id and sample.request_id == request_id:
                sample.was_clicked = True
                return

    def generate_report(
        self,
        top_k_ctr: float = 0.0,
    ) -> RankerQualityReport:
        """Generate ranker quality report.

        Args:
            top_k_ctr: CTR of top-k recommendations for comparison.

        Returns:
            Ranker quality report.
        """
        import uuid

        report = RankerQualityReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow().isoformat(),
            total_samples=len(self._samples),
            top_k_ctr=top_k_ctr,
        )

        if not self._samples:
            return report

        # Overall metrics
        clicks = sum(1 for s in self._samples if s.was_clicked)
        completions = sum(1 for s in self._samples if s.was_completed)
        watch_times = [s.watch_time_seconds for s in self._samples if s.watch_time_seconds > 0]

        report.sampled_ctr = clicks / len(self._samples)
        if clicks > 0:
            report.sampled_completion_rate = completions / clicks
        if watch_times:
            report.avg_watch_time = np.mean(watch_times)

        # CTR gap (how much worse are sampled videos)
        if top_k_ctr > 0:
            report.ctr_gap = (top_k_ctr - report.sampled_ctr) / top_k_ctr

        # Metrics by rank tier
        tiers = [
            ("tier_11_15", 11, 15),
            ("tier_16_20", 16, 20),
            ("tier_21_25", 21, 25),
            ("tier_26_30", 26, 30),
        ]

        for tier_name, rank_min, rank_max in tiers:
            tier_samples = [
                s for s in self._samples
                if rank_min <= s.original_rank <= rank_max
            ]

            if tier_samples:
                tier_clicks = sum(1 for s in tier_samples if s.was_clicked)
                report.metrics_by_tier[tier_name] = {
                    "sample_count": len(tier_samples),
                    "ctr": tier_clicks / len(tier_samples),
                    "avg_ranker_score": np.mean([s.ranker_score for s in tier_samples]),
                }

        # Compute ranking quality score
        # Higher score = ranker is doing a good job (big gap between top-k and sampled)
        # Lower score = ranker is not effective (similar performance)
        if top_k_ctr > 0 and report.sampled_ctr >= 0:
            # Normalize to 0-1 scale
            # If sampled CTR is 0, quality score is 1 (perfect ranking)
            # If sampled CTR equals top_k CTR, quality score is 0 (random ranking)
            report.ranking_quality_score = min(1.0, report.ctr_gap)
        else:
            report.ranking_quality_score = 0.5  # Unknown

        logger.info(
            f"Ranker quality report: {report.total_samples} samples, "
            f"sampled CTR={report.sampled_ctr:.2%}, "
            f"quality score={report.ranking_quality_score:.2f}"
        )

        return report

    def clear_samples(self) -> None:
        """Clear all stored samples."""
        self._samples = []
        self._samples_by_id = {}

    def get_sample(self, sample_id: str) -> Optional[RankerSample]:
        """Get sample by ID.

        Args:
            sample_id: Sample identifier.

        Returns:
            Sample or None.
        """
        return self._samples_by_id.get(sample_id)

    def get_samples_for_request(self, request_id: str) -> List[RankerSample]:
        """Get all samples for a request.

        Args:
            request_id: Request identifier.

        Returns:
            List of samples.
        """
        return [s for s in self._samples if s.request_id == request_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics.

        Returns:
            Dictionary with stats.
        """
        if not self._samples:
            return {
                "total_samples": 0,
                "click_rate": 0,
            }

        clicks = sum(1 for s in self._samples if s.was_clicked)
        return {
            "total_samples": len(self._samples),
            "click_rate": clicks / len(self._samples),
            "avg_original_rank": np.mean([s.original_rank for s in self._samples]),
            "unique_requests": len(set(s.request_id for s in self._samples)),
        }
