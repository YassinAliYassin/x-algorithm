"""
Post Freshness Classifier for Grox Content Understanding Pipeline.

Scores content based on recency signals to help the ranking system
prioritize fresh content while still allowing evergreen posts to surface.
"""

import logging
import time
from datetime import datetime, timezone

from grox.data_loaders.data_types import (
    Post,
    ContentCategoryType,
    ContentCategoryResult,
)
from grox.classifiers.content.classifier import ContentClassifier

logger = logging.getLogger(__name__)


# Freshness scoring thresholds (in hours)
FRESHNESS_VERY_FRESH_HOURS = 2
FRESHNESS_FRESH_HOURS = 6
FRESHNESS_MODERATE_HOURS = 24
FRESHNESS_STALE_HOURS = 72
FRESHNESS_VERY_STALE_HOURS = 168  # 1 week


class PostFreshnessClassifier(ContentClassifier):
    """
    Classifies post freshness into categories based on age.
    
    This is a rule-based classifier (no LLM needed) that categorizes
    posts by age into freshness buckets. The freshness score can be
    used as a signal in the ranking pipeline to balance recency vs.
    relevance.
    
    Freshness categories:
    - VERY_FRESH (0-2h): Breaking news, live events
    - FRESH (2-6h): Recent content, trending discussions
    - MODERATE (6-24h): Day-old content, still relevant
    - STALE (24-72h): Multi-day content, relevance declining
    - VERY_STALE (72h+): Old content, only surface if highly relevant
    """

    FRESHNESS_CATEGORIES = [
        ContentCategoryType.FRESHNESS_VERY_FRESH,
        ContentCategoryType.FRESHNESS_FRESH,
        ContentCategoryType.FRESHNESS_MODERATE,
        ContentCategoryType.FRESHNESS_STALE,
        ContentCategoryType.FRESHNESS_VERY_STALE,
    ]

    def __init__(self):
        # No LLM needed - this is a rule-based classifier
        super().__init__(categories=self.FRESHNESS_CATEGORIES, llm=None)

    @property
    def model_name(self) -> str:
        return "freshness_rule_based"

    def _compute_age_hours(self, post: Post) -> float:
        """Compute the age of a post in hours."""
        now = time.time()
        
        # Try to get creation timestamp from post metadata
        created_at = None
        if hasattr(post, 'created_at') and post.created_at:
            created_at = post.created_at
        elif hasattr(post, 'timestamp') and post.timestamp:
            created_at = post.timestamp
        
        if created_at is None:
            # If no timestamp available, assume moderate freshness
            logger.debug(f"No timestamp for post {post.id}, defaulting to moderate freshness")
            return FRESHNESS_MODERATE_HOURS
        
        # Handle both epoch seconds and datetime objects
        if isinstance(created_at, datetime):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            post_timestamp = created_at.timestamp()
        else:
            # Assume epoch seconds (or milliseconds)
            post_timestamp = created_at
            if post_timestamp > 1e12:  # Likely milliseconds
                post_timestamp /= 1000.0
        
        age_seconds = max(0, now - post_timestamp)
        age_hours = age_seconds / 3600.0
        return age_hours

    def _score_to_freshness(self, age_hours: float) -> tuple[ContentCategoryType, float]:
        """
        Convert age in hours to a freshness category and score.
        
        Returns:
            Tuple of (category, score) where score is 1.0 for very fresh
            and decreases for older content.
        """
        if age_hours <= FRESHNESS_VERY_FRESH_HOURS:
            return ContentCategoryType.FRESHNESS_VERY_FRESH, 1.0
        elif age_hours <= FRESHNESS_FRESH_HOURS:
            # Linear decay from 1.0 to 0.8
            t = (age_hours - FRESHNESS_VERY_FRESH_HOURS) / (
                FRESHNESS_FRESH_HOURS - FRESHNESS_VERY_FRESH_HOURS
            )
            return ContentCategoryType.FRESHNESS_FRESH, 1.0 - 0.2 * t
        elif age_hours <= FRESHNESS_MODERATE_HOURS:
            # Linear decay from 0.8 to 0.5
            t = (age_hours - FRESHNESS_FRESH_HOURS) / (
                FRESHNESS_MODERATE_HOURS - FRESHNESS_FRESH_HOURS
            )
            return ContentCategoryType.FRESHNESS_MODERATE, 0.8 - 0.3 * t
        elif age_hours <= FRESHNESS_STALE_HOURS:
            # Linear decay from 0.5 to 0.2
            t = (age_hours - FRESHNESS_MODERATE_HOURS) / (
                FRESHNESS_STALE_HOURS - FRESHNESS_MODERATE_HOURS
            )
            return ContentCategoryType.FRESHNESS_STALE, 0.5 - 0.3 * t
        else:
            # Asymptotic decay towards 0.1 for very old content
            weeks_old = age_hours / 168.0
            score = max(0.1, 0.2 * (0.5 ** (weeks_old - 1)))
            return ContentCategoryType.FRESHNESS_VERY_STALE, score

    async def _classify(self, post: Post) -> list[ContentCategoryResult]:
        """Classify a post's freshness based on its age."""
        age_hours = self._compute_age_hours(post)
        category, score = self._score_to_freshness(age_hours)

        logger.debug(
            f"Post {post.id} age={age_hours:.1f}h -> {category.name} (score={score:.3f})"
        )

        return [
            ContentCategoryResult(
                category=cat,
                positive=(cat == category and score >= 0.5),
                score=score if cat == category else 0.0,
                summary=f"Post age: {age_hours:.1f} hours",
            )
            for cat in self.FRESHNESS_CATEGORIES
        ]
