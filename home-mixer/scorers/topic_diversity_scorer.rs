use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

/// TopicDiversityScorer reduces the score of candidates whose topics are
/// already over-represented in the ranked list. This prevents the feed from
/// being dominated by a single topic and encourages content variety.
///
/// The scorer works by:
/// 1. Sorting candidates by their current weighted score (descending)
/// 2. Tracking topic occurrence counts as we iterate through the sorted list
/// 3. Applying a multiplicative penalty based on how many times a candidate's
///    topics have already appeared
/// 4. The penalty decays with a configurable floor to avoid over-penalizing
pub struct TopicDiversityScorer;

impl TopicDiversityScorer {
    /// Compute the diversity penalty multiplier for a given topic occurrence count.
    ///
    /// The penalty follows an exponential decay curve:
    ///   multiplier = (1.0 - floor) * decay^count + floor
    ///
    /// Where:
    /// - decay: how quickly the penalty increases per occurrence (0.0-1.0)
    /// - floor: minimum multiplier to avoid zeroing out scores
    /// - count: number of times this topic has already appeared
    fn penalty_multiplier(decay: f64, floor: f64, count: usize) -> f64 {
        if count == 0 {
            return 1.0;
        }
        (1.0 - floor) * decay.powf(count as f64) + floor
    }

    /// Get the effective topic IDs for a candidate, preferring filtered topics
    /// over unfiltered ones when available.
    fn candidate_topic_ids(candidate: &PostCandidate) -> &[i64] {
        match &candidate.filtered_topic_ids {
            Some(ids) if !ids.is_empty() => ids,
            _ => match &candidate.unfiltered_topic_ids {
                Some(ids) if !ids.is_empty() => ids,
                _ => &[],
            },
        }
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for TopicDiversityScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableTopicDiversityScorer)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let decay = query.params.get(TopicDiversityDecay);
        let floor = query.params.get(TopicDiversityFloor);

        // If decay is 1.0, there's no penalty — return unchanged scores
        if decay >= 1.0 || decay <= 0.0 {
            return candidates
                .iter()
                .map(|c| {
                    Ok(PostCandidate {
                        score: c.score,
                        ..Default::default()
                    })
                })
                .collect();
        }

        // Sort candidates by current score (descending) to process highest first
        let mut indexed: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.score.unwrap_or(0.0)))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        let mut topic_counts: HashMap<i64, usize> = HashMap::new();
        let mut adjusted_scores = vec![0.0_f64; candidates.len()];

        for (original_idx, base_score) in indexed {
            let candidate = &candidates[original_idx];
            let topic_ids = Self::candidate_topic_ids(candidate);

            if topic_ids.is_empty() {
                // No topic info — don't penalize
                adjusted_scores[original_idx] = base_score;
                continue;
            }

            // Find the maximum penalty across all topics for this candidate
            // (the most over-represented topic determines the penalty)
            let max_penalty = topic_ids
                .iter()
                .map(|tid| *topic_counts.get(tid).unwrap_or(&0))
                .max()
                .unwrap_or(0);

            let multiplier = Self::penalty_multiplier(decay, floor, max_penalty);
            adjusted_scores[original_idx] = base_score * multiplier;

            // Increment topic counts for all topics this candidate belongs to
            for tid in topic_ids {
                *topic_counts.entry(*tid).or_insert(0) += 1;
            }
        }

        candidates
            .iter()
            .enumerate()
            .map(|(i, _)| {
                Ok(PostCandidate {
                    score: Some(adjusted_scores[i]),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(tweet_id: u64, score: f64, topic_ids: Vec<i64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            score: Some(score),
            filtered_topic_ids: Some(topic_ids),
            ..Default::default()
        }
    }

    #[test]
    fn test_penalty_multiplier_no_decay() {
        // With decay=1.0, penalty is always 1.0 (no penalty)
        assert!((TopicDiversityScorer::penalty_multiplier(1.0, 0.1, 0) - 1.0).abs() < 1e-10);
        assert!((TopicDiversityScorer::penalty_multiplier(1.0, 0.1, 5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_multiplier_with_decay() {
        // With decay=0.5, floor=0.1:
        // count=0: (1-0.1)*0.5^0 + 0.1 = 0.9 + 0.1 = 1.0
        assert!((TopicDiversityScorer::penalty_multiplier(0.5, 0.1, 0) - 1.0).abs() < 1e-10);
        // count=1: (1-0.1)*0.5^1 + 0.1 = 0.45 + 0.1 = 0.55
        assert!((TopicDiversityScorer::penalty_multiplier(0.5, 0.1, 1) - 0.55).abs() < 1e-10);
        // count=2: (1-0.1)*0.5^2 + 0.1 = 0.225 + 0.1 = 0.325
        assert!((TopicDiversityScorer::penalty_multiplier(0.5, 0.1, 2) - 0.325).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_respects_floor() {
        // With decay=0.0, floor=0.2:
        // count=10: (1-0.2)*0.0^10 + 0.2 = 0.0 + 0.2 = 0.2 (floor)
        assert!((TopicDiversityScorer::penalty_multiplier(0.0, 0.2, 10) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_candidate_topic_ids_prefers_filtered() {
        let candidate = PostCandidate {
            filtered_topic_ids: Some(vec![1, 2]),
            unfiltered_topic_ids: Some(vec![3, 4]),
            ..Default::default()
        };
        assert_eq!(TopicDiversityScorer::candidate_topic_ids(&candidate), &[1, 2]);
    }

    #[test]
    fn test_candidate_topic_ids_falls_back_to_unfiltered() {
        let candidate = PostCandidate {
            filtered_topic_ids: None,
            unfiltered_topic_ids: Some(vec![3, 4]),
            ..Default::default()
        };
        assert_eq!(TopicDiversityScorer::candidate_topic_ids(&candidate), &[3, 4]);
    }

    #[test]
    fn test_candidate_topic_ids_empty_when_none() {
        let candidate = PostCandidate {
            filtered_topic_ids: None,
            unfiltered_topic_ids: None,
            ..Default::default()
        };
        assert!(TopicDiversityScorer::candidate_topic_ids(&candidate).is_empty());
    }
}
