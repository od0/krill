//! Embedding-based skill retrieval.
//!
//! Implements the retrieval formula from the paper:
//!
//!   S_ret = TopK({s in S_k : sim(e_d, e_s) > delta}, K)
//!
//! where `e_d` is the embedding of the task description, `e_s` is the skill embedding,
//! `delta` is the similarity threshold, and `K` is the number of skills to return.

use ordered_float::OrderedFloat;

use super::library::SkillBank;
use super::types::Skill;
use crate::config::SkillRetrievalConfig;

/// Computes the cosine similarity between two vectors.
///
/// Returns 0.0 if either vector is the zero vector (to avoid division by zero).
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Retrieves the most relevant skills for a given task from the skill bank.
///
/// The retriever searches the task-specific pool for the given category and
/// falls back to including general skills as well. Skills are ranked by cosine
/// similarity between the task description embedding and each skill's embedding.
pub struct SkillRetriever;

impl SkillRetriever {
    /// Retrieve the top-K most relevant skills for a task.
    ///
    /// # Algorithm
    ///
    /// 1. Collect candidate skills from:
    ///    - The task-specific pool for `task_category`
    ///    - The general skill pool
    /// 2. For each candidate that has an embedding, compute cosine similarity with
    ///    `task_description_embedding`.
    /// 3. Filter candidates whose similarity exceeds `config.similarity_threshold` (delta).
    /// 4. Sort by descending similarity and return the top `config.top_k` skills.
    ///
    /// # Arguments
    ///
    /// * `task_description_embedding` - The embedding vector of the current task description.
    /// * `task_category` - The category of the current task (used to select the task-specific pool).
    /// * `bank` - The skill bank to retrieve from.
    /// * `config` - Retrieval configuration (top_k, similarity_threshold).
    ///
    /// # Returns
    ///
    /// A vector of references to the top-K most similar skills, ordered by descending similarity.
    pub fn retrieve<'a>(
        task_description_embedding: &[f64],
        task_category: &str,
        bank: &'a SkillBank,
        config: &SkillRetrievalConfig,
    ) -> Vec<&'a Skill> {
        // Collect candidates: task-specific + general skills.
        let task_skills = bank.get_task_skills(task_category);
        let general_skills = bank.get_general_skills();

        let candidates = task_skills.iter().chain(general_skills.iter());

        // Score each candidate.
        let mut scored: Vec<(OrderedFloat<f64>, &Skill)> = candidates
            .filter_map(|skill| {
                let emb = skill.embedding.as_ref()?;
                let sim = cosine_similarity(task_description_embedding, emb);
                if sim > config.similarity_threshold {
                    Some((OrderedFloat(sim), skill))
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by similarity.
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        // Take top-K.
        scored
            .into_iter()
            .take(config.top_k)
            .map(|(_, skill)| skill)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::types::{Skill, SkillCategory};

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    fn make_skill_with_embedding(name: &str, category: SkillCategory, emb: Vec<f64>) -> Skill {
        let mut s = Skill::new(name, "principle", "when", category);
        s.embedding = Some(emb);
        s
    }

    #[test]
    fn test_retrieve_top_k() {
        let mut bank = SkillBank::new();

        // Add task-specific skills with varying similarity to [1, 0, 0].
        bank.add_skill(make_skill_with_embedding(
            "close_match",
            SkillCategory::TaskSpecific("web".into()),
            vec![0.9, 0.1, 0.0],
        ));
        bank.add_skill(make_skill_with_embedding(
            "medium_match",
            SkillCategory::TaskSpecific("web".into()),
            vec![0.5, 0.5, 0.0],
        ));
        bank.add_skill(make_skill_with_embedding(
            "poor_match",
            SkillCategory::TaskSpecific("web".into()),
            vec![0.0, 0.0, 1.0],
        ));
        // Add a general skill.
        bank.add_skill(make_skill_with_embedding(
            "general_close",
            SkillCategory::General,
            vec![0.95, 0.05, 0.0],
        ));

        let query = vec![1.0, 0.0, 0.0];
        let config = SkillRetrievalConfig {
            top_k: 2,
            similarity_threshold: 0.4,
        };

        let results = SkillRetriever::retrieve(&query, "web", &bank, &config);

        // Should get the two closest (general_close ~ 0.998, close_match ~ 0.994).
        assert_eq!(results.len(), 2);
        // The best match should come first.
        assert_eq!(results[0].name, "general_close");
        assert_eq!(results[1].name, "close_match");
    }

    #[test]
    fn test_retrieve_threshold_filtering() {
        let mut bank = SkillBank::new();
        bank.add_skill(make_skill_with_embedding(
            "orthogonal",
            SkillCategory::TaskSpecific("web".into()),
            vec![0.0, 1.0],
        ));

        let query = vec![1.0, 0.0];
        let config = SkillRetrievalConfig {
            top_k: 10,
            similarity_threshold: 0.5,
        };

        let results = SkillRetriever::retrieve(&query, "web", &bank, &config);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_no_embedding_skipped() {
        let mut bank = SkillBank::new();
        // Skill without an embedding should be skipped.
        bank.add_skill(Skill::new(
            "no_emb",
            "principle",
            "when",
            SkillCategory::TaskSpecific("web".into()),
        ));

        let query = vec![1.0, 0.0];
        let config = SkillRetrievalConfig {
            top_k: 10,
            similarity_threshold: 0.0,
        };

        let results = SkillRetriever::retrieve(&query, "web", &bank, &config);
        assert!(results.is_empty());
    }
}
