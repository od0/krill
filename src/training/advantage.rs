//! GRPO advantage estimation utilities.
//!
//! Implements the group-relative advantage normalization from the SkillRL paper:
//!
//!   A_i = (R_i - mean({R_j}_{j=1}^G)) / std({R_j}_{j=1}^G)
//!
//! Along with importance ratio computation and PPO-style clipping used in the
//! GRPO objective (Equation 9).

/// Compute group-relative advantages for a group of G rewards.
///
/// Each advantage is the z-score of the reward within the group:
///
///   A_i = (R_i - mean(R)) / std(R)
///
/// # Edge cases
///
/// - If `rewards` is empty, returns an empty vector.
/// - If all rewards are identical (std = 0), all advantages are set to 0.0.
///   This is the correct behaviour: when all outputs receive the same reward,
///   no output should be preferred over another.
pub fn compute_group_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }

    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;

    // Population standard deviation (we are normalizing within a fixed group,
    // not estimating a population parameter, so we divide by N rather than N-1).
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    // Guard against division by zero when all rewards are identical.
    if std < 1e-8 {
        return vec![0.0; rewards.len()];
    }

    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// Compute the importance sampling ratio between the current and old policies.
///
///   rho_i = exp(log pi_theta - log pi_old)
///
/// This is the probability ratio pi_theta(a|s) / pi_old(a|s) expressed in
/// log-space for numerical stability.
pub fn compute_importance_ratio(current_log_prob: f64, old_log_prob: f64) -> f64 {
    (current_log_prob - old_log_prob).exp()
}

/// Clip an importance ratio to the interval [1 - epsilon, 1 + epsilon].
///
/// This is the PPO clipping mechanism that prevents the policy from changing
/// too drastically in a single update step.
pub fn clip_ratio(ratio: f64, epsilon: f64) -> f64 {
    ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // compute_group_advantages
    // ------------------------------------------------------------------

    #[test]
    fn test_advantages_basic() {
        // Simple group: [0, 0, 1, 1]
        let rewards = vec![0.0, 0.0, 1.0, 1.0];
        let advs = compute_group_advantages(&rewards);
        assert_eq!(advs.len(), 4);

        // Mean = 0.5, std = 0.5
        // A_0 = (0 - 0.5) / 0.5 = -1.0
        // A_2 = (1 - 0.5) / 0.5 =  1.0
        assert!((advs[0] - (-1.0)).abs() < 1e-9);
        assert!((advs[1] - (-1.0)).abs() < 1e-9);
        assert!((advs[2] - 1.0).abs() < 1e-9);
        assert!((advs[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_advantages_all_same() {
        // When all rewards are identical, advantages should be zero.
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let advs = compute_group_advantages(&rewards);
        for a in &advs {
            assert!(a.abs() < 1e-9, "Expected 0.0, got {a}");
        }
    }

    #[test]
    fn test_advantages_empty() {
        let advs = compute_group_advantages(&[]);
        assert!(advs.is_empty());
    }

    #[test]
    fn test_advantages_single_element() {
        // Single element means std = 0, so advantage should be 0.
        let advs = compute_group_advantages(&[0.5]);
        assert_eq!(advs.len(), 1);
        assert!(advs[0].abs() < 1e-9);
    }

    #[test]
    fn test_advantages_sum_to_zero() {
        // Z-scores of any distribution sum to zero (up to floating point).
        let rewards = vec![0.1, 0.4, 0.7, 0.9, 1.0, 0.0, 0.3, 0.6];
        let advs = compute_group_advantages(&rewards);
        let sum: f64 = advs.iter().sum();
        assert!(sum.abs() < 1e-9, "Sum of advantages should be ~0, got {sum}");
    }

    #[test]
    fn test_advantages_binary_rewards() {
        // Typical GRPO scenario: binary rewards in a group of 8.
        let rewards = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let advs = compute_group_advantages(&rewards);
        // Successful trajectories should have positive advantage.
        assert!(advs[1] > 0.0);
        assert!(advs[4] > 0.0);
        assert!(advs[6] > 0.0);
        // Failed trajectories should have negative advantage.
        assert!(advs[0] < 0.0);
        assert!(advs[2] < 0.0);
    }

    // ------------------------------------------------------------------
    // compute_importance_ratio
    // ------------------------------------------------------------------

    #[test]
    fn test_importance_ratio_same_policy() {
        // When current == old, ratio should be 1.0.
        let ratio = compute_importance_ratio(-2.5, -2.5);
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_importance_ratio_higher_prob() {
        // Current policy assigns higher probability -> ratio > 1.
        let ratio = compute_importance_ratio(-1.0, -2.0);
        // exp(-1.0 - (-2.0)) = exp(1.0) ~ 2.718
        assert!((ratio - 1.0_f64.exp()).abs() < 1e-6);
    }

    #[test]
    fn test_importance_ratio_lower_prob() {
        // Current policy assigns lower probability -> ratio < 1.
        let ratio = compute_importance_ratio(-3.0, -2.0);
        // exp(-3.0 - (-2.0)) = exp(-1.0) ~ 0.368
        assert!((ratio - (-1.0_f64).exp()).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // clip_ratio
    // ------------------------------------------------------------------

    #[test]
    fn test_clip_ratio_within_bounds() {
        let clipped = clip_ratio(1.1, 0.2);
        assert!((clipped - 1.1).abs() < 1e-9);
    }

    #[test]
    fn test_clip_ratio_above_upper() {
        let clipped = clip_ratio(1.5, 0.2);
        assert!((clipped - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_clip_ratio_below_lower() {
        let clipped = clip_ratio(0.5, 0.2);
        assert!((clipped - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_clip_ratio_at_boundaries() {
        assert!((clip_ratio(0.8, 0.2) - 0.8).abs() < 1e-9);
        assert!((clip_ratio(1.2, 0.2) - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_clip_ratio_exact_one() {
        let clipped = clip_ratio(1.0, 0.2);
        assert!((clipped - 1.0).abs() < 1e-9);
    }
}
