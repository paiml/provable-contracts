//! Codebase scoring — how well a consumer project uses contracts.

use std::collections::BTreeSet;

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::{Contract, LeanStatus};

use super::types::{CodebaseScore, Grade, ScoringGap};
use super::score_contract;

/// Score a codebase that consumes contracts via a binding registry.
///
/// Five dimensions (weights from spec):
/// - CD1: Contract coverage (30%) — fraction of available contracts that are bound
/// - CD2: Binding completeness (20%) — implemented / total bindings
/// - CD3: Mean contract score (20%) — avg composite of bound contracts
/// - CD4: Proof depth distribution (15%) — weighted L1-L5 distribution
/// - CD5: Drift detection (15%) — currently 1.0 (no stale detection yet)
#[allow(clippy::cast_precision_loss)]
pub fn score_codebase(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
) -> CodebaseScore {
    let bound_stems: BTreeSet<_> = binding
        .bindings
        .iter()
        .map(|b| b.contract.as_str())
        .collect();

    let total_contracts = contracts.len();
    let bound_count = contracts
        .iter()
        .filter(|(stem, _)| bound_stems.contains(stem.as_str()))
        .count();

    // CD1: Contract coverage
    let contract_coverage = if total_contracts == 0 {
        0.0
    } else {
        bound_count as f64 / total_contracts as f64
    };

    // CD2: Binding completeness
    let total_bindings = binding.bindings.len();
    let implemented_bindings: f64 = binding
        .bindings
        .iter()
        .map(|b| match b.status {
            ImplStatus::Implemented => 1.0,
            ImplStatus::Partial => 0.5,
            ImplStatus::NotImplemented => 0.0,
        })
        .sum();
    let binding_completeness = if total_bindings == 0 {
        0.0
    } else {
        implemented_bindings / total_bindings as f64
    };

    // CD3: Mean contract score of bound contracts
    let bound_scores: Vec<f64> = contracts
        .iter()
        .filter(|(stem, _)| bound_stems.contains(stem.as_str()))
        .map(|(stem, c)| score_contract(c, Some(binding), stem).composite)
        .collect();
    let mean_contract_score = if bound_scores.is_empty() {
        0.0
    } else {
        bound_scores.iter().sum::<f64>() / bound_scores.len() as f64
    };

    // CD4: Proof depth distribution (weighted L1-L5)
    let proof_depth_dist = compute_proof_depth(contracts, &bound_stems);

    // CD5: Drift detection (placeholder — always fresh for now)
    let drift = 1.0;

    let composite = contract_coverage * 0.30
        + binding_completeness * 0.20
        + mean_contract_score * 0.20
        + proof_depth_dist * 0.15
        + drift * 0.15;

    let top_gaps = compute_gaps(contracts, binding, &bound_stems);

    CodebaseScore {
        path: "codebase".to_string(),
        contract_coverage,
        binding_completeness,
        mean_contract_score,
        proof_depth_dist,
        drift,
        composite,
        grade: Grade::from_score(composite),
        top_gaps,
    }
}

#[allow(clippy::cast_precision_loss)]
fn compute_proof_depth(
    contracts: &[(String, &Contract)],
    bound_stems: &BTreeSet<&str>,
) -> f64 {
    let mut total_obligations = 0usize;
    let mut weighted_sum = 0.0;

    for (stem, contract) in contracts {
        if !bound_stems.contains(stem.as_str()) {
            continue;
        }
        for ob in &contract.proof_obligations {
            total_obligations += 1;
            weighted_sum += 0.1; // L1 (type system)
            if !contract.falsification_tests.is_empty() {
                weighted_sum += 0.3; // L3 (probar)
            }
            if !contract.kani_harnesses.is_empty() {
                weighted_sum += 0.4; // L4 (Kani)
            }
            if ob
                .lean
                .as_ref()
                .is_some_and(|l| l.status == LeanStatus::Proved)
            {
                weighted_sum += 0.2; // L5 (Lean)
            }
        }
    }

    if total_obligations == 0 {
        return 0.0;
    }
    (weighted_sum / total_obligations as f64).min(1.0)
}

#[allow(clippy::cast_precision_loss)]
fn compute_gaps(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    bound_stems: &BTreeSet<&str>,
) -> Vec<ScoringGap> {
    let mut gaps = Vec::new();

    for (stem, contract) in contracts {
        if !bound_stems.contains(stem.as_str()) {
            continue;
        }
        let ob_count = contract.proof_obligations.len();
        let kani_count = contract.kani_harnesses.len();
        let ft_count = contract.falsification_tests.len();

        if ob_count > 0 && kani_count < ob_count {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "kani_coverage".into(),
                current: kani_count as f64 / ob_count as f64,
                target: 1.0,
                impact: (ob_count - kani_count) as f64,
            });
        }

        if ob_count > 0 && ft_count < ob_count {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "falsification_coverage".into(),
                current: ft_count as f64 / ob_count as f64,
                target: 1.0,
                impact: (ob_count - ft_count) as f64,
            });
        }

        let unimpl_count = binding
            .bindings
            .iter()
            .filter(|b| b.contract == *stem && b.status == ImplStatus::NotImplemented)
            .count();
        if unimpl_count > 0 {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "binding_coverage".into(),
                current: 0.0,
                target: 1.0,
                impact: unimpl_count as f64,
            });
        }
    }

    gaps.sort_by(|a, b| b.impact.partial_cmp(&a.impact).unwrap_or(std::cmp::Ordering::Equal));
    gaps.truncate(10);
    gaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_contracts_and_binding() -> (Vec<(String, Contract)>, BindingRegistry) {
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let content = std::fs::read_to_string(binding_path).unwrap();
        let binding: BindingRegistry = serde_yaml::from_str(&content).unwrap();

        let contracts_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let mut parsed = Vec::new();
        for entry in std::fs::read_dir(contracts_dir).unwrap().flatten() {
            let p = entry.path();
            if p.extension().and_then(|x| x.to_str()) != Some("yaml") {
                continue;
            }
            let Ok(c) = crate::schema::parse_contract(&p) else {
                continue;
            };
            // Binding uses filename WITH .yaml extension as the stem
            let stem = p.file_name().unwrap().to_str().unwrap().to_string();
            parsed.push((stem, c));
        }
        (parsed, binding)
    }

    #[test]
    fn score_codebase_with_binding() {
        let (parsed, binding) = load_contracts_and_binding();
        let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();
        let score = score_codebase(&contracts, &binding);
        assert!(score.contract_coverage > 0.0);
        assert!(score.binding_completeness > 0.5);
        assert!(score.composite > 0.20, "composite={}", score.composite);
        assert!(!score.top_gaps.is_empty());
    }

    #[test]
    fn codebase_display_format() {
        let score = CodebaseScore {
            path: "test".into(),
            contract_coverage: 0.5,
            binding_completeness: 0.8,
            mean_contract_score: 0.6,
            proof_depth_dist: 0.4,
            drift: 1.0,
            composite: 0.6,
            grade: Grade::C,
            top_gaps: vec![ScoringGap {
                contract: "softmax".into(),
                dimension: "kani".into(),
                current: 0.3,
                target: 1.0,
                impact: 4.0,
            }],
        };
        let text = score.to_string();
        assert!(text.contains("Grade C"));
        assert!(text.contains("softmax"));
    }

    #[test]
    fn empty_binding_scores_low() {
        let binding = BindingRegistry {
            version: "1.0.0".into(),
            target_crate: "test".into(),
            bindings: Vec::new(),
        };
        let score = score_codebase(&[], &binding);
        assert_eq!(score.contract_coverage, 0.0);
        assert_eq!(score.composite, 0.15); // only drift=1.0 * 0.15
        assert_eq!(score.grade, Grade::F);
    }
}
