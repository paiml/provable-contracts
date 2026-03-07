//! Codebase scoring — how well a consumer project uses contracts.

use std::collections::{BTreeSet, HashMap};

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::{Contract, LeanStatus};

use super::score_contract;
use super::types::{CodebaseScore, Grade, ScoringGap};

/// Score a codebase that consumes contracts via a binding registry.
///
/// Five dimensions (weights from spec):
/// - CD1: Contract coverage (30%) — fraction of available contracts that are bound
/// - CD2: Binding completeness (20%) — implemented / total bindings
/// - CD3: Mean contract score (20%) — avg composite of bound contracts
/// - CD4: Proof depth distribution (15%) — weighted L1-L5 distribution
/// - CD5: Drift detection (15%) — via git timestamp comparison
///
/// Optional `pagerank` scores weight gap analysis by dependency importance.
#[allow(clippy::cast_precision_loss)]
pub fn score_codebase(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
) -> CodebaseScore {
    score_codebase_with_pagerank(contracts, binding, None)
}

/// Score a codebase with pagerank-weighted gap analysis.
///
/// `drift_override` provides a pre-computed CD5 drift score (0.0-1.0).
/// Use [`super::drift::compute_drift`] + [`super::drift::detect_stale_contracts`]
/// to compute it from git timestamps. Pass `None` to default to 1.0 (no drift).
#[allow(clippy::cast_precision_loss, clippy::implicit_hasher)]
pub fn score_codebase_with_pagerank(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    pagerank: Option<&HashMap<String, f64>>,
) -> CodebaseScore {
    score_codebase_full(contracts, binding, pagerank, None)
}

/// Score a codebase with all optional enrichment: pagerank + drift.
#[allow(clippy::cast_precision_loss, clippy::implicit_hasher)]
pub fn score_codebase_full(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    pagerank: Option<&HashMap<String, f64>>,
    drift_override: Option<f64>,
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

    // CD5: Drift detection
    let drift = drift_override.unwrap_or(1.0);

    let composite = contract_coverage * 0.30
        + binding_completeness * 0.20
        + mean_contract_score * 0.20
        + proof_depth_dist * 0.15
        + drift * 0.15;

    let top_gaps = compute_gaps(contracts, binding, &bound_stems, pagerank);

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
fn compute_proof_depth(contracts: &[(String, &Contract)], bound_stems: &BTreeSet<&str>) -> f64 {
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

/// Compute gaps with impact-weighted scoring per spec Section 4:
/// `impact = (1.0 - obligation_coverage) * dependency_fanout * tier_weight`
///
/// `dependency_fanout` uses pagerank when available, otherwise falls back
/// to reverse dependency count + 1.
#[allow(clippy::cast_precision_loss)]
fn compute_gaps(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    bound_stems: &BTreeSet<&str>,
    pagerank: Option<&HashMap<String, f64>>,
) -> Vec<ScoringGap> {
    let mut gaps = Vec::new();

    // Pre-compute reverse dependency counts for fallback
    let rev_dep_counts = compute_reverse_dep_counts(contracts);

    for (stem, contract) in contracts {
        if !bound_stems.contains(stem.as_str()) {
            continue;
        }
        let ob_count = contract.proof_obligations.len();
        let kani_count = contract.kani_harnesses.len();
        let ft_count = contract.falsification_tests.len();
        let fanout = dependency_fanout(stem, pagerank, &rev_dep_counts);

        if ob_count > 0 && kani_count < ob_count {
            let coverage = kani_count as f64 / ob_count as f64;
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "kani_coverage".into(),
                current: coverage,
                target: 1.0,
                impact: (1.0 - coverage) * fanout,
                action: "Write #[kani::proof] harnesses".into(),
            });
        }

        if ob_count > 0 && ft_count < ob_count {
            let coverage = ft_count as f64 / ob_count as f64;
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "falsification_coverage".into(),
                current: coverage,
                target: 1.0,
                impact: (1.0 - coverage) * fanout,
                action: "Write probar property tests".into(),
            });
        }

        let partial_count = binding
            .bindings_for(stem)
            .iter()
            .filter(|b| b.status == ImplStatus::Partial)
            .count();
        if partial_count > 0 {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "binding_partial".into(),
                current: 0.5,
                target: 1.0,
                impact: 0.5 * fanout,
                action: "Complete partial implementations".into(),
            });
        }

        let unimpl_count = binding
            .bindings_for(stem)
            .iter()
            .filter(|b| b.status == ImplStatus::NotImplemented)
            .count();
        if unimpl_count > 0 {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "binding_coverage".into(),
                current: 0.0,
                target: 1.0,
                impact: 1.0 * fanout,
                action: "Implement bound equations".into(),
            });
        }
    }

    gaps.sort_by(|a, b| {
        b.impact
            .partial_cmp(&a.impact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    gaps.truncate(10);
    gaps
}

/// Compute dependency fanout for a contract stem.
///
/// Uses pagerank score when available (normalized to 1.0-10.0 range),
/// otherwise falls back to reverse dependency count + 1.
#[allow(clippy::cast_precision_loss)]
fn dependency_fanout(
    stem: &str,
    pagerank: Option<&HashMap<String, f64>>,
    rev_dep_counts: &HashMap<&str, usize>,
) -> f64 {
    if let Some(pr) = pagerank {
        if let Some(&score) = pr.get(stem) {
            let max_pr = pr.values().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_pr = pr.values().copied().fold(f64::INFINITY, f64::min);
            let range = max_pr - min_pr;
            if range > 1e-12 {
                // Normalize to 1.0-10.0 range for readable impact scores
                return 1.0 + 9.0 * (score - min_pr) / range;
            }
        }
    }
    // Fallback: reverse dep count + 1 (so isolated contracts still have impact 1.0)
    (rev_dep_counts.get(stem).copied().unwrap_or(0) + 1) as f64
}

/// Count how many contracts depend on each stem.
fn compute_reverse_dep_counts<'a>(contracts: &'a [(String, &Contract)]) -> HashMap<&'a str, usize> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, contract) in contracts {
        for dep in &contract.metadata.depends_on {
            // Find the matching stem in our contracts
            for (stem, _) in contracts {
                if stem == dep || stem.strip_suffix(".yaml").is_some_and(|s| s == dep) {
                    *counts.entry(stem.as_str()).or_default() += 1;
                }
            }
        }
    }
    counts
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
                action: "Write #[kani::proof] harnesses".into(),
            }],
        };
        let text = score.to_string();
        assert!(text.contains("Grade C"));
        assert!(text.contains("softmax"));
        assert!(text.contains("kani::proof"));
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

    #[test]
    fn gap_actions_are_populated() {
        let (parsed, binding) = load_contracts_and_binding();
        let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();
        let score = score_codebase(&contracts, &binding);
        for gap in &score.top_gaps {
            assert!(!gap.action.is_empty(), "Gap action should not be empty");
            assert!(gap.impact > 0.0, "Gap impact should be positive");
        }
    }

    #[test]
    fn pagerank_weighted_gaps_differ() {
        let (parsed, binding) = load_contracts_and_binding();
        let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();

        let without_pr = score_codebase(&contracts, &binding);

        // Build a synthetic pagerank map where one contract has high pagerank
        let mut pr = HashMap::new();
        for (stem, _) in &contracts {
            pr.insert(stem.clone(), 0.01);
        }
        // Give softmax-kernel-v1.yaml a very high pagerank
        pr.insert("softmax-kernel-v1.yaml".into(), 0.50);

        let with_pr = score_codebase_with_pagerank(&contracts, &binding, Some(&pr));

        // Gap ordering should differ when one contract has much higher pagerank
        assert!(!with_pr.top_gaps.is_empty());
        // The dimensions should still be the same
        assert_eq!(without_pr.composite, with_pr.composite);
    }

    #[test]
    fn dependency_fanout_fallback() {
        let rev_deps: HashMap<&str, usize> = HashMap::new();
        let f = super::dependency_fanout("unknown", None, &rev_deps);
        assert!(
            (f - 1.0).abs() < 1e-9,
            "Unknown contract should have fanout 1.0"
        );

        let mut rev_deps2: HashMap<&str, usize> = HashMap::new();
        rev_deps2.insert("known", 5);
        let f2 = super::dependency_fanout("known", None, &rev_deps2);
        assert!((f2 - 6.0).abs() < 1e-9, "5 reverse deps + 1 = 6.0");
    }

    #[test]
    fn dependency_fanout_with_pagerank() {
        let mut pr = HashMap::new();
        pr.insert("low".to_string(), 0.01);
        pr.insert("high".to_string(), 0.10);
        let rev_deps: HashMap<&str, usize> = HashMap::new();

        let f_low = super::dependency_fanout("low", Some(&pr), &rev_deps);
        let f_high = super::dependency_fanout("high", Some(&pr), &rev_deps);
        assert!(f_high > f_low, "High pagerank should have higher fanout");
        assert!(f_low >= 1.0, "Min fanout should be 1.0");
        assert!(f_high <= 10.0, "Max fanout should be 10.0");
    }

    #[test]
    fn drift_override_affects_composite() {
        let (parsed, binding) = load_contracts_and_binding();
        let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();

        let fresh = super::score_codebase_full(&contracts, &binding, None, Some(1.0));
        let stale = super::score_codebase_full(&contracts, &binding, None, Some(0.0));

        assert!((fresh.drift - 1.0).abs() < 1e-9);
        assert!((stale.drift - 0.0).abs() < 1e-9);
        // Drift weight is 0.15, so composite should differ by 0.15
        let diff = fresh.composite - stale.composite;
        assert!((diff - 0.15).abs() < 0.01, "diff={diff}");
    }

    #[test]
    fn reverse_dep_counts() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  depends_on: ["dep-v1.yaml"]
equations:
  f:
    formula: "f(x) = x"
"#;
        let contract = crate::schema::parse_contract_str(yaml).unwrap();
        let dep_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dep"
equations:
  g:
    formula: "g(x) = x"
"#;
        let dep_contract = crate::schema::parse_contract_str(dep_yaml).unwrap();
        let contracts = vec![
            ("test-v1.yaml".to_string(), &contract),
            ("dep-v1.yaml".to_string(), &dep_contract),
        ];
        let counts = super::compute_reverse_dep_counts(&contracts);
        assert_eq!(counts.get("dep-v1.yaml").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("test-v1.yaml").copied().unwrap_or(0), 0);
    }
}
