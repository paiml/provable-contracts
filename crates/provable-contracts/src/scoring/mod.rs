//! Contract and codebase scoring.
//!
//! Provides quantitative quality assessment for individual contracts
//! and codebases that consume them. Five dimensions per contract,
//! five dimensions per codebase, grades A-F.
//!
//! Spec: `docs/specifications/sub/scoring.md`

mod codebase;
mod types;

pub use codebase::score_codebase;
pub use types::{
    CodebaseScore, ContractScore, Grade, ScoringGap,
};

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::{Contract, KaniHarness, KaniStrategy, LeanStatus};

/// Score a single contract against its completeness and verification depth.
///
/// Five dimensions (weights from spec):
/// - D1: Specification depth (20%)
/// - D2: Falsification coverage (25%)
/// - D3: Kani proof coverage (25%)
/// - D4: Lean proof coverage (10%)
/// - D5: Binding coverage (20%)
pub fn score_contract(
    contract: &Contract,
    binding: Option<&BindingRegistry>,
    stem: &str,
) -> ContractScore {
    let spec_depth = compute_spec_depth(contract);
    let falsification = compute_falsification_coverage(contract);
    let kani = compute_kani_coverage(contract);
    let lean = compute_lean_coverage(contract);
    let binding_cov = compute_binding_coverage(contract, binding, stem);

    let composite =
        spec_depth * 0.20 + falsification * 0.25 + kani * 0.25 + lean * 0.10 + binding_cov * 0.20;

    ContractScore {
        stem: stem.to_string(),
        spec_depth,
        falsification_coverage: falsification,
        kani_coverage: kani,
        lean_coverage: lean,
        binding_coverage: binding_cov,
        composite,
        grade: Grade::from_score(composite),
    }
}

fn compute_spec_depth(contract: &Contract) -> f64 {
    let mut score = 0.0;

    // Has equations (0.30)
    if !contract.equations.is_empty() {
        score += 0.30;
    }

    // Has domains on equations (0.15)
    let has_domains = contract
        .equations
        .values()
        .any(|eq| eq.domain.is_some());
    if has_domains {
        score += 0.15;
    }

    // Has invariants on equations (0.15)
    let has_invariants = contract
        .equations
        .values()
        .any(|eq| !eq.invariants.is_empty());
    if has_invariants {
        score += 0.15;
    }

    // Has kernel structure (0.15)
    if contract.kernel_structure.is_some() {
        score += 0.15;
    }

    // Has tolerances on obligations (0.10)
    let has_tolerances = contract
        .proof_obligations
        .iter()
        .any(|ob| ob.tolerance.is_some());
    if has_tolerances {
        score += 0.10;
    }

    // Has references (0.10)
    if !contract.metadata.references.is_empty() {
        score += 0.10;
    }

    // Has depends_on (0.05)
    if !contract.metadata.depends_on.is_empty() {
        score += 0.05;
    }

    score
}

#[allow(clippy::cast_precision_loss)]
fn compute_falsification_coverage(contract: &Contract) -> f64 {
    let total = contract.proof_obligations.len();
    if total == 0 {
        return if contract.falsification_tests.is_empty() {
            0.0
        } else {
            1.0
        };
    }
    let covered = contract.falsification_tests.len().min(total);
    covered as f64 / total as f64
}

#[allow(clippy::cast_precision_loss)]
fn compute_kani_coverage(contract: &Contract) -> f64 {
    let total = contract.proof_obligations.len();
    if total == 0 {
        return if contract.kani_harnesses.is_empty() {
            0.0
        } else {
            1.0
        };
    }

    let strategy_weight = |h: &KaniHarness| -> f64 {
        match h.strategy.as_ref() {
            Some(KaniStrategy::Exhaustive) => 1.0,
            Some(KaniStrategy::BoundedInt) => 0.9,
            Some(KaniStrategy::StubFloat) => 0.8,
            Some(KaniStrategy::Compositional) => 0.7,
            None => 0.5,
        }
    };

    let weighted_sum: f64 = contract.kani_harnesses.iter().map(strategy_weight).sum();
    (weighted_sum / total as f64).min(1.0)
}

#[allow(clippy::cast_precision_loss)]
fn compute_lean_coverage(contract: &Contract) -> f64 {
    let applicable: Vec<_> = contract
        .proof_obligations
        .iter()
        .filter(|ob| {
            ob.lean
                .as_ref()
                .is_some_and(|l| l.status != LeanStatus::NotApplicable)
        })
        .collect();

    if applicable.is_empty() {
        return 0.0;
    }

    let proved = applicable
        .iter()
        .filter(|ob| {
            ob.lean
                .as_ref()
                .is_some_and(|l| l.status == LeanStatus::Proved)
        })
        .count();

    proved as f64 / applicable.len() as f64
}

#[allow(clippy::cast_precision_loss)]
fn compute_binding_coverage(
    _contract: &Contract,
    binding: Option<&BindingRegistry>,
    stem: &str,
) -> f64 {
    let Some(binding) = binding else {
        return 0.0;
    };

    let relevant: Vec<_> = binding
        .bindings
        .iter()
        .filter(|b| b.contract == stem)
        .collect();

    if relevant.is_empty() {
        return 0.0;
    }

    let implemented: f64 = relevant
        .iter()
        .map(|b| match b.status {
            ImplStatus::Implemented => 1.0,
            ImplStatus::Partial => 0.5,
            ImplStatus::NotImplemented => 0.0,
        })
        .sum();

    implemented / relevant.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn minimal_kernel_yaml() -> &'static str {
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
  depends_on: ["other-v1"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
    codomain: "R"
    invariants: ["output finite"]
proof_obligations:
  - type: invariant
    property: "finite"
    tolerance: 1e-6
falsification_tests:
  - id: FALSIFY-001
    rule: "finite"
    prediction: "finite"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: stub_float
    solver: cadical
"#
    }

    #[test]
    fn score_complete_contract() {
        let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
        let score = score_contract(&contract, None, "test-v1");
        assert!(score.spec_depth >= 0.70, "spec_depth={}", score.spec_depth);
        assert_eq!(score.falsification_coverage, 1.0);
        assert!(score.kani_coverage > 0.0);
        assert_eq!(score.lean_coverage, 0.0);
        assert_eq!(score.binding_coverage, 0.0);
        assert!(score.composite > 0.0);
    }

    #[test]
    fn grade_thresholds() {
        assert_eq!(Grade::from_score(0.95), Grade::A);
        assert_eq!(Grade::from_score(0.80), Grade::B);
        assert_eq!(Grade::from_score(0.65), Grade::C);
        assert_eq!(Grade::from_score(0.45), Grade::D);
        assert_eq!(Grade::from_score(0.30), Grade::F);
    }

    #[test]
    fn empty_contract_scores_low() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty"
  registry: true
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let score = score_contract(&contract, None, "empty-v1");
        assert!(score.composite < 0.40, "composite={}", score.composite);
        assert_eq!(score.grade, Grade::F);
    }

    #[test]
    fn score_real_softmax_contract() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/softmax-kernel-v1.yaml");
        let contract = crate::schema::parse_contract(&path).unwrap();
        let score = score_contract(&contract, None, "softmax-kernel-v1");
        assert!(score.composite > 0.40, "softmax should score well: {}", score.composite);
        assert!(score.spec_depth > 0.50);
        assert!(score.falsification_coverage > 0.0);
        assert!(score.kani_coverage > 0.0);
    }
}
