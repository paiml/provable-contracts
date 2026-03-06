//! Contract and codebase scoring.
//!
//! Provides quantitative quality assessment for individual contracts
//! and codebases that consume them. Five dimensions per contract,
//! five dimensions per codebase, grades A-F.
//!
//! Spec: `docs/specifications/sub/scoring.md`

mod codebase;
pub mod drift;
mod types;

pub use codebase::{score_codebase, score_codebase_full, score_codebase_with_pagerank};
pub use types::{
    CodebaseScore, ContractScore, Grade, ScoringGap, ScoringWeights,
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
    score_contract_weighted(contract, binding, stem, &ScoringWeights::default())
}

/// Score a contract with custom weights for each dimension.
pub fn score_contract_weighted(
    contract: &Contract,
    binding: Option<&BindingRegistry>,
    stem: &str,
    weights: &ScoringWeights,
) -> ContractScore {
    let w = weights.normalized();
    let spec_depth = compute_spec_depth(contract);
    let falsification = compute_falsification_coverage(contract);
    let kani = compute_kani_coverage(contract);
    let lean = compute_lean_coverage(contract);
    let binding_cov = compute_binding_coverage(contract, binding, stem);

    let composite = spec_depth * w.spec_depth
        + falsification * w.falsification
        + kani * w.kani
        + lean * w.lean
        + binding_cov * w.binding;

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
    include!("scoring_tests.rs");
}
