use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A complete YAML kernel contract.
///
/// This is the root type for the contract schema defined in
/// `docs/specifications/provable-contracts.md` Section 5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub metadata: Metadata,
    pub equations: BTreeMap<String, Equation>,
    #[serde(default)]
    pub proof_obligations: Vec<ProofObligation>,
    #[serde(default)]
    pub kernel_structure: Option<KernelStructure>,
    #[serde(default)]
    pub simd_dispatch: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default)]
    pub enforcement: BTreeMap<String, EnforcementRule>,
    #[serde(default)]
    pub falsification_tests: Vec<FalsificationTest>,
    #[serde(default)]
    pub kani_harnesses: Vec<KaniHarness>,
    #[serde(default)]
    pub qa_gate: Option<QaGate>,
    /// Phase 7: Lean 4 verification summary across all obligations.
    #[serde(default)]
    pub verification_summary: Option<VerificationSummary>,
}

/// Contract metadata block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub version: String,
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    pub description: String,
    #[serde(default)]
    pub references: Vec<String>,
    /// Contract dependencies — other contracts this one composes.
    /// Values are contract stems (e.g. "silu-kernel-v1").
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// A mathematical equation extracted from a paper (Phase 1 output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Equation {
    pub formula: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub codomain: Option<String>,
    #[serde(default)]
    pub invariants: Vec<String>,
}

/// A proof obligation derived from an equation.
///
/// Maps to one of the types in the Proof Obligation Taxonomy
/// (spec Section 12): invariant, equivalence, bound, monotonicity,
/// idempotency, linearity, symmetry, associativity, conservation,
/// ordering, completeness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofObligation {
    #[serde(rename = "type")]
    pub obligation_type: ObligationType,
    pub property: String,
    #[serde(default)]
    pub formal: Option<String>,
    #[serde(default)]
    pub tolerance: Option<f64>,
    #[serde(default)]
    pub applies_to: Option<AppliesTo>,
    /// Phase 7: Lean 4 theorem proving metadata.
    #[serde(default)]
    pub lean: Option<LeanProof>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObligationType {
    Invariant,
    Equivalence,
    Bound,
    Monotonicity,
    Idempotency,
    Linearity,
    Symmetry,
    Associativity,
    Conservation,
    Ordering,
    Completeness,
    Soundness,
}

impl std::fmt::Display for ObligationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Invariant => "invariant",
            Self::Equivalence => "equivalence",
            Self::Bound => "bound",
            Self::Monotonicity => "monotonicity",
            Self::Idempotency => "idempotency",
            Self::Linearity => "linearity",
            Self::Symmetry => "symmetry",
            Self::Associativity => "associativity",
            Self::Conservation => "conservation",
            Self::Ordering => "ordering",
            Self::Completeness => "completeness",
            Self::Soundness => "soundness",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppliesTo {
    All,
    Scalar,
    Simd,
    Converter,
}

/// Kernel phase decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStructure {
    pub phases: Vec<KernelPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPhase {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub invariant: Option<String>,
}

/// An enforcement rule from the contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementRule {
    pub description: String,
    #[serde(default)]
    pub check: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub reference: Option<String>,
}

/// A Popperian falsification test.
///
/// Each makes a falsifiable prediction about the implementation.
/// If the prediction is wrong, the test identifies root cause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationTest {
    pub id: String,
    pub rule: String,
    pub prediction: String,
    #[serde(default)]
    pub test: Option<String>,
    pub if_fails: String,
}

/// A Kani bounded model checking harness definition.
///
/// Corresponds to Phase 6 (Verify) of the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaniHarness {
    pub id: String,
    pub obligation: String,
    #[serde(default)]
    pub property: Option<String>,
    #[serde(default)]
    pub bound: Option<u32>,
    #[serde(default)]
    pub strategy: Option<KaniStrategy>,
    #[serde(default)]
    pub solver: Option<String>,
    #[serde(default)]
    pub harness: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KaniStrategy {
    Exhaustive,
    StubFloat,
    Compositional,
    BoundedInt,
}

impl std::fmt::Display for KaniStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Exhaustive => "exhaustive",
            Self::StubFloat => "stub_float",
            Self::Compositional => "compositional",
            Self::BoundedInt => "bounded_int",
        };
        write!(f, "{s}")
    }
}

/// Phase 7: Lean 4 theorem proving metadata for a proof obligation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeanProof {
    /// Lean 4 theorem name (e.g., `Softmax.partition_of_unity`).
    pub theorem: String,
    /// Lean 4 module path (e.g., `ProvableContracts.Softmax`).
    #[serde(default)]
    pub module: Option<String>,
    /// Current status of the Lean proof.
    #[serde(default)]
    pub status: LeanStatus,
    /// Lean-level theorem dependencies.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Mathlib import paths required.
    #[serde(default)]
    pub mathlib_imports: Vec<String>,
    /// Free-form notes (e.g., "Proof over reals; f32 gap addressed separately").
    #[serde(default)]
    pub notes: Option<String>,
}

/// Status of a Lean 4 proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeanStatus {
    /// Proof is complete and type-checks.
    Proved,
    /// Proof uses `sorry` (axiomatized, not yet proved).
    #[default]
    Sorry,
    /// Work in progress.
    Wip,
    /// Obligation is not amenable to Lean proof (e.g., performance).
    NotApplicable,
}

impl std::fmt::Display for LeanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Proved => "proved",
            Self::Sorry => "sorry",
            Self::Wip => "wip",
            Self::NotApplicable => "not-applicable",
        };
        write!(f, "{s}")
    }
}

/// Phase 7: Verification summary across all obligations in a contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub total_obligations: u32,
    #[serde(default)]
    pub l2_property_tested: u32,
    #[serde(default)]
    pub l3_kani_proved: u32,
    #[serde(default)]
    pub l4_lean_proved: u32,
    #[serde(default)]
    pub l4_sorry_count: u32,
    #[serde(default)]
    pub l4_not_applicable: u32,
}

/// QA gate definition for certeza integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaGate {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub checks: Vec<String>,
    #[serde(default)]
    pub pass_criteria: Option<String>,
    #[serde(default)]
    pub falsification: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obligation_type_display() {
        assert_eq!(ObligationType::Invariant.to_string(), "invariant");
        assert_eq!(ObligationType::Equivalence.to_string(), "equivalence");
        assert_eq!(ObligationType::Bound.to_string(), "bound");
        assert_eq!(ObligationType::Monotonicity.to_string(), "monotonicity");
        assert_eq!(ObligationType::Idempotency.to_string(), "idempotency");
        assert_eq!(ObligationType::Linearity.to_string(), "linearity");
        assert_eq!(ObligationType::Symmetry.to_string(), "symmetry");
        assert_eq!(ObligationType::Associativity.to_string(), "associativity");
        assert_eq!(ObligationType::Conservation.to_string(), "conservation");
        assert_eq!(ObligationType::Ordering.to_string(), "ordering");
        assert_eq!(ObligationType::Completeness.to_string(), "completeness");
        assert_eq!(ObligationType::Soundness.to_string(), "soundness");
    }

    #[test]
    fn lean_status_display() {
        assert_eq!(LeanStatus::Proved.to_string(), "proved");
        assert_eq!(LeanStatus::Sorry.to_string(), "sorry");
        assert_eq!(LeanStatus::Wip.to_string(), "wip");
        assert_eq!(LeanStatus::NotApplicable.to_string(), "not-applicable");
    }

    #[test]
    fn lean_status_default_is_sorry() {
        assert_eq!(LeanStatus::default(), LeanStatus::Sorry);
    }

    #[test]
    fn kani_strategy_display() {
        assert_eq!(KaniStrategy::Exhaustive.to_string(), "exhaustive");
        assert_eq!(KaniStrategy::StubFloat.to_string(), "stub_float");
        assert_eq!(KaniStrategy::Compositional.to_string(), "compositional");
        assert_eq!(KaniStrategy::BoundedInt.to_string(), "bounded_int");
    }
}
