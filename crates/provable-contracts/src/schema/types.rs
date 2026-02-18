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
/// idempotency, linearity, symmetry, associativity, conservation.
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
}

impl std::fmt::Display for KaniStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Exhaustive => "exhaustive",
            Self::StubFloat => "stub_float",
            Self::Compositional => "compositional",
        };
        write!(f, "{s}")
    }
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
