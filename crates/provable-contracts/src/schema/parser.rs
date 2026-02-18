use std::path::Path;

use crate::error::ContractError;
use crate::schema::types::Contract;

/// Parse a YAML contract file into a [`Contract`] struct.
///
/// This is the entry point for Phase 2 validation. The parser
/// deserializes the YAML and performs structural checks.
///
/// # Errors
///
/// Returns [`ContractError::Io`] if the file cannot be read,
/// or [`ContractError::Yaml`] if the YAML is malformed.
pub fn parse_contract(path: &Path) -> Result<Contract, ContractError> {
    let content = std::fs::read_to_string(path)?;
    parse_contract_str(&content)
}

/// Parse a YAML contract from a string.
pub fn parse_contract_str(yaml: &str) -> Result<Contract, ContractError> {
    let contract: Contract = serde_yaml::from_str(yaml)?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CONTRACT: &str = r#"
metadata:
  version: "1.0.0"
  description: "Test contract"
  references:
    - "Test paper (2024)"
equations:
  test_eq:
    formula: "f(x) = x + 1"
proof_obligations: []
falsification_tests: []
"#;

    #[test]
    fn parse_minimal_contract() {
        let contract = parse_contract_str(MINIMAL_CONTRACT).unwrap();
        assert_eq!(contract.metadata.version, "1.0.0");
        assert_eq!(contract.metadata.description, "Test contract");
        assert_eq!(contract.equations.len(), 1);
        assert!(contract.equations.contains_key("test_eq"));
    }

    #[test]
    fn parse_contract_with_all_fields() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  created: "2026-02-18"
  author: "Test Author"
  description: "Full contract"
  references:
    - "Paper A (2024)"
    - "Paper B (2025)"
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))"
    domain: "x ∈ ℝ^n, n ≥ 1"
    codomain: "σ(x) ∈ (0,1)^n"
    invariants:
      - "sum(output) = 1.0"
      - "output_i > 0"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|sum(softmax(x)) - 1.0| < ε"
    tolerance: 1.0e-6
    applies_to: all
  - type: equivalence
    property: "SIMD matches scalar"
    tolerance: 8.0
    applies_to: simd
kernel_structure:
  phases:
    - name: find_max
      description: "Find max element"
      invariant: "max >= all elements"
    - name: exp_subtract
      description: "Compute exp(x_i - max)"
      invariant: "all values in (0, 1]"
simd_dispatch:
  softmax:
    scalar: "softmax_scalar"
    avx2: "softmax_avx2"
enforcement:
  normalization:
    description: "Output sums to 1.0"
    check: "contract_tests::FALSIFY-SM-001"
    severity: "ERROR"
falsification_tests:
  - id: FALSIFY-SM-001
    rule: "Normalization"
    prediction: "sum(output) ≈ 1.0"
    test: "proptest with random vectors"
    if_fails: "Missing max-subtraction trick"
kani_harnesses:
  - id: KANI-SM-001
    obligation: SM-INV-001
    property: "Softmax sums to 1.0"
    bound: 16
    strategy: stub_float
    solver: cadical
    harness: verify_softmax_normalization
qa_gate:
  id: F-SM-001
  name: "Softmax Contract"
  checks:
    - "normalization"
  pass_criteria: "All falsification tests pass"
  falsification: "Introduce off-by-one in max reduction"
"#;

        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.metadata.version, "1.0.0");
        assert_eq!(contract.metadata.references.len(), 2);
        assert_eq!(contract.equations.len(), 1);
        assert_eq!(contract.proof_obligations.len(), 2);
        assert!(contract.kernel_structure.is_some());
        let ks = contract.kernel_structure.unwrap();
        assert_eq!(ks.phases.len(), 2);
        assert_eq!(contract.simd_dispatch.len(), 1);
        assert_eq!(contract.enforcement.len(), 1);
        assert_eq!(contract.falsification_tests.len(), 1);
        assert_eq!(contract.falsification_tests[0].id, "FALSIFY-SM-001");
        assert_eq!(contract.kani_harnesses.len(), 1);
        assert_eq!(contract.kani_harnesses[0].bound, Some(16));
        assert!(contract.qa_gate.is_some());
    }

    #[test]
    fn parse_invalid_yaml_returns_error() {
        let result = parse_contract_str("not: [valid: yaml: {{");
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_metadata_returns_error() {
        let yaml = r#"
equations:
  test:
    formula: "f(x) = x"
"#;
        let result = parse_contract_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn parse_obligation_types() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "type test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
    if_fails: ""
  - type: equivalence
    property: "test"
  - type: bound
    property: "test"
  - type: monotonicity
    property: "test"
  - type: idempotency
    property: "test"
  - type: linearity
    property: "test"
  - type: symmetry
    property: "test"
  - type: associativity
    property: "test"
  - type: conservation
    property: "test"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.proof_obligations.len(), 9);
    }

    #[test]
    fn parse_kani_strategies() {
        use crate::schema::types::KaniStrategy;

        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "kani test"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: K1
    obligation: OBL-1
    strategy: exhaustive
  - id: K2
    obligation: OBL-2
    strategy: stub_float
  - id: K3
    obligation: OBL-3
    strategy: compositional
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.kani_harnesses.len(), 3);
        assert_eq!(
            contract.kani_harnesses[0].strategy,
            Some(KaniStrategy::Exhaustive)
        );
        assert_eq!(
            contract.kani_harnesses[1].strategy,
            Some(KaniStrategy::StubFloat)
        );
        assert_eq!(
            contract.kani_harnesses[2].strategy,
            Some(KaniStrategy::Compositional)
        );
    }
}
