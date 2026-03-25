use super::*;
use crate::schema::parse_contract_str;

#[test]
fn explain_minimal_contract() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test kernel"
  references: ["Paper A"]
equations:
  f:
    formula: "f(x) = x"
    domain: "x ∈ ℝ^n"
proof_obligations:
  - type: invariant
    property: "output finite"
    formal: "|f(x)| < ∞"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    test: "proptest"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "test-kernel-v1", None);
    assert!(output.contains("test-kernel-v1 (v1.0.0)"));
    assert!(output.contains("Test kernel"));
    assert!(output.contains("Paper A"));
    assert!(output.contains("f(x) = x"));
    assert!(output.contains("[invariant] output finite"));
    assert!(output.contains("FALSIFY-001"));
    assert!(output.contains("KANI-001"));
    assert!(output.contains("Verification ladder"));
}

#[test]
fn explain_with_preconditions() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Pre/post test"
equations:
  eq:
    formula: "f(x) = x + 1"
    preconditions:
      - "x > 0"
    postconditions:
      - "ret > 1"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "prepost-v1", None);
    assert!(output.contains("Preconditions (caller must guarantee)"));
    assert!(output.contains("x > 0"));
    assert!(output.contains("Postconditions (kernel guarantees)"));
    assert!(output.contains("ret > 1"));
}

#[test]
fn explain_with_lean_proof() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Lean test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
    lean:
      theorem: test_theorem
      module: Test.Module
      status: proved
      depends_on: [dep1]
      notes: "Proved over reals"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "lean-v1", None);
    assert!(output.contains("Test.Module.test_theorem (proved)"));
    assert!(output.contains("Depends: dep1"));
    assert!(output.contains("Note: Proved over reals"));
}

#[test]
fn explain_markdown_has_headers_and_latex() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Markdown test"
  references: ["Paper A"]
equations:
  f:
    formula: "σ(x)_i = exp(x_i)"
    domain: "x ∈ ℝ^n"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "test-v1", None);
    assert!(output.contains("# test-v1"));
    assert!(output.contains("## Equations"));
    assert!(output.contains("$$"));
    assert!(output.contains("\\sigma"));
    assert!(output.contains("## Proof Obligations"));
    assert!(output.contains("| # | Type |"));
}

#[test]
fn explain_json_is_valid() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "JSON test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: precondition
    property: "input valid"
  - type: postcondition
    property: "output bounded"
    requires: "PRE-001"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_json(&contract, "json-v1", None);
    let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(parsed["stem"], "json-v1");
    assert_eq!(parsed["version"], "1.0.0");
    assert_eq!(parsed["obligations"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["obligations"][0]["type"], "precondition");
    assert_eq!(parsed["obligations"][1]["requires"], "PRE-001");
}

#[test]
fn explain_renders_dbc_fields() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "DbC fields test"
  references: ["Meyer (1997)"]
  depends_on: ["parent-v1"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: postcondition
    property: "output bounded"
    requires: "PRE-001"
  - type: loop_invariant
    property: "max tracks"
    applies_to_phase: "find_max"
  - type: subcontract
    property: "refines parent"
    parent_contract: "parent-v1"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "dbc-v1", None);
    assert!(output.contains("Requires: PRE-001"));
    assert!(output.contains("Phase: find_max"));
    assert!(output.contains("Refines: parent-v1"));
    assert!(output.contains("behavioral subtyping"));
}

#[test]
fn obligation_pattern_coverage() {
    // Verify all obligation types have a pattern
    let types = [
        ObligationType::Invariant,
        ObligationType::Equivalence,
        ObligationType::Bound,
        ObligationType::Monotonicity,
        ObligationType::Idempotency,
        ObligationType::Linearity,
        ObligationType::Symmetry,
        ObligationType::Associativity,
        ObligationType::Conservation,
        ObligationType::Ordering,
        ObligationType::Completeness,
        ObligationType::Soundness,
        ObligationType::Involution,
        ObligationType::Determinism,
        ObligationType::Roundtrip,
        ObligationType::StateMachine,
        ObligationType::Classification,
        ObligationType::Independence,
        ObligationType::Termination,
        ObligationType::Precondition,
        ObligationType::Postcondition,
        ObligationType::Frame,
        ObligationType::LoopInvariant,
        ObligationType::LoopVariant,
        ObligationType::OldState,
        ObligationType::Subcontract,
    ];
    for t in types {
        let pattern = obligation_pattern(t);
        assert!(!pattern.is_empty(), "empty pattern for {t}");
    }
}
