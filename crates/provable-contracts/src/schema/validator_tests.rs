    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn valid_contract_has_no_errors() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Valid"
  references:
    - "Paper (2024)"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output is finite"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    if_fails: "overflow in computation"
kani_harnesses:
  - id: KANI-001
    obligation: "output is finite"
    bound: 8
    strategy: stub_float
    solver: cadical
    harness: verify_finiteness
qa_gate:
  id: F-001
  name: "Test Gate"
  checks:
    - "finiteness"
  pass_criteria: "All tests pass"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn missing_references_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No refs"
  references: []
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-001"));
    }

    #[test]
    fn empty_formula_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty formula"
  references:
    - "Paper"
equations:
  bad:
    formula: ""
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-004"));
    }

    #[test]
    fn duplicate_falsification_id_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dup IDs"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
  - id: FALSIFY-001
    rule: "test2"
    prediction: "works2"
    if_fails: "broken2"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-007"));
    }

    #[test]
    fn kani_harness_without_bound_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No bound"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-012"));
    }

    #[test]
    fn no_equations_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No equations"
  references:
    - "Paper"
equations: {}
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-003"));
    }

    #[test]
    fn empty_version_is_error() {
        let yaml = r#"
metadata:
  version: ""
  description: "Empty version"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-002"));
    }

    #[test]
    fn empty_property_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty prop"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: ""
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-005"));
    }

    #[test]
    fn duplicate_formal_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dup formal"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "prop1"
    formal: "same_formal"
  - type: bound
    property: "prop2"
    formal: "same_formal"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-006"));
    }

    #[test]
    fn empty_prediction_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty pred"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: ""
    if_fails: "broken"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-008"));
    }

    #[test]
    fn empty_if_fails_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty if_fails"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: ""
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-009"));
    }

    #[test]
    fn duplicate_kani_id_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dup kani"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 8
  - id: KANI-001
    obligation: OBL-002
    bound: 16
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-010"));
    }

    #[test]
    fn empty_kani_obligation_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty kani obl"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: ""
    bound: 8
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-011"));
    }

    #[test]
    fn kernel_contract_without_kani_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Missing kani"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output is finite"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "finite"
    if_fails: "overflow"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(
            violations.iter().any(|v| v.rule == "PROVABILITY-001"),
            "Kernel contract without kani_harnesses must fail PROVABILITY-001"
        );
    }

    #[test]
    fn registry_contract_without_kani_is_ok() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Data registry"
  registry: true
  references:
    - "Internal"
equations:
  lookup:
    formula: "f(key) = table[key]"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(
            !violations.iter().any(|v| v.rule == "PROVABILITY-001"),
            "Registry contract should be exempt from provability invariant"
        );
    }

    #[test]
    fn kernel_with_fewer_tests_than_obligations_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Imbalanced"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "prop1"
  - type: bound
    property: "prop2"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
kani_harnesses:
  - id: KANI-001
    obligation: "prop1"
    bound: 8
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(
            violations
                .iter()
                .any(|v| v.rule == "PROVABILITY-001"
                    && v.message.contains("falsification_tests")),
            "Fewer falsification tests than obligations must fail PROVABILITY-001"
        );
    }

    #[test]
    fn no_qa_gate_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No qa gate"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-013"));
    }

    #[test]
    fn requires_only_valid_on_postcondition() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Bad requires"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
    requires: "PRE-001"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-014"));
    }

    #[test]
    fn applies_to_phase_only_valid_on_loop_types() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Bad applies_to_phase"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: bound
    property: "test"
    applies_to_phase: "find_max"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-015"));
    }

    #[test]
    fn parent_contract_only_valid_on_subcontract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Bad parent_contract"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: equivalence
    property: "test"
    parent_contract: "parent-v1"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-016"));
    }

    #[test]
    fn parent_contract_must_be_in_depends_on() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Missing depends_on"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: subcontract
    property: "refines parent"
    parent_contract: "not-in-depends-on"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-017"));
    }

    #[test]
    fn valid_dbc_fields_pass_validation() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Valid DbC"
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
falsification_tests:
  - id: F-001
    rule: "test"
    prediction: "test"
    if_fails: "test"
  - id: F-002
    rule: "test"
    prediction: "test"
    if_fails: "test"
  - id: F-003
    rule: "test"
    prediction: "test"
    if_fails: "test"
kani_harnesses:
  - id: K-001
    obligation: "O-001"
    bound: 8
qa_gate:
  id: QA-001
  name: "Test"
  pass_criteria: "all pass"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        // Should have no SCHEMA-014/015/016/017 errors
        let dbc_errors: Vec<_> = violations
            .iter()
            .filter(|v| {
                v.rule == "SCHEMA-014"
                    || v.rule == "SCHEMA-015"
                    || v.rule == "SCHEMA-016"
                    || v.rule == "SCHEMA-017"
            })
            .collect();
        assert!(dbc_errors.is_empty(), "unexpected DbC errors: {dbc_errors:?}");
    }
