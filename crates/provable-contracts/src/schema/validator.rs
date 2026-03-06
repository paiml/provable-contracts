use std::collections::HashSet;

use crate::error::{Severity, Violation};
use crate::schema::types::Contract;

/// Validate a parsed contract for completeness and consistency.
///
/// Returns a list of violations. If any violation has
/// [`Severity::Error`], the contract is considered invalid.
pub fn validate_contract(contract: &Contract) -> Vec<Violation> {
    let mut violations = Vec::new();

    validate_metadata(contract, &mut violations);
    validate_equations(contract, &mut violations);
    validate_provability_invariant(contract, &mut violations);
    validate_proof_obligations(contract, &mut violations);
    validate_falsification_tests(contract, &mut violations);
    validate_kani_harnesses(contract, &mut violations);
    validate_qa_gate(contract, &mut violations);

    violations
}

/// Enforce the provability invariant: kernel contracts (non-registry) MUST have
/// proof_obligations, falsification_tests, and kani_harnesses.
fn validate_provability_invariant(contract: &Contract, violations: &mut Vec<Violation>) {
    for v in contract.provability_violations() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "PROVABILITY-001".to_string(),
            message: v,
            location: None,
        });
    }
}

fn validate_metadata(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.metadata.references.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-001".to_string(),
            message: "metadata.references must not be empty — \
                      every contract must cite its source paper(s)"
                .to_string(),
            location: Some("metadata.references".to_string()),
        });
    }

    if contract.metadata.version.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-002".to_string(),
            message: "metadata.version must not be empty".to_string(),
            location: Some("metadata.version".to_string()),
        });
    }
}

fn validate_equations(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.equations.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-003".to_string(),
            message: "equations must contain at least one equation".to_string(),
            location: Some("equations".to_string()),
        });
    }

    for (name, eq) in &contract.equations {
        if eq.formula.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-004".to_string(),
                message: format!("equations.{name}.formula must not be empty"),
                location: Some(format!("equations.{name}.formula")),
            });
        }
    }
}

fn validate_proof_obligations(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut seen_ids = HashSet::new();
    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        if ob.property.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-005".to_string(),
                message: format!("proof_obligations[{i}].property must not be empty"),
                location: Some(format!("proof_obligations[{i}].property")),
            });
        }
        if let Some(ref formal) = ob.formal {
            if !seen_ids.insert(formal.clone()) {
                violations.push(Violation {
                    severity: Severity::Warning,
                    rule: "SCHEMA-006".to_string(),
                    message: format!("Duplicate formal predicate: {formal}"),
                    location: Some(format!("proof_obligations[{i}].formal")),
                });
            }
        }
    }
}

fn validate_falsification_tests(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for test in &contract.falsification_tests {
        if !ids.insert(&test.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-007".to_string(),
                message: format!("Duplicate falsification test ID: {}", test.id),
                location: Some(format!("falsification_tests.{}", test.id)),
            });
        }
        if test.prediction.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-008".to_string(),
                message: format!(
                    "falsification_tests.{}.prediction must not be empty — \
                     every test must make a falsifiable prediction",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.prediction", test.id)),
            });
        }
        if test.if_fails.is_empty() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-009".to_string(),
                message: format!(
                    "falsification_tests.{}.if_fails is empty — \
                     should describe root cause diagnosis",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.if_fails", test.id)),
            });
        }
    }
}

fn validate_kani_harnesses(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for harness in &contract.kani_harnesses {
        if !ids.insert(&harness.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-010".to_string(),
                message: format!("Duplicate Kani harness ID: {}", harness.id),
                location: Some(format!("kani_harnesses.{}", harness.id)),
            });
        }
        if harness.obligation.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-011".to_string(),
                message: format!(
                    "kani_harnesses.{}.obligation must not be empty — \
                     every harness must reference a proof obligation",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.obligation", harness.id)),
            });
        }
        if harness.bound.is_none() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-012".to_string(),
                message: format!(
                    "kani_harnesses.{}.bound not specified — \
                     Kani requires an unwind bound",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.bound", harness.id)),
            });
        }
    }
}

fn validate_qa_gate(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.qa_gate.is_none() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "SCHEMA-013".to_string(),
            message: "No qa_gate defined — contract should define a \
                      certeza quality gate"
                .to_string(),
            location: Some("qa_gate".to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
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
}
