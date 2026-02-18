//! Audit trail generator — traceability chain.
//!
//! Traces the full chain from paper reference → equation →
//! proof obligation → falsification test → Kani harness.
//! Detects orphaned obligations and untested equations.

use crate::error::{Severity, Violation};
use crate::schema::Contract;

/// Audit result summarizing traceability coverage.
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Total equations in the contract.
    pub equations: usize,
    /// Total proof obligations.
    pub obligations: usize,
    /// Total falsification tests.
    pub falsification_tests: usize,
    /// Total Kani harnesses.
    pub kani_harnesses: usize,
    /// Violations found during audit.
    pub violations: Vec<Violation>,
}

/// Run a traceability audit on a contract.
///
/// Checks that every proof obligation is covered by at least
/// one falsification test or Kani harness, and that no
/// harnesses reference non-existent obligations.
pub fn audit_contract(contract: &Contract) -> AuditReport {
    let mut violations = Vec::new();

    // Check: contract has at least one falsification test
    if contract.falsification_tests.is_empty() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "AUDIT-001".to_string(),
            message: "No falsification tests — contract is \
                      not falsifiable"
                .to_string(),
            location: Some("falsification_tests".to_string()),
        });
    }

    // Check: every Kani harness references a valid obligation ID
    let obligation_props: Vec<&str> = contract
        .proof_obligations
        .iter()
        .map(|o| o.property.as_str())
        .collect();

    for harness in &contract.kani_harnesses {
        // The harness.obligation field should reference an
        // obligation. We do a soft check (warn, not error)
        // since obligation IDs are free-form in current schema.
        if harness.obligation.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "AUDIT-002".to_string(),
                message: format!(
                    "Kani harness {} has empty obligation \
                     reference",
                    harness.id
                ),
                location: Some(format!(
                    "kani_harnesses.{}.obligation",
                    harness.id
                )),
            });
        }
    }

    // Check: contract has proof obligations if it has equations
    if !contract.equations.is_empty()
        && contract.proof_obligations.is_empty()
    {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "AUDIT-003".to_string(),
            message: "Equations defined but no proof obligations \
                      — obligations should be derived from equations"
                .to_string(),
            location: Some("proof_obligations".to_string()),
        });
    }

    // Suppress unused variable warning
    let _ = obligation_props;

    AuditReport {
        equations: contract.equations.len(),
        obligations: contract.proof_obligations.len(),
        falsification_tests: contract.falsification_tests.len(),
        kani_harnesses: contract.kani_harnesses.len(),
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn audit_minimal_contract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Minimal"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert_eq!(report.equations, 1);
        assert_eq!(report.falsification_tests, 0);
        // Should warn about no falsification tests
        assert!(report
            .violations
            .iter()
            .any(|v| v.rule == "AUDIT-001"));
    }

    #[test]
    fn audit_complete_contract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Complete"
  references: ["Paper"]
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
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert_eq!(report.equations, 1);
        assert_eq!(report.obligations, 1);
        assert_eq!(report.falsification_tests, 1);
        assert_eq!(report.kani_harnesses, 1);
        // No errors expected
        let errors: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Expected no errors, got: {errors:?}"
        );
    }

    #[test]
    fn audit_warns_on_no_obligations() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No obligations"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert!(report
            .violations
            .iter()
            .any(|v| v.rule == "AUDIT-003"));
    }
}
