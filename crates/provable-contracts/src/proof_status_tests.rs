    use super::*;
    use crate::schema::parse_contract_str;

    /// Build a minimal contract with configurable obligation, test, and harness counts
    fn minimal_contract(n_ob: usize, n_ft: usize, n_kani: usize) -> Contract {
        let mut yaml = String::from(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
"#,
        );
        for i in 0..n_ob {
            yaml.push_str(&format!(
                "  - type: invariant\n    property: \"prop {i}\"\n"
            ));
        }
        yaml.push_str("falsification_tests:\n");
        for i in 0..n_ft {
            yaml.push_str(&format!(
                "  - id: FT-{i:03}\n    rule: \"r\"\n    prediction: \"p\"\n    if_fails: \"f\"\n"
            ));
        }
        yaml.push_str("kani_harnesses:\n");
        for i in 0..n_kani {
            yaml.push_str(&format!(
                "  - id: KH-{i:03}\n    obligation: OBL-{i:03}\n    bound: 16\n"
            ));
        }
        parse_contract_str(&yaml).unwrap()
    }

    /// Build a contract with Lean verification summary for L4/L5 level testing
    fn contract_with_lean(total: u32, lean_proved: u32) -> Contract {
        let mut c = minimal_contract(total as usize, total as usize, total as usize);
        c.verification_summary = Some(crate::schema::VerificationSummary {
            total_obligations: total,
            l2_property_tested: total,
            l3_kani_proved: total,
            l4_lean_proved: lean_proved,
            l4_sorry_count: total - lean_proved,
            l4_not_applicable: 0,
        });
        c
    }

    #[test]
    fn proof_level_display() {
        assert_eq!(ProofLevel::L1.to_string(), "L1");
        assert_eq!(ProofLevel::L2.to_string(), "L2");
        assert_eq!(ProofLevel::L3.to_string(), "L3");
        assert_eq!(ProofLevel::L4.to_string(), "L4");
        assert_eq!(ProofLevel::L5.to_string(), "L5");
    }

    #[test]
    fn proof_level_ordering() {
        assert!(ProofLevel::L1 < ProofLevel::L2);
        assert!(ProofLevel::L2 < ProofLevel::L3);
        assert!(ProofLevel::L3 < ProofLevel::L4);
        assert!(ProofLevel::L4 < ProofLevel::L5);
    }

    #[test]
    fn level_l1_for_equations_only() {
        let c = minimal_contract(0, 0, 0);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
    }

    #[test]
    fn level_l2_for_falsification_covered() {
        let c = minimal_contract(3, 3, 0);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L2);
    }

    #[test]
    fn level_l2_not_enough_tests() {
        let c = minimal_contract(3, 2, 0);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
    }

    #[test]
    fn level_l3_kani_plus_falsification() {
        let c = minimal_contract(3, 3, 2);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L3);
    }

    #[test]
    fn level_l3_kani_without_enough_tests() {
        // Has kani but not enough falsification tests
        let c = minimal_contract(3, 2, 2);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
    }

    #[test]
    fn level_l4_all_lean_proved() {
        let c = contract_with_lean(3, 3);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L4);
    }

    #[test]
    fn level_l4_partial_lean_stays_l3() {
        let c = contract_with_lean(3, 2);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L3);
    }

    #[test]
    fn level_l5_lean_plus_all_bound() {
        let c = contract_with_lean(3, 3);
        assert_eq!(compute_proof_level(&c, Some((1, 1))), ProofLevel::L5);
    }

    #[test]
    fn level_l4_when_bindings_incomplete() {
        let c = contract_with_lean(3, 3);
        assert_eq!(compute_proof_level(&c, Some((0, 1))), ProofLevel::L4);
    }

    #[test]
    fn report_empty_contracts() {
        let report = proof_status_report(&[], None, false);
        assert_eq!(report.totals.contracts, 0);
        assert_eq!(report.totals.obligations, 0);
        assert!(report.contracts.is_empty());
    }

    #[test]
    fn report_single_contract() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("test-v1".to_string(), &c)], None, false);
        assert_eq!(report.contracts.len(), 1);
        assert_eq!(report.contracts[0].stem, "test-v1");
        assert_eq!(report.contracts[0].proof_level, ProofLevel::L3);
        assert_eq!(report.totals.obligations, 3);
        assert_eq!(report.totals.falsification_tests, 3);
        assert_eq!(report.totals.kani_harnesses, 2);
    }

    #[test]
    fn report_with_binding() {
        let c = minimal_contract(3, 3, 2);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test-v1.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
        )
        .unwrap();
        let report = proof_status_report(&[("test-v1".to_string(), &c)], Some(&binding), false);
        assert_eq!(report.contracts[0].bindings_implemented, 1);
        assert_eq!(report.contracts[0].bindings_total, 1);
    }

    #[test]
    fn report_with_kernel_classes() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("softmax-kernel-v1".to_string(), &c)], None, true);
        assert!(!report.kernel_classes.is_empty());
        // Softmax is in classes A, B, C, D, E
        let class_a = report
            .kernel_classes
            .iter()
            .find(|kc| kc.label == "A")
            .unwrap();
        assert!(
            class_a
                .contract_stems
                .contains(&"softmax-kernel-v1".to_string())
        );
    }

    #[test]
    fn format_text_produces_output() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("softmax-kernel-v1".to_string(), &c)], None, true);
        let text = format_text(&report);
        assert!(text.contains("Proof Status"));
        assert!(text.contains("softmax-kernel-v1"));
        assert!(text.contains("Kernel Classes:"));
        assert!(text.contains("Totals:"));
    }

    #[test]
    fn format_text_without_classes() {
        let c = minimal_contract(2, 2, 0);
        let report = proof_status_report(&[("test-v1".to_string(), &c)], None, false);
        let text = format_text(&report);
        assert!(text.contains("test-v1"));
        assert!(!text.contains("Kernel Classes:"));
    }

    #[test]
    fn json_roundtrip() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("test-v1".to_string(), &c)], None, true);
        let json = serde_json::to_string(&report).unwrap();
        let parsed: ProofStatusReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.contracts.len(), 1);
        assert_eq!(parsed.contracts[0].proof_level, ProofLevel::L3);
        assert_eq!(parsed.totals.obligations, 3);
    }

    #[test]
    fn kernel_class_min_level() {
        let c1 = minimal_contract(3, 3, 2); // L3
        let c2 = minimal_contract(3, 3, 0); // L2
        let report = proof_status_report(
            &[
                ("softmax-kernel-v1".to_string(), &c1),
                ("matmul-kernel-v1".to_string(), &c2),
            ],
            None,
            true,
        );
        let class_a = report
            .kernel_classes
            .iter()
            .find(|kc| kc.label == "A")
            .unwrap();
        // min of L3 and L2 is L2
        assert_eq!(class_a.min_proof_level, ProofLevel::L2);
    }

    #[test]
    fn count_bindings_helper() {
        let c = minimal_contract(1, 1, 1);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    status: implemented
  - contract: other.yaml
    equation: g
    status: implemented
"#,
        )
        .unwrap();
        let (implemented, total) = count_bindings("test.yaml", &c, &binding);
        assert_eq!(implemented, 1);
        assert_eq!(total, 1);
    }

    #[test]
    fn truncate_helper() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn schema_version_present() {
        let report = proof_status_report(&[], None, false);
        assert_eq!(report.schema_version, "1.0.0");
    }

    #[test]
    fn timestamp_is_populated() {
        let report = proof_status_report(&[], None, false);
        assert!(!report.timestamp.is_empty());
        assert!(report.timestamp.ends_with('Z'));
    }
