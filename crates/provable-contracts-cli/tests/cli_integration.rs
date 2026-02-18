use std::path::Path;
use std::process::Command;

/// Helper to get the path to a contract fixture.
fn contract_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .join(name)
}

/// Helper to get the binding registry path.
fn binding_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml")
}

/// Helper to get the pv binary path.
fn pv_bin() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pv"));
    // Fallback for test environments
    if !path.exists() {
        path = std::path::PathBuf::from("target/debug/pv");
    }
    path
}

// ================================================================
// validate command
// ================================================================

mod validate {
    use provable_contracts::error::Severity;
    use provable_contracts::schema::{parse_contract, validate_contract};

    #[test]
    fn valid_softmax_contract() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_rmsnorm_contract() {
        let path = super::contract_path("rmsnorm-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_rope_contract() {
        let path = super::contract_path("rope-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_activation_contract() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_attention_contract() {
        let path = super::contract_path("attention-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_matmul_contract() {
        let path = super::contract_path("matmul-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_flash_attention_contract() {
        let path = super::contract_path("flash-attention-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn invalid_yaml_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "not: [valid: {{").unwrap();
        let result = parse_contract(&path);
        assert!(result.is_err());
    }

    #[test]
    fn nonexistent_file_fails() {
        let result = parse_contract(std::path::Path::new("/nonexistent.yaml"));
        assert!(result.is_err());
    }
}

// ================================================================
// scaffold command
// ================================================================

mod scaffold {
    use provable_contracts::scaffold::{generate_contract_tests, generate_trait};
    use provable_contracts::schema::parse_contract;

    #[test]
    fn scaffold_softmax() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let trait_code = generate_trait(&contract);
        let test_code = generate_contract_tests(&contract);
        assert!(trait_code.contains("trait"));
        assert!(test_code.contains("#[test]"));
    }

    #[test]
    fn scaffold_activation() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let trait_code = generate_trait(&contract);
        assert!(trait_code.contains("fn"));
    }
}

// ================================================================
// kani command
// ================================================================

mod kani {
    use provable_contracts::kani_gen::generate_kani_harnesses;
    use provable_contracts::schema::parse_contract;

    #[test]
    fn kani_softmax() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("kani"));
    }

    #[test]
    fn kani_matmul() {
        let path = super::contract_path("matmul-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(!code.is_empty());
    }
}

// ================================================================
// probar command
// ================================================================

mod probar {
    use provable_contracts::binding::parse_binding;
    use provable_contracts::probar_gen::{generate_probar_tests, generate_wired_probar_tests};
    use provable_contracts::schema::parse_contract;

    #[test]
    fn probar_softmax_plain() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("probar_tests"));
        assert!(code.contains("#[test]"));
    }

    #[test]
    fn probar_softmax_wired() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).unwrap();
        let binding = parse_binding(&bp).unwrap();
        let code = generate_wired_probar_tests(&contract, "softmax-kernel-v1.yaml", &binding);
        assert!(code.contains("proptest!"));
        assert!(code.contains("softmax"));
    }

    #[test]
    fn probar_activation_wired() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).unwrap();
        let binding = parse_binding(&bp).unwrap();
        let code = generate_wired_probar_tests(&contract, "activation-kernel-v1.yaml", &binding);
        assert!(code.contains("CONTRACT: activation"));
    }

    #[test]
    fn probar_rmsnorm_plain() {
        let path = super::contract_path("rmsnorm-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("#[test]"));
    }
}

// ================================================================
// status command
// ================================================================

mod status {
    use provable_contracts::schema::parse_contract;

    #[test]
    fn status_softmax_has_qa_gate() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        assert!(contract.qa_gate.is_some());
        assert_eq!(contract.equations.len(), 1);
        assert_eq!(contract.proof_obligations.len(), 6);
        assert_eq!(contract.falsification_tests.len(), 6);
        assert_eq!(contract.kani_harnesses.len(), 3);
    }

    #[test]
    fn status_matmul_has_no_qa_gate() {
        let path = super::contract_path("matmul-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        // matmul may or may not have qa_gate, just check it parses
        assert!(!contract.equations.is_empty());
    }

    #[test]
    fn status_all_contracts_parse() {
        let contracts = [
            "softmax-kernel-v1.yaml",
            "rmsnorm-kernel-v1.yaml",
            "rope-kernel-v1.yaml",
            "activation-kernel-v1.yaml",
            "attention-kernel-v1.yaml",
            "matmul-kernel-v1.yaml",
            "flash-attention-v1.yaml",
        ];
        for name in contracts {
            let path = super::contract_path(name);
            let contract = parse_contract(&path).unwrap();
            assert!(
                !contract.metadata.description.is_empty(),
                "{name} has empty description"
            );
        }
    }
}

// ================================================================
// audit command
// ================================================================

mod audit {
    use provable_contracts::audit::{audit_binding, audit_contract};
    use provable_contracts::binding::parse_binding;
    use provable_contracts::error::Severity;
    use provable_contracts::schema::parse_contract;

    #[test]
    fn audit_softmax_contract() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let report = audit_contract(&contract);
        assert_eq!(report.equations, 1);
        assert_eq!(report.obligations, 6);
        assert_eq!(report.falsification_tests, 6);
        assert_eq!(report.kani_harnesses, 3);
    }

    #[test]
    fn audit_softmax_with_binding() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).unwrap();
        let binding = parse_binding(&bp).unwrap();
        let report = audit_binding(&[("softmax-kernel-v1.yaml", &contract)], &binding);
        assert_eq!(report.total_equations, 1);
        assert_eq!(report.implemented, 1);
        assert!(
            report
                .violations
                .iter()
                .all(|v| v.severity != Severity::Error)
        );
    }

    #[test]
    fn audit_activation_with_binding() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).unwrap();
        let binding = parse_binding(&bp).unwrap();
        let report = audit_binding(&[("activation-kernel-v1.yaml", &contract)], &binding);
        // activation has 3+ equations, silu is not_implemented
        assert!(report.total_equations >= 2);
    }

    #[test]
    fn audit_flash_attention_no_binding() {
        let path = super::contract_path("flash-attention-v1.yaml");
        let contract = parse_contract(&path).unwrap();
        let report = audit_contract(&contract);
        assert!(report.equations > 0);
    }
}

// ================================================================
// Binary integration tests (exercise CLI main + commands)
// ================================================================

mod binary {
    use super::*;

    #[test]
    fn pv_validate_softmax() {
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract is valid"));
    }

    #[test]
    fn pv_validate_invalid_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "{{invalid").unwrap();
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
    }

    #[test]
    fn pv_scaffold_softmax() {
        let output = Command::new(pv_bin())
            .arg("scaffold")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Trait Definition"));
    }

    #[test]
    fn pv_kani_softmax() {
        let output = Command::new(pv_bin())
            .arg("kani")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("kani"));
    }

    #[test]
    fn pv_probar_softmax() {
        let output = Command::new(pv_bin())
            .arg("probar")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("#[test]"));
    }

    #[test]
    fn pv_probar_with_binding() {
        let output = Command::new(pv_bin())
            .arg("probar")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("proptest!"));
    }

    #[test]
    fn pv_status_softmax() {
        let output = Command::new(pv_bin())
            .arg("status")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract:"));
        assert!(stdout.contains("QA gate:"));
    }

    #[test]
    fn pv_audit_softmax() {
        let output = Command::new(pv_bin())
            .arg("audit")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Traceability Audit"));
    }

    #[test]
    fn pv_audit_with_binding() {
        let output = Command::new(pv_bin())
            .arg("audit")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Binding Audit"));
    }

    #[test]
    fn pv_validate_nonexistent() {
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg("/nonexistent.yaml")
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
    }

    #[test]
    fn pv_status_matmul() {
        let output = Command::new(pv_bin())
            .arg("status")
            .arg(contract_path("matmul-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
    }

    #[test]
    fn pv_validate_with_warnings() {
        // Use a contract that has no qa_gate (generates SCHEMA-013 warning)
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("warn.yaml");
        std::fs::write(
            &path,
            r#"
metadata:
  version: "1.0.0"
  description: "Warn test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("0 error(s)"));
        assert!(stdout.contains("warning(s)"));
    }

    #[test]
    fn pv_validate_with_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("errors.yaml");
        std::fs::write(
            &path,
            r#"
metadata:
  version: "1.0.0"
  description: "Error test"
  references: []
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("error"));
    }

    #[test]
    fn pv_status_no_qa_gate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_qa.yaml");
        std::fs::write(
            &path,
            r#"
metadata:
  version: "1.0.0"
  description: "No QA"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();
        let output = Command::new(pv_bin())
            .arg("status")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("QA gate: not defined"));
    }

    #[test]
    fn pv_audit_with_errors() {
        let dir = tempfile::tempdir().unwrap();
        let contract = dir.path().join("audit_err.yaml");
        std::fs::write(
            &contract,
            r#"
metadata:
  version: "1.0.0"
  description: "Audit error"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: ""
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#,
        )
        .unwrap();
        let output = Command::new(pv_bin())
            .arg("audit")
            .arg(&contract)
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
    }

    #[test]
    fn pv_scaffold_activation() {
        let output = Command::new(pv_bin())
            .arg("scaffold")
            .arg(contract_path("activation-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract Tests"));
    }

    #[test]
    fn pv_probar_rope() {
        let output = Command::new(pv_bin())
            .arg("probar")
            .arg(contract_path("rope-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
    }

    #[test]
    fn pv_diff_identical() {
        let output = Command::new(pv_bin())
            .arg("diff")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("identical"));
    }

    #[test]
    fn pv_diff_different() {
        let output = Command::new(pv_bin())
            .arg("diff")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg(contract_path("rmsnorm-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("major"));
    }

    #[test]
    fn pv_coverage_contracts() {
        let output = Command::new(pv_bin())
            .arg("coverage")
            .arg(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../contracts"),
            )
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Obligation Coverage Report"));
        assert!(stdout.contains("softmax-kernel-v1"));
    }

    #[test]
    fn pv_coverage_with_binding() {
        let output = Command::new(pv_bin())
            .arg("coverage")
            .arg(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../contracts"),
            )
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Binding implemented"));
    }

    #[test]
    fn pv_generate_softmax() {
        let dir = tempfile::tempdir().unwrap();
        let output = Command::new(pv_bin())
            .arg("generate")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--output")
            .arg(dir.path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Generated"));
        assert!(stdout.contains("scaffold"));
    }

    #[test]
    fn pv_generate_with_binding() {
        let dir = tempfile::tempdir().unwrap();
        let output = Command::new(pv_bin())
            .arg("generate")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--output")
            .arg(dir.path())
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("wired-probar"));
    }

    #[test]
    fn pv_graph_contracts() {
        let output = Command::new(pv_bin())
            .arg("graph")
            .arg(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../contracts"),
            )
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract Dependency Graph"));
        assert!(stdout.contains("Topological order"));
    }
}
