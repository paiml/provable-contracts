use super::*;

fn test_contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/softmax-kernel-v1.yaml")
}

fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

#[test]
fn dispatch_validate() {
    let result = run_command(Commands::Validate {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_scaffold() {
    let result = run_command(Commands::Scaffold {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_kani() {
    let result = run_command(Commands::Kani {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_status() {
    let result = run_command(Commands::Status {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_diff() {
    let c = test_contract();
    let result = run_command(Commands::Diff {
        old: c.clone(),
        new: c,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_lean() {
    let result = run_command(Commands::Lean {
        contract: test_contract(),
        output_dir: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_lean_status() {
    let result = run_command(Commands::LeanStatus {
        path: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_lean_status_directory() {
    let result = run_command(Commands::LeanStatus {
        path: contracts_dir(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status() {
    let result = run_command(Commands::ProofStatus {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_json() {
    let result = run_command(Commands::ProofStatus {
        path: test_contract(),
        binding: None,
        format: "json".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_directory() {
    let result = run_command(Commands::ProofStatus {
        path: contracts_dir(),
        binding: None,
        format: "text".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_with_binding() {
    let binding =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml");
    let result = run_command(Commands::ProofStatus {
        path: contracts_dir(),
        binding: Some(binding),
        format: "json".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_single() {
    let result = run_command(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 5,
        weights: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_directory() {
    let result = run_command(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "json".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 5,
        weights: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_min_threshold_fails() {
    let result = run_command(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        min_score: Some(0.99),
        summary: false,
        top_gaps: 5,
        weights: None,
    });
    assert!(result.is_err());
}

#[test]
fn dispatch_score_custom_weights() {
    let result = run_command(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 5,
        weights: Some(
            r#"{"spec_depth":0.1,"falsification":0.3,"kani":0.3,"lean":0.1,"binding":0.2}"#
                .to_string(),
        ),
    });
    assert!(result.is_ok());
}
