use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::schema::{parse_contract, validate_contract};

fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .canonicalize()
        .expect("contracts directory must exist")
}

fn all_contract_paths() -> Vec<std::path::PathBuf> {
    let dir = contracts_dir();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                && !path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with('.')
            {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths
}

fn validate_contract_file(path: &Path) {
    assert!(path.exists(), "Contract file not found: {}", path.display());

    let contract =
        parse_contract(path).unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

    let violations = validate_contract(&contract);
    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();

    assert!(
        errors.is_empty(),
        "Contract {} has validation errors: {:?}",
        path.display(),
        errors
    );
}

#[test]
fn validate_softmax_contract() {
    validate_contract_file(&contracts_dir().join("softmax-kernel-v1.yaml"));
}

#[test]
fn validate_rmsnorm_contract() {
    validate_contract_file(&contracts_dir().join("rmsnorm-kernel-v1.yaml"));
}

#[test]
fn validate_rope_contract() {
    validate_contract_file(&contracts_dir().join("rope-kernel-v1.yaml"));
}

#[test]
fn validate_activation_contract() {
    validate_contract_file(&contracts_dir().join("activation-kernel-v1.yaml"));
}

#[test]
fn validate_attention_contract() {
    validate_contract_file(&contracts_dir().join("attention-kernel-v1.yaml"));
}

#[test]
fn validate_matmul_contract() {
    validate_contract_file(&contracts_dir().join("matmul-kernel-v1.yaml"));
}

#[test]
fn validate_flash_attention_contract() {
    validate_contract_file(&contracts_dir().join("flash-attention-v1.yaml"));
}

#[test]
fn validate_all_contracts() {
    let paths = all_contract_paths();
    assert!(
        paths.len() >= 41,
        "Expected at least 41 contracts, found {}",
        paths.len()
    );
    for path in &paths {
        validate_contract_file(path);
    }
}
