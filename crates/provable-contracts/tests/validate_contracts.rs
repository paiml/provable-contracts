use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::schema::{parse_contract, validate_contract};

fn validate_contract_file(path: &str) {
    let path = Path::new(path);
    assert!(path.exists(), "Contract file not found: {}", path.display());

    let contract = parse_contract(path)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

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
    validate_contract_file("../../contracts/softmax-kernel-v1.yaml");
}

#[test]
fn validate_rmsnorm_contract() {
    validate_contract_file("../../contracts/rmsnorm-kernel-v1.yaml");
}

#[test]
fn validate_rope_contract() {
    validate_contract_file("../../contracts/rope-kernel-v1.yaml");
}

#[test]
fn validate_activation_contract() {
    validate_contract_file("../../contracts/activation-kernel-v1.yaml");
}

#[test]
fn validate_attention_contract() {
    validate_contract_file("../../contracts/attention-kernel-v1.yaml");
}

#[test]
fn validate_matmul_contract() {
    validate_contract_file("../../contracts/matmul-kernel-v1.yaml");
}

#[test]
fn validate_flash_attention_contract() {
    validate_contract_file("../../contracts/flash-attention-v1.yaml");
}
