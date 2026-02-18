//! Integration tests for probar property test generation from contract YAML files.
//!
//! These tests verify that the full pipeline (parse → generate) produces
//! correct probar property tests mapping obligation types to test patterns.

use std::path::Path;

use provable_contracts::probar_gen::generate_probar_tests;
use provable_contracts::schema::parse_contract;

fn load_and_generate(path: &str) -> String {
    let path = Path::new(path);
    let contract =
        parse_contract(path).unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    generate_probar_tests(&contract)
}

// --- Softmax: invariant and equivalence obligations ---

#[test]
fn softmax_generates_probar_module() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    assert!(code.contains("#[cfg(test)]"));
    assert!(code.contains("mod probar_tests"));
}

#[test]
fn softmax_maps_invariant_obligations() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    // Softmax has invariant obligations (SM-INV-001, SM-INV-002)
    assert!(code.contains("Pattern: invariant"));
}

#[test]
fn softmax_generates_falsification_stubs() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    // Should have falsification test stubs
    assert!(code.contains("Falsification test stubs"));
}

// --- Matmul: bound obligations ---

#[test]
fn matmul_generates_bound_tests() {
    let code = load_and_generate("../../contracts/matmul-kernel-v1.yaml");
    assert!(code.contains("#[cfg(test)]"));
    assert!(code.contains("Pattern: bound"));
}

// --- Attention: various obligation types ---

#[test]
fn attention_generates_probar_tests() {
    let code = load_and_generate("../../contracts/attention-kernel-v1.yaml");
    assert!(code.contains("#[cfg(test)]"));
    assert!(code.contains("mod probar_tests"));
}

// --- RMSNorm: invariant and equivalence ---

#[test]
fn rmsnorm_generates_equivalence_tests() {
    let code = load_and_generate("../../contracts/rmsnorm-kernel-v1.yaml");
    assert!(code.contains("#[cfg(test)]"));
}

// --- Cross-contract structural tests ---

#[test]
fn all_contracts_generate_valid_probar_output() {
    let contracts = [
        "../../contracts/softmax-kernel-v1.yaml",
        "../../contracts/rmsnorm-kernel-v1.yaml",
        "../../contracts/rope-kernel-v1.yaml",
        "../../contracts/activation-kernel-v1.yaml",
        "../../contracts/attention-kernel-v1.yaml",
        "../../contracts/matmul-kernel-v1.yaml",
        "../../contracts/flash-attention-v1.yaml",
    ];

    for path in &contracts {
        let code = load_and_generate(path);
        // Every contract should generate a test module
        assert!(
            code.contains("#[cfg(test)]"),
            "{path} should generate cfg(test) module"
        );
        // Every contract should have at least one test function
        assert!(
            code.contains("#[test]"),
            "{path} should contain test functions"
        );
    }
}

#[test]
fn contracts_with_obligations_generate_property_tests() {
    // These contracts have proof_obligations defined
    let contracts = [
        "../../contracts/softmax-kernel-v1.yaml",
        "../../contracts/rmsnorm-kernel-v1.yaml",
        "../../contracts/matmul-kernel-v1.yaml",
        "../../contracts/attention-kernel-v1.yaml",
    ];

    for path in &contracts {
        let code = load_and_generate(path);
        assert!(
            code.contains("proof obligations"),
            "{path} should have property tests from proof obligations"
        );
        assert!(
            code.contains("// Pattern:"),
            "{path} should show pattern type"
        );
    }
}

#[test]
fn contracts_with_falsification_tests_generate_stubs() {
    let contracts = [
        "../../contracts/softmax-kernel-v1.yaml",
        "../../contracts/rmsnorm-kernel-v1.yaml",
    ];

    for path in &contracts {
        let code = load_and_generate(path);
        assert!(
            code.contains("Falsification test stubs"),
            "{path} should have falsification test stubs"
        );
    }
}
