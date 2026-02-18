//! Integration tests for Kani harness generation from contract YAML files.
//!
//! These tests verify that the full pipeline (parse → generate) produces
//! correct Kani harnesses for each strategy: exhaustive, `stub_float`,
//! and compositional.

use std::path::Path;

use provable_contracts::kani_gen::generate_kani_harnesses;
use provable_contracts::schema::parse_contract;

fn load_and_generate(path: &str) -> String {
    let path = Path::new(path);
    let contract = parse_contract(path)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    generate_kani_harnesses(&contract)
}

// --- Softmax: stub_float strategy ---

#[test]
fn softmax_generates_cfg_kani_module() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    assert!(code.starts_with("#[cfg(kani)]"));
    assert!(code.contains("mod verification"));
    assert!(code.contains("use super::*;"));
}

#[test]
fn softmax_generates_all_harnesses() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    assert!(code.contains("fn verify_softmax_normalization()"));
    assert!(code.contains("fn verify_softmax_positivity()"));
    assert!(code.contains("fn verify_softmax_bounded()"));
}

#[test]
fn softmax_harnesses_have_kani_attributes() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    // Each harness should have #[kani::proof]
    assert_eq!(code.matches("#[kani::proof]").count(), 3);
    // All softmax harnesses have bound=8 → unwind(9)
    assert_eq!(code.matches("#[kani::unwind(9)]").count(), 3);
    // First harness has solver=cadical
    assert!(code.contains("#[kani::solver(cadical)]"));
}

#[test]
fn softmax_harnesses_use_stub_float_strategy() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    // stub_float strategy should produce f32 symbolic inputs
    assert!(code.contains("Vec<f32>"));
    assert!(code.contains("is_finite()"));
    assert!(code.contains("stub_float"));
}

#[test]
fn softmax_harnesses_have_doc_comments() {
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    assert!(code.contains("/// KANI-SM-001: Softmax sums to 1.0"));
    assert!(code.contains("/// Obligation: SM-INV-001"));
    assert!(code.contains("/// Strategy: stub_float"));
    assert!(code.contains("/// Bound: 8 elements"));
}

// --- RMSNorm: mixed strategies ---

#[test]
fn rmsnorm_generates_harnesses() {
    let code = load_and_generate("../../contracts/rmsnorm-kernel-v1.yaml");
    assert!(code.starts_with("#[cfg(kani)]"));
    assert!(code.contains("#[kani::proof]"));
}

// --- Matmul: exhaustive strategy ---

#[test]
fn matmul_generates_exhaustive_harnesses() {
    let code = load_and_generate("../../contracts/matmul-kernel-v1.yaml");
    assert!(code.contains("#[kani::proof]"));
    // Exhaustive strategy uses i32 symbolic inputs
    assert!(code.contains("Vec<i32>"));
    assert!(code.contains("exhaustive"));
}

// --- Attention: stub_float strategy ---

#[test]
fn attention_generates_stub_float_harnesses() {
    let code = load_and_generate("../../contracts/attention-kernel-v1.yaml");
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("stub_float"));
}

// --- Flash Attention: stub_float strategy ---

#[test]
fn flash_attention_generates_harnesses() {
    let code = load_and_generate("../../contracts/flash-attention-v1.yaml");
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("mod verification"));
}

// --- Activation: exhaustive strategy ---

#[test]
fn activation_generates_exhaustive_harnesses() {
    let code = load_and_generate("../../contracts/activation-kernel-v1.yaml");
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("exhaustive"));
    assert!(code.contains("Vec<i32>"));
}

// --- RoPE: exhaustive strategy ---

#[test]
fn rope_generates_exhaustive_harnesses() {
    let code = load_and_generate("../../contracts/rope-kernel-v1.yaml");
    assert!(code.contains("#[kani::proof]"));
}

// --- Cross-contract structural tests ---

#[test]
fn all_contracts_generate_valid_kani_output() {
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
        // Every contract with harnesses should produce a cfg(kani) module
        assert!(
            code.starts_with("#[cfg(kani)]"),
            "{path} should generate cfg(kani) module"
        );
        // Every harness must have kani::proof attribute
        assert!(
            code.contains("#[kani::proof]"),
            "{path} should contain kani::proof"
        );
        // Every harness uses symbolic input (kani::any)
        assert!(
            code.contains("kani::any()"),
            "{path} should use symbolic inputs"
        );
        // Module should close properly
        assert!(code.ends_with("}\n"), "{path} should end with closing brace");
    }
}

#[test]
fn harness_count_matches_contract_definitions() {
    let contracts_and_counts = [
        ("../../contracts/softmax-kernel-v1.yaml", 3),
        ("../../contracts/matmul-kernel-v1.yaml", 1),
        ("../../contracts/activation-kernel-v1.yaml", 2),
    ];

    for (path, expected_count) in &contracts_and_counts {
        let code = load_and_generate(path);
        let actual = code.matches("#[kani::proof]").count();
        assert_eq!(
            actual, *expected_count,
            "{path}: expected {expected_count} harnesses, found {actual}"
        );
    }
}

// --- Concrete playback test ---

#[test]
fn generated_harness_contains_unimplemented_marker() {
    // All generated harnesses should contain unimplemented! marker
    // since they need manual wiring to actual kernel code
    let code = load_and_generate("../../contracts/softmax-kernel-v1.yaml");
    assert!(code.contains("unimplemented!"));
}
