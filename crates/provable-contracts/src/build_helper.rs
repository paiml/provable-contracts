//! Build script helper for consuming crates.
//!
//! Consuming crates (realizar, aprender, trueno, entrenar) use this module
//! in their `build.rs` to:
//!
//! 1. Read `binding.yaml` and extract all implemented bindings
//! 2. Set `CONTRACT_<NAME>_<EQ>=bound` env vars for each binding
//! 3. Fail the build if any binding has status `not_implemented`
//!
//! ## Usage in build.rs
//!
//! ```rust,ignore
//! // build.rs
//! fn main() {
//!     provable_contracts::build_helper::verify_bindings(
//!         "../provable-contracts/contracts/aprender/binding.yaml",
//!         BindingPolicy::AllImplemented,
//!     );
//! }
//! ```

use std::path::Path;

use crate::binding::{BindingRegistry, ImplStatus};

/// Policy for handling unimplemented bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingPolicy {
    /// All bindings must have status `implemented`. Any `partial` or
    /// `not_implemented` binding is a compile error.
    AllImplemented,

    /// Only `implemented` bindings get env vars. `partial` and
    /// `not_implemented` are warnings (printed to cargo stderr).
    WarnOnGaps,

    /// Tiered: `not_implemented` is an error, `partial` is a warning.
    TieredEnforcement,
}

/// Result of binding verification.
#[derive(Debug)]
pub struct VerifyResult {
    /// Number of bindings that got env vars set.
    pub bound_count: usize,
    /// Number of partial bindings (warnings).
    pub partial_count: usize,
    /// Number of not-implemented bindings (errors or warnings depending on policy).
    pub not_implemented_count: usize,
}

/// Read `binding.yaml` and set `CONTRACT_*` env vars for `#[contract]` macros.
///
/// Call this from your `build.rs`. It:
/// 1. Parses the binding YAML
/// 2. For each `implemented` binding, emits `cargo:rustc-env=CONTRACT_<KEY>=bound`
/// 3. Enforces the given policy for gaps
///
/// # Panics
///
/// Panics (failing the build) if:
/// - The binding YAML cannot be read or parsed
/// - Policy is `AllImplemented` and any binding is not `implemented`
/// - Policy is `TieredEnforcement` and any binding is `not_implemented`
pub fn verify_bindings(binding_yaml_path: &str, policy: BindingPolicy) -> VerifyResult {
    let path = Path::new(binding_yaml_path);

    // Rerun build.rs if binding.yaml changes
    println!("cargo:rerun-if-changed={binding_yaml_path}");

    // Also rerun if the contracts directory changes
    if let Some(parent) = path.parent() {
        if let Some(grandparent) = parent.parent() {
            println!(
                "cargo:rerun-if-changed={}",
                grandparent.display()
            );
        }
    }

    let yaml_content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "CONTRACT BUILD ERROR: Cannot read binding YAML at '{}': {e}\n\
             Hint: Ensure provable-contracts is checked out as a sibling directory.",
            path.display()
        );
    });

    let registry: BindingRegistry = serde_yaml::from_str(&yaml_content).unwrap_or_else(|e| {
        panic!(
            "CONTRACT BUILD ERROR: Cannot parse binding YAML at '{}': {e}",
            path.display()
        );
    });

    let mut result = VerifyResult {
        bound_count: 0,
        partial_count: 0,
        not_implemented_count: 0,
    };

    for binding in &registry.bindings {
        let env_key = make_env_key(&binding.contract, &binding.equation);

        match binding.status {
            ImplStatus::Implemented => {
                println!("cargo:rustc-env={env_key}=bound");
                result.bound_count += 1;
            }
            ImplStatus::Partial => {
                result.partial_count += 1;
                match policy {
                    BindingPolicy::AllImplemented => {
                        panic!(
                            "CONTRACT BUILD ERROR: Binding {}.{} has status 'partial'. \
                             Policy requires all bindings to be 'implemented'.\n\
                             Module: {}\n\
                             See: unified-contract-by-design.md §10",
                            binding.contract,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("(unknown)"),
                        );
                    }
                    BindingPolicy::WarnOnGaps | BindingPolicy::TieredEnforcement => {
                        println!(
                            "cargo:warning=CONTRACT: partial binding {}.{} ({})",
                            binding.contract,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("?"),
                        );
                        // Still set env var for partial — the function exists, just incomplete
                        println!("cargo:rustc-env={env_key}=partial");
                    }
                }
            }
            ImplStatus::NotImplemented => {
                result.not_implemented_count += 1;
                match policy {
                    BindingPolicy::AllImplemented | BindingPolicy::TieredEnforcement => {
                        panic!(
                            "CONTRACT BUILD ERROR: Binding {}.{} has status 'not_implemented'. \
                             All bindings must be implemented.\n\
                             Equation: {}\n\
                             Target: {}\n\
                             See: unified-contract-by-design.md §10",
                            binding.contract,
                            binding.equation,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("(unassigned)"),
                        );
                    }
                    BindingPolicy::WarnOnGaps => {
                        println!(
                            "cargo:warning=CONTRACT: not_implemented binding {}.{} ({})",
                            binding.contract,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("?"),
                        );
                    }
                }
            }
        }
    }

    println!(
        "cargo:warning=CONTRACT: {}/{} bindings bound ({} partial, {} not_implemented)",
        result.bound_count,
        registry.bindings.len(),
        result.partial_count,
        result.not_implemented_count,
    );

    result
}

/// Generate the env var key from contract name and equation name.
///
/// Same convention as `provable-contracts-macros::make_env_key`.
fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract
        .to_uppercase()
        .replace('-', "_")
        .replace('.', "_");
    let equation_part = equation
        .to_uppercase()
        .replace('-', "_")
        .replace('.', "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_env_key_matches_macro_convention() {
        assert_eq!(
            make_env_key("rmsnorm-kernel-v1", "rmsnorm"),
            "CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"
        );
        assert_eq!(
            make_env_key("gated-delta-net-v1", "decay"),
            "CONTRACT_GATED_DELTA_NET_V1_DECAY"
        );
    }

    #[test]
    fn test_verify_result_defaults() {
        let r = VerifyResult {
            bound_count: 0,
            partial_count: 0,
            not_implemented_count: 0,
        };
        assert_eq!(r.bound_count, 0);
    }
}
