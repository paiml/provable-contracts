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
            println!("cargo:rerun-if-changed={}", grandparent.display());
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

/// Verify that functions named in binding.yaml actually exist in the crate source.
///
/// Scans `src_dir` for `pub fn <name>` declarations and checks that every
/// `function` field in the binding registry has a matching source function.
/// Returns the names of missing functions. If the list is non-empty and
/// `hard_fail` is true, panics to fail the build.
///
/// This closes the "ghost binding" gap where `status: implemented` passes
/// build.rs but the actual function doesn't exist (renamed, deleted, typo).
///
/// # Example
///
/// ```rust,ignore
/// // build.rs
/// let missing = provable_contracts::build_helper::verify_source_functions(
///     "../provable-contracts/contracts/aprender/binding.yaml",
///     "src/",
///     true, // hard fail
/// );
/// ```
pub fn verify_source_functions(
    binding_yaml_path: &str,
    src_dir: &str,
    hard_fail: bool,
) -> Vec<String> {
    let path = Path::new(binding_yaml_path);
    let Ok(yaml_content) = std::fs::read_to_string(path) else {
        println!("cargo:warning=verify_source_functions: cannot read {binding_yaml_path}");
        return vec![];
    };
    let Ok(registry) = serde_yaml::from_str::<BindingRegistry>(&yaml_content) else {
        println!("cargo:warning=verify_source_functions: cannot parse {binding_yaml_path}");
        return vec![];
    };

    // Collect all function names from bindings
    let mut expected_fns: std::collections::HashSet<String> = std::collections::HashSet::new();
    for b in &registry.bindings {
        if b.status != ImplStatus::Implemented {
            continue;
        }
        if let Some(ref func) = b.function {
            // Extract just the function name (after last ::)
            let short = func.rsplit("::").next().unwrap_or(func);
            expected_fns.insert(short.to_lowercase());
        }
    }

    if expected_fns.is_empty() {
        return vec![];
    }

    // Scan source files for pub fn declarations
    let mut found_fns: std::collections::HashSet<String> = std::collections::HashSet::new();
    let src = Path::new(src_dir);
    if src.exists() {
        scan_source_fns(src, &mut found_fns);
    }
    // Also check crates/ subdirectory
    let crates_dir = Path::new("crates");
    if crates_dir.exists() {
        scan_source_fns(crates_dir, &mut found_fns);
    }

    let mut missing: Vec<String> = expected_fns
        .iter()
        .filter(|name| !found_fns.contains(name.as_str()))
        .cloned()
        .collect();
    missing.sort();

    if !missing.is_empty() {
        let count = missing.len();
        let sample: Vec<_> = missing.iter().take(10).collect();
        let msg = format!(
            "[contract] verify_source_functions: {count} bound function(s) not found in source: {}{}",
            sample
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if count > 10 {
                format!(" (and {} more)", count - 10)
            } else {
                String::new()
            },
        );

        if hard_fail {
            panic!("{msg}");
        } else {
            println!("cargo:warning={msg}");
        }
    }

    missing
}

/// Recursively scan a directory for `pub fn` declarations.
fn scan_source_fns(dir: &Path, found: &mut std::collections::HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name != "target" && name != ".git" {
                scan_source_fns(&path, found);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("pub(crate) fn ")
                    {
                        let fn_part = trimmed
                            .trim_start_matches("pub async fn ")
                            .trim_start_matches("pub(crate) fn ")
                            .trim_start_matches("pub fn ");
                        let fn_name = fn_part
                            .split('(')
                            .next()
                            .unwrap_or("")
                            .split('<')
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_lowercase();
                        if !fn_name.is_empty() {
                            found.insert(fn_name);
                        }
                    }
                }
            }
        }
    }
}

/// Generate the env var key from contract name and equation name.
///
/// Same convention as `provable-contracts-macros::make_env_key`.
fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
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

    #[test]
    fn verify_bindings_warn_on_gaps() {
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        // WarnOnGaps policy doesn't panic on partial/not_implemented
        let result = verify_bindings(binding_path.to_str().unwrap(), BindingPolicy::WarnOnGaps);
        assert!(
            result.bound_count > 0,
            "Should have some implemented bindings"
        );
    }

    #[test]
    fn make_env_key_with_yaml_extension() {
        assert_eq!(
            make_env_key("softmax-kernel-v1.yaml", "softmax"),
            "CONTRACT_SOFTMAX_KERNEL_V1_YAML_SOFTMAX"
        );
    }

    #[test]
    fn binding_policy_debug() {
        // Exercise Debug derive
        assert_eq!(
            format!("{:?}", BindingPolicy::AllImplemented),
            "AllImplemented"
        );
        assert_eq!(
            format!("{:?}", BindingPolicy::TieredEnforcement),
            "TieredEnforcement"
        );
    }
}
