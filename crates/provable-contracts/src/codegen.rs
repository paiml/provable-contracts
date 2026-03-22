//! Code generation from YAML contracts → Rust `debug_assert`!() checks.
//!
//! Reads contract YAML files and generates a Rust module with assertion
//! functions that can be called from production code. Zero cost in release.
//!
//! Also generates Lean 4 obligation stubs for unproved theorems.

use crate::schema::Contract;
use std::path::Path;

/// Generated contract enforcement code for a single contract.
#[derive(Debug, Clone)]
pub struct GeneratedContract {
    /// Contract name (from YAML filename stem).
    pub name: String,
    /// Generated Rust assertion functions.
    pub rust_assertions: String,
    /// Generated Lean 4 theorem stubs (for unproved obligations).
    pub lean_stubs: String,
    /// Number of preconditions generated.
    pub precondition_count: usize,
    /// Number of postconditions generated.
    pub postcondition_count: usize,
    /// Number of Lean theorems linked.
    pub lean_theorem_count: usize,
}

/// Generate Rust assertion code from a contract's equations.
///
/// For each equation with `preconditions` or `postconditions`, generates:
/// ```rust,ignore
/// pub fn check_gemv_preconditions(a_len: usize, rows: usize, cols: usize) {
///     debug_assert!(a_len == rows * cols, "Pre: a.len() == rows * cols");
/// }
/// ```
pub fn generate_from_contract(name: &str, contract: &Contract) -> GeneratedContract {
    let mut rust = String::new();
    let mut lean = String::new();
    let mut pre_count = 0;
    let mut post_count = 0;
    let mut lean_count = 0;

    rust.push_str(&format!(
        "// Auto-generated from contracts/{name}.yaml — DO NOT EDIT\n"
    ));
    rust.push_str(&format!("// Contract: {name}\n\n"));

    for (eq_name, equation) in &contract.equations {
        let macro_name = eq_name.replace('-', "_").to_lowercase();

        // Preconditions as a macro — caller passes variables
        if !equation.preconditions.is_empty() {
            rust.push_str(&format!("/// Preconditions for equation `{eq_name}`.\n"));
            rust.push_str(&format!(
                "/// Call at function entry: `contract_pre_{macro_name}!(var1, var2, ...)`\n"
            ));
            rust.push_str(&format!("macro_rules! contract_pre_{macro_name} {{\n"));
            rust.push_str("    ($($arg:ident),* $(,)?) => {{\n");
            for pre in &equation.preconditions {
                let escaped = pre.replace('"', "\\\"");
                rust.push_str(&format!(
                    "        debug_assert!({pre}, \"Pre-condition violated: {escaped}\");\n"
                ));
                pre_count += 1;
            }
            rust.push_str("    }};\n");
            rust.push_str("}\n\n");
        }

        // Postconditions as a macro
        if !equation.postconditions.is_empty() {
            rust.push_str(&format!("/// Postconditions for equation `{eq_name}`.\n"));
            rust.push_str(&format!(
                "/// Call before return: `contract_post_{macro_name}!(ret, var1, ...)`\n"
            ));
            rust.push_str(&format!("macro_rules! contract_post_{macro_name} {{\n"));
            rust.push_str("    ($($arg:ident),* $(,)?) => {{\n");
            for post in &equation.postconditions {
                let escaped = post.replace('"', "\\\"");
                rust.push_str(&format!(
                    "        debug_assert!({post}, \"Post-condition violated: {escaped}\");\n"
                ));
                post_count += 1;
            }
            rust.push_str("    }};\n");
            rust.push_str("}\n\n");
        }

        // Lean theorem linkage
        if let Some(ref theorem) = equation.lean_theorem {
            lean.push_str(&format!("-- Equation: {eq_name}\n"));
            lean.push_str(&format!("-- Lean theorem: {theorem}\n"));
            lean.push_str(&format!(
                "-- Formula: {}\n\n",
                equation.formula.lines().next().unwrap_or("")
            ));
            lean_count += 1;
        }
    }

    GeneratedContract {
        name: name.to_string(),
        rust_assertions: rust,
        lean_stubs: lean,
        precondition_count: pre_count,
        postcondition_count: post_count,
        lean_theorem_count: lean_count,
    }
}

/// Generate code for all contracts in a directory.
pub fn generate_all(contract_dir: &Path) -> Vec<GeneratedContract> {
    let mut results = Vec::new();
    let Ok(entries) = std::fs::read_dir(contract_dir) else {
        return results;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        match crate::schema::parse_contract(&path) {
            Ok(contract) => {
                let generated = generate_from_contract(&stem, &contract);
                if generated.precondition_count > 0
                    || generated.postcondition_count > 0
                    || generated.lean_theorem_count > 0
                {
                    results.push(generated);
                }
            }
            Err(_) => continue,
        }
    }

    results.sort_by(|a, b| a.name.cmp(&b.name));
    results
}

/// Write generated Rust code to a file.
pub fn write_rust_module(contracts: &[GeneratedContract], output: &Path) -> std::io::Result<()> {
    let mut content = String::new();
    content.push_str("//! Auto-generated contract assertions from YAML.\n");
    content.push_str("//! Zero cost in release builds (debug_assert!).\n");
    content.push_str("//! Regenerate with: pv codegen\n\n");
    content.push_str("#![allow(dead_code, unused_variables)]\n\n");

    let mut total_pre = 0;
    let mut total_post = 0;

    for c in contracts {
        content.push_str(&c.rust_assertions);
        total_pre += c.precondition_count;
        total_post += c.postcondition_count;
    }

    content.push_str(&format!(
        "// Total: {} preconditions, {} postconditions from {} contracts\n",
        total_pre,
        total_post,
        contracts.len()
    ));

    std::fs::write(output, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_generate_empty_contract() {
        let contract = Contract {
            metadata: crate::schema::Metadata {
                version: "1.0.0".into(),
                created: Some("2026-01-01".into()),
                author: Some("test".into()),
                description: "test".into(),
                references: vec![],
                depends_on: vec![],
                registry: false,
            },
            equations: BTreeMap::new(),
            proof_obligations: vec![],
            kernel_structure: None,
            simd_dispatch: BTreeMap::new(),
            enforcement: BTreeMap::new(),
            falsification_tests: vec![],
            kani_harnesses: vec![],
            qa_gate: None,
            verification_summary: None,
            type_invariants: vec![],
            coq_spec: None,
        };
        let generated = generate_from_contract("test", &contract);
        assert_eq!(generated.precondition_count, 0);
        assert_eq!(generated.postcondition_count, 0);
    }
}
