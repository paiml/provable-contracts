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
        pre_count +=
            emit_precondition_macro(&mut rust, eq_name, &macro_name, &equation.preconditions);
        post_count +=
            emit_postcondition_macro(&mut rust, eq_name, &macro_name, &equation.postconditions);
        emit_combined_macro(
            &mut rust,
            eq_name,
            &macro_name,
            &equation.preconditions,
            &equation.postconditions,
        );

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

/// Emit a precondition macro for an equation. Returns number of assertions emitted.
fn emit_precondition_macro(
    rust: &mut String,
    eq_name: &str,
    macro_name: &str,
    pres: &[String],
) -> usize {
    if pres.is_empty() {
        return 0;
    }
    let uses_domain = pres.iter().any(|p| {
        p.contains("==")
            || p.contains("eps")
            || p.contains("weight")
            || p.contains("freqs")
            || p.contains("scale")
            || p.contains('.') && !p.contains("is_empty")
    });
    let mut count = 0;
    rust.push_str(&format!("/// Preconditions for equation `{eq_name}`.\n"));
    if uses_domain {
        let pv = detect_primary_var(pres);
        rust.push_str(&format!(
            "/// Domain-specific. Call: `contract_pre_{macro_name}!(slice_expr)`\n"
        ));
        rust.push_str(&format!("macro_rules! contract_pre_{macro_name} {{\n"));
        // Zero-arg form: no-op (proc-macro compatibility)
        rust.push_str("    () => {{}};\n");
        rust.push_str("    ($input:expr) => {{\n");
        rust.push_str(&format!("        let {pv} = &$input;\n"));
        for pre in pres {
            if has_unbound_vars(pre, &pv) {
                continue;
            }
            let esc = pre.replace('"', "\\\"");
            rust.push_str(&format!("        debug_assert!({pre},\n            \"Contract {eq_name}: precondition violated — {esc}\");\n"));
            count += 1;
        }
        rust.push_str("    }};\n}\n\n");
    } else {
        rust.push_str(&format!(
            "/// Call at function entry: `contract_pre_{macro_name}!(input_expr)`\n"
        ));
        rust.push_str(&format!("macro_rules! contract_pre_{macro_name} {{\n"));
        rust.push_str("    () => {{}};\n");
        rust.push_str("    ($input:expr) => {{\n        let _contract_input = &$input;\n");
        for pre in pres {
            // Map common variable names to _contract_input
            let assertion = pre
                .replace("input", "_contract_input")
                .replace("x.", "_contract_input.")
                .replace("x)", "_contract_input)");
            // Skip assertions that still have unbound variables after substitution
            if has_unbound_vars(&assertion, "_contract_input") {
                continue;
            }
            let esc = pre.replace('"', "\\\"");
            rust.push_str(&format!("        debug_assert!({assertion},\n            \"Contract {eq_name}: precondition violated — {esc}\");\n"));
            count += 1;
        }
        rust.push_str("    }};\n}\n\n");
    }
    count
}

/// Emit a postcondition macro for an equation. Returns number of assertions emitted.
fn emit_postcondition_macro(
    rust: &mut String,
    eq_name: &str,
    macro_name: &str,
    posts: &[String],
) -> usize {
    if posts.is_empty() {
        return 0;
    }
    let mut count = 0;
    rust.push_str(&format!("/// Postconditions for equation `{eq_name}`.\n"));
    rust.push_str(&format!(
        "/// Call before return: `contract_post_{macro_name}!(result_expr)`\n"
    ));
    rust.push_str(&format!("macro_rules! contract_post_{macro_name} {{\n"));
    rust.push_str("    ($result:expr) => {{\n        let _contract_result = &$result;\n");
    for post in posts {
        let fixed = post.replace("result", "_contract_result");
        let esc = post.replace('"', "\\\"");
        rust.push_str(&format!("        debug_assert!({fixed}, \"Contract {eq_name}: postcondition violated — {esc}\");\n"));
        count += 1;
    }
    rust.push_str("    }};\n}\n\n");
    count
}

/// Emit a combined pre+post wrapper macro.
fn emit_combined_macro(
    rust: &mut String,
    eq_name: &str,
    macro_name: &str,
    pres: &[String],
    posts: &[String],
) {
    if pres.is_empty() || posts.is_empty() {
        return;
    }
    rust.push_str(&format!(
        "/// Combined pre+post contract for equation `{eq_name}`.\n"
    ));
    rust.push_str(&format!("macro_rules! contract_{macro_name} {{\n"));
    rust.push_str("    ($input:expr, $body:expr) => {{\n");
    rust.push_str(&format!("        contract_pre_{macro_name}!($input);\n"));
    rust.push_str("        let _contract_result = $body;\n");
    rust.push_str(&format!(
        "        contract_post_{macro_name}!(_contract_result);\n"
    ));
    rust.push_str("        _contract_result\n");
    rust.push_str("    }};\n}\n\n");
}

/// Detect the primary variable name used in preconditions.
/// Scans for the first `<var>.` pattern (e.g., `x.len()` → `x`).
fn detect_primary_var(preconditions: &[String]) -> String {
    for pre in preconditions {
        // Match patterns like "x.len()", "logits.iter()", "a.len()"
        if let Some(dot_pos) = pre.find('.') {
            let candidate = &pre[..dot_pos];
            // Must be a simple identifier (no spaces, operators)
            if !candidate.is_empty()
                && candidate.chars().all(|c| c.is_alphanumeric() || c == '_')
                && candidate != "result"
            {
                return candidate.to_string();
            }
        }
    }
    "x".to_string() // default fallback
}

/// Check if a precondition expression references variables beyond the primary
/// and standard library methods. Returns true if it has unbound names.
fn has_unbound_vars(expr: &str, primary_var: &str) -> bool {
    // Extract all identifiers that appear before `.` (method call targets)
    // or standalone (bare variables like m, k, n)
    let safe_names = [
        primary_var,
        "_contract_input",
        "true",
        "false",
        "f32",
        "f64",
        "usize",
        "i32",
        "i64",
    ];
    // Tokenize crudely: split on operators and delimiters
    for token in expr.split(|c: char| "().&|!<>=+- */%,;{}[]".contains(c)) {
        let token = token.trim();
        if token.is_empty() || token.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            continue; // skip empty, numeric literals
        }
        // Skip known safe identifiers and closures
        if safe_names.contains(&token)
            || token == "v"
            || token == "id"
            || token.starts_with("is_")
            || token == "iter"
            || token == "all"
            || token == "any"
            || token == "len"
            || token == "abs"
            || token == "sum"
        {
            continue;
        }
        // This token is an unbound variable
        if token.chars().all(|c| c.is_alphanumeric() || c == '_') && token.len() <= 20 {
            return true;
        }
    }
    false
}

/// Generate code for all contracts in a directory (recursive).
pub fn generate_all(contract_dir: &Path) -> Vec<GeneratedContract> {
    let mut yaml_paths = Vec::new();
    collect_yaml_files(contract_dir, &mut yaml_paths);

    let mut results = Vec::new();
    for path in &yaml_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        if let Ok(contract) = crate::schema::parse_contract(path) {
            let generated = generate_from_contract(&stem, &contract);
            if generated.precondition_count > 0
                || generated.postcondition_count > 0
                || generated.lean_theorem_count > 0
            {
                results.push(generated);
            }
        }
    }

    results.sort_by(|a, b| a.name.cmp(&b.name));
    results
}

/// Recursively collect `.yaml` contract files, skipping non-contract directories.
fn collect_yaml_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if dirname == "kaizen" || dirname == "legacy" || dirname == "pipelines" {
                continue;
            }
            collect_yaml_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml")
            && path.file_name().and_then(|n| n.to_str()) != Some("binding.yaml")
        {
            out.push(path);
        }
    }
}

/// Write generated Rust code to a file.
pub fn write_rust_module(contracts: &[GeneratedContract], output: &Path) -> std::io::Result<()> {
    let mut content = String::new();
    content.push_str("// Auto-generated contract assertions from YAML — DO NOT EDIT.\n");
    content.push_str("// Zero cost in release builds (debug_assert!).\n");
    content.push_str("// Regenerate: pv codegen contracts/ -o src/generated_contracts.rs\n");
    content.push_str(
        "// Include:   #[macro_use] #[allow(unused_macros)] mod generated_contracts;\n\n",
    );

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
                enforcement_level: None,
                locked_level: None,
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
