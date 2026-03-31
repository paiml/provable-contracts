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
            let mut assertion = pre
                .replace("input", "_contract_input")
                .replace("x.", "_contract_input.")
                .replace("x)", "_contract_input)");
            // Handle !var.method() patterns — map leading var to _contract_input
            // Only for safe methods: is_empty, len, is_finite, iter (type-polymorphic)
            if has_unbound_vars(&assertion, "_contract_input") {
                let stripped = pre.trim_start_matches('!');
                if let Some(dot) = stripped.find('.') {
                    let var = &stripped[..dot];
                    let method = &stripped[dot + 1..];
                    // Only map for methods that exist on slices/vecs (not is_empty which fails on scalars)
                    let safe_method = method.starts_with("len()")
                        || method.starts_with("iter()")
                        || method.starts_with("is_finite()");
                    if safe_method
                        && !var.is_empty()
                        && var.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        let mapped = pre.replace(var, "_contract_input");
                        if !has_unbound_vars(&mapped, "_contract_input") {
                            assertion = mapped;
                        }
                    }
                }
            }
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
        // Replace result with *_contract_result for scalar comparisons (>= 0.0, etc.)
        // and _contract_result for method calls (.is_finite(), .iter(), .len())
        let fixed = if post.contains("result.") || post.contains("result)") {
            post.replace("result", "_contract_result")
        } else {
            // Scalar comparison: result >= 0.0 → *_contract_result >= 0.0
            post.replace("result", "*_contract_result")
        };
        // Skip postconditions that reference unbound variables (same hygiene fix as preconditions)
        if has_unbound_vars(&fixed, "_contract_result") {
            continue;
        }
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
    #![allow(clippy::all)]
    use super::*;
    use crate::schema::{Equation, Metadata};
    use std::collections::BTreeMap;

    fn make_metadata() -> Metadata {
        Metadata {
            version: "1.0.0".into(),
            created: Some("2026-01-01".into()),
            author: Some("test".into()),
            description: "test".into(),
            references: vec![],
            depends_on: vec![],
            registry: false,
            enforcement_level: None,
            locked_level: None,
        }
    }

    fn make_equation(
        formula: &str,
        pres: Vec<&str>,
        posts: Vec<&str>,
        lean_theorem: Option<&str>,
    ) -> Equation {
        Equation {
            formula: formula.to_string(),
            domain: None,
            codomain: None,
            invariants: vec![],
            preconditions: pres.into_iter().map(|s| s.to_string()).collect(),
            postconditions: posts.into_iter().map(|s| s.to_string()).collect(),
            lean_theorem: lean_theorem.map(|s| s.to_string()),
        }
    }

    fn make_contract(equations: BTreeMap<String, Equation>) -> Contract {
        Contract {
            metadata: make_metadata(),
            equations,
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
        }
    }

    // ---------------------------------------------------------------
    // generate_from_contract
    // ---------------------------------------------------------------

    #[test]
    fn test_generate_empty_contract() {
        let contract = make_contract(BTreeMap::new());
        let generated = generate_from_contract("test", &contract);
        assert_eq!(generated.precondition_count, 0);
        assert_eq!(generated.postcondition_count, 0);
        assert_eq!(generated.lean_theorem_count, 0);
        assert_eq!(generated.name, "test");
        assert!(generated.rust_assertions.contains("Auto-generated"));
        assert!(generated.rust_assertions.contains("Contract: test"));
    }

    #[test]
    fn test_generate_contract_with_preconditions_only() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-01".to_string(),
            make_equation("y = f(x)", vec!["!input.is_empty()"], vec![], None),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("pre-only", &contract);
        assert_eq!(out.precondition_count, 1);
        assert_eq!(out.postcondition_count, 0);
        assert!(out.rust_assertions.contains("contract_pre_eq_01"));
        assert!(!out.rust_assertions.contains("contract_post_"));
    }

    #[test]
    fn test_generate_contract_with_postconditions_only() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-02".to_string(),
            make_equation("y = g(x)", vec![], vec!["result >= 0.0"], None),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("post-only", &contract);
        assert_eq!(out.precondition_count, 0);
        assert_eq!(out.postcondition_count, 1);
        assert!(out.rust_assertions.contains("contract_post_eq_02"));
    }

    #[test]
    fn test_generate_contract_with_both_emits_combined() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-03".to_string(),
            make_equation(
                "y = h(x)",
                vec!["!input.is_empty()"],
                vec!["result >= 0.0"],
                None,
            ),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("combined", &contract);
        assert_eq!(out.precondition_count, 1);
        assert_eq!(out.postcondition_count, 1);
        // Combined macro should be emitted when both pre and post exist
        assert!(out.rust_assertions.contains("contract_eq_03"));
        assert!(out.rust_assertions.contains("contract_pre_eq_03!($input)"));
        assert!(
            out.rust_assertions
                .contains("contract_post_eq_03!(_contract_result)")
        );
    }

    #[test]
    fn test_generate_contract_with_lean_theorem() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-LEAN".to_string(),
            make_equation(
                "softmax(x)_i = exp(x_i) / sum(exp(x))",
                vec![],
                vec![],
                Some("ProvableContracts.Theorems.Softmax"),
            ),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("lean-test", &contract);
        assert_eq!(out.lean_theorem_count, 1);
        assert!(out.lean_stubs.contains("Equation: EQ-LEAN"));
        assert!(
            out.lean_stubs
                .contains("ProvableContracts.Theorems.Softmax")
        );
        assert!(out.lean_stubs.contains("Formula: softmax(x)_i"));
    }

    #[test]
    fn test_generate_contract_multiple_equations() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-A".to_string(),
            make_equation("a", vec!["!input.is_empty()"], vec![], None),
        );
        eqs.insert(
            "EQ-B".to_string(),
            make_equation("b", vec![], vec!["result >= 0.0"], None),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("multi", &contract);
        assert_eq!(out.precondition_count, 1);
        assert_eq!(out.postcondition_count, 1);
        assert!(out.rust_assertions.contains("contract_pre_eq_a"));
        assert!(out.rust_assertions.contains("contract_post_eq_b"));
    }

    #[test]
    fn test_generate_contract_hyphenated_name_becomes_underscore() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "GEMV-MxN".to_string(),
            make_equation("y=Ax", vec!["!input.is_empty()"], vec![], None),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("gemv", &contract);
        // Hyphens replaced with underscores, lowercased
        assert!(out.rust_assertions.contains("contract_pre_gemv_mxn"));
    }

    // ---------------------------------------------------------------
    // detect_primary_var
    // ---------------------------------------------------------------

    #[test]
    fn test_detect_primary_var_simple_dot() {
        let pres = vec!["x.len() > 0".to_string()];
        assert_eq!(detect_primary_var(&pres), "x");
    }

    #[test]
    fn test_detect_primary_var_logits() {
        let pres = vec!["logits.iter().all(|v| v.is_finite())".to_string()];
        assert_eq!(detect_primary_var(&pres), "logits");
    }

    #[test]
    fn test_detect_primary_var_skips_result() {
        // "result" should be skipped; fallback to next or default
        let pres = vec![
            "result.len() > 0".to_string(),
            "data.is_finite()".to_string(),
        ];
        assert_eq!(detect_primary_var(&pres), "data");
    }

    #[test]
    fn test_detect_primary_var_skips_operator_prefix() {
        // "3 + x" has a dot-less first entry, second has dot
        let pres = vec!["3 + y".to_string(), "a.len() > 0".to_string()];
        // First entry "3 + y" has no dot, skipped. "a.len()" → "a"
        assert_eq!(detect_primary_var(&pres), "a");
    }

    #[test]
    fn test_detect_primary_var_fallback_default() {
        // No dot patterns at all → fallback "x"
        let pres = vec!["true".to_string(), "42 > 0".to_string()];
        assert_eq!(detect_primary_var(&pres), "x");
    }

    #[test]
    fn test_detect_primary_var_skips_invalid_prefix() {
        // "a + b.len()" → prefix "a + b" contains spaces, not simple identifier
        let pres = vec!["a + b.len() > 0".to_string()];
        assert_eq!(detect_primary_var(&pres), "x"); // fallback
    }

    #[test]
    fn test_detect_primary_var_empty_list() {
        let pres: Vec<String> = vec![];
        assert_eq!(detect_primary_var(&pres), "x");
    }

    #[test]
    fn test_detect_primary_var_underscore_name() {
        let pres = vec!["my_weights.len() > 0".to_string()];
        assert_eq!(detect_primary_var(&pres), "my_weights");
    }

    // ---------------------------------------------------------------
    // has_unbound_vars
    // ---------------------------------------------------------------

    #[test]
    fn test_has_unbound_vars_simple_primary() {
        // Only uses primary var — no unbound
        assert!(!has_unbound_vars("x.len() > 0", "x"));
    }

    #[test]
    fn test_has_unbound_vars_with_extra_var() {
        // "rows" is unbound when primary is "x"
        assert!(has_unbound_vars("x.len() == rows * cols", "x"));
    }

    #[test]
    fn test_has_unbound_vars_numeric_literal() {
        // Numeric literals are not unbound
        assert!(!has_unbound_vars("x.len() > 0", "x"));
    }

    #[test]
    fn test_has_unbound_vars_contract_input() {
        // _contract_input is always safe
        assert!(!has_unbound_vars("_contract_input.len() > 0", "x"));
    }

    #[test]
    fn test_has_unbound_vars_safe_methods() {
        // "is_finite", "iter", "all", "len", etc. are safe
        assert!(!has_unbound_vars("x.iter().all(|v| v.is_finite())", "x"));
    }

    #[test]
    fn test_has_unbound_vars_bool_literals() {
        assert!(!has_unbound_vars("true", "x"));
        assert!(!has_unbound_vars("false", "x"));
    }

    #[test]
    fn test_has_unbound_vars_type_names() {
        // f32, f64, usize, etc. are safe
        assert!(!has_unbound_vars("f32::MAX", "x"));
        assert!(!has_unbound_vars("f64::MIN", "x"));
    }

    #[test]
    fn test_has_unbound_vars_abs_sum() {
        // "abs" and "sum" are safe names
        assert!(!has_unbound_vars("x.iter().sum()", "x"));
    }

    #[test]
    fn test_has_unbound_vars_v_and_id_safe() {
        // "v" and "id" are special safe names (closure vars)
        assert!(!has_unbound_vars("x.iter().all(|v| v > 0)", "x"));
    }

    #[test]
    fn test_has_unbound_vars_long_token_not_flagged() {
        // Tokens > 20 chars are not flagged as unbound
        assert!(!has_unbound_vars(
            "some_very_long_identifier_name_exceeding_twenty_chars",
            "x"
        ));
    }

    #[test]
    fn test_has_unbound_vars_empty_expr() {
        assert!(!has_unbound_vars("", "x"));
    }

    #[test]
    fn test_has_unbound_vars_only_numbers() {
        assert!(!has_unbound_vars("42 + 7", "x"));
    }

    #[test]
    fn test_has_unbound_vars_with_any() {
        assert!(!has_unbound_vars("x.iter().any(|v| v.is_finite())", "x"));
    }

    // ---------------------------------------------------------------
    // emit_precondition_macro — domain-specific path
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_precondition_domain_path_contains_eq() {
        // Triggers domain path because "==" is present
        let pres = vec!["x.len() == 10".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-DOM", "eq_dom", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("let x = &$input;"));
        assert!(rust.contains("Domain-specific"));
        assert!(rust.contains("x.len() == 10"));
    }

    #[test]
    fn test_emit_precondition_domain_path_contains_eps() {
        let pres = vec!["eps > 0.0".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-EPS", "eq_eps", &pres);
        // "eps" triggers domain but eps itself is an unbound var → skipped
        assert_eq!(count, 0);
        assert!(rust.contains("Domain-specific"));
    }

    #[test]
    fn test_emit_precondition_domain_path_contains_weight() {
        let pres = vec!["weight.len() > 0".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-W", "eq_w", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("let weight = &$input;"));
    }

    #[test]
    fn test_emit_precondition_domain_path_contains_freqs() {
        let pres = vec!["freqs.len() > 0".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-F", "eq_f", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("let freqs = &$input;"));
    }

    #[test]
    fn test_emit_precondition_domain_path_contains_scale() {
        let pres = vec!["scale.is_finite()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-S", "eq_s", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("let scale = &$input;"));
    }

    #[test]
    fn test_emit_precondition_domain_path_dot_not_is_empty() {
        // "a.len() > 0" has a dot and does NOT contain "is_empty" → domain
        let pres = vec!["a.len() > 0".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-D", "eq_d", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("Domain-specific"));
    }

    #[test]
    fn test_emit_precondition_domain_skips_unbound() {
        // "a.len() == rows * cols" — rows and cols are unbound, skipped
        let pres = vec!["a.len() == rows * cols".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-UB", "eq_ub", &pres);
        assert_eq!(count, 0); // skipped due to unbound vars
    }

    #[test]
    fn test_emit_precondition_domain_zero_arg_form() {
        let pres = vec!["x.len() == 10".to_string()];
        let mut rust = String::new();
        emit_precondition_macro(&mut rust, "EQ-ZA", "eq_za", &pres);
        // Zero-arg form always present
        assert!(rust.contains("() => {{}};"));
    }

    // ---------------------------------------------------------------
    // emit_precondition_macro — generic path
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_precondition_generic_path_input_replacement() {
        // "!input.is_empty()" — generic path, "input" → "_contract_input"
        // Note: ".is_empty" in precondition prevents domain detection
        let pres = vec!["!input.is_empty()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-GEN", "eq_gen", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_input"));
        assert!(!rust.contains("Domain-specific"));
    }

    #[test]
    fn test_emit_precondition_generic_path_x_dot_replacement() {
        // "x.is_empty()" should NOT trigger domain (because it contains "is_empty")
        // Wait, the condition is: `p.contains('.') && !p.contains("is_empty")`
        // "!x.is_empty()" contains both '.' AND "is_empty", so the dot rule doesn't fire.
        // But neither do ==, eps, weight, freqs, scale → generic path
        let pres = vec!["!x.is_empty()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-X", "eq_x", &pres);
        assert_eq!(count, 1);
        // "x." → "_contract_input."
        assert!(rust.contains("_contract_input.is_empty()"));
    }

    #[test]
    fn test_emit_precondition_generic_path_x_paren_replacement() {
        // Replaces "x)" with "_contract_input)"
        let pres = vec!["!x.is_empty()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-XP", "eq_xp", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_input"));
    }

    #[test]
    fn test_emit_precondition_generic_safe_method_mapping() {
        // Generic path with a var.len() pattern — should map to _contract_input
        // "arr.len() > 0" — has dot, no "is_empty" → domain path triggers
        // Need a case where the generic path's safe-method rewrite kicks in.
        // The expression must not contain ==, eps, weight, freqs, scale,
        // and must contain is_empty (to suppress the dot rule)
        // but use a different method...
        // Actually, let me think about this more carefully.
        // The generic path safe_method rewrite: handles var.len(), var.iter(), var.is_finite()
        // It only fires in the generic path when has_unbound_vars is true after initial substitution.
        // Example: "data.len() > 0" where "data" isn't replaced by input/x replacements.
        // But "data.len()" has a dot and no "is_empty" → domain path.
        // To hit the generic path safe method mapping we need:
        // 1. Expression not triggering domain (no ==, eps, weight, freqs, scale, no non-is_empty dot)
        // 2. After input/x substitution, still has unbound vars
        // 3. Has var.safe_method() pattern
        // This is hard to construct since the dot rule catches most .method() patterns.
        // Only way: the expression must contain ".is_empty" so the dot rule doesn't fire,
        // but also have a second assertion in the list that uses a safe method.
        // Actually: the dot test is `p.contains('.') && !p.contains("is_empty")`.
        // So if ALL pres contain is_empty, no domain. But we can mix:
        // If one pre has ".is_empty" and another has nothing → uses_domain checks any().
        // We need ALL pres to not trigger domain for the whole block to be generic.
        // Each pre is checked via .any() for domain triggers.
        // So even one pre with "a.len()" (dot without is_empty) triggers domain for the whole block.
        //
        // The only way to get into the generic safe-method path is:
        // pres = ["buf.is_empty()"] — dot with is_empty doesn't trigger, no ==, eps, weight, freqs, scale
        // → generic path. Then "buf" is not "input" or "x", so initial substitution doesn't map it.
        // Then has_unbound_vars check fires → safe_method check: "is_empty()" doesn't start with "len()", "iter()", "is_finite()"
        // Hmm. is_empty is NOT in the safe list. So it won't be remapped.
        // Let's try "buf.len() > 0" as the sole pre with is_empty somehow...
        // Actually I realize: for domain detection, the check is:
        //   pres.iter().any(|p| { ... p.contains('.') && !p.contains("is_empty") })
        // So "buf.len() > 0" → contains('.') && !contains("is_empty") → domain!
        // The only way to the generic path with a dot is if the dot expr also contains "is_empty".
        // Like "buf.is_empty()" which is_empty(). But is_empty isn't a safe_method.
        //
        // Let's try: "buf.is_finite()" — contains '.', but also... does NOT contain "is_empty"
        // → triggers domain. So we can't get a generic path + safe method + dot easily.
        //
        // OK let me try the simplest case that hits the generic safe method:
        // We need: the precondition contains a dot, but domain is NOT triggered.
        // Domain is triggered by: ==, eps, weight, freqs, scale, (. && !is_empty).
        // The last rule means ANY dot without is_empty triggers domain.
        // So the ONLY dots that don't trigger domain have is_empty next to them.
        //
        // For the safe_method rewrite in the generic path, we need:
        // 1. After initial substitution, assertion still has unbound vars
        // 2. Original pre has a dot
        // 3. Method starts with len(), iter(), or is_finite()
        //
        // But rule: if pre has a dot and NOT is_empty → domain.
        // "data.len()" has dot, no is_empty → domain.
        //
        // Therefore the safe_method rewrite in the generic path can only be reached
        // when ALL of these are in the pre:
        // - Contains a dot
        // - Also contains "is_empty" in the string (to suppress domain for the .any() check)
        //
        // Like: "data.len() > 0 && !data.is_empty()" — this has both dot and is_empty
        // Wait, the check is per-pre: `p.contains('.') && !p.contains("is_empty")`
        // So "data.len() > 0 && !data.is_empty()" → contains('.') = true, contains("is_empty") = true
        // → the conjunction is false for THIS pre. But if another pre doesn't have is_empty...
        // We need NO pre to trigger domain. So all pres must either have no dot, or have dot+is_empty.
        //
        // Example: pres = ["data.len() > 0 && !data.is_empty()"]
        // This does not trigger domain (dot + is_empty → false).
        // No ==, no eps, no weight, no freqs, no scale → generic path!
        // After initial substitution: "data" isn't input/x → stays "data" → still unbound.
        // Then safe_method check: stripped = "data.len() > 0 && !data.is_empty()"
        // dot_pos = first dot = 4. var = "data", method = "len() > 0 && !data.is_empty()"
        // method.starts_with("len()") → true! var is alphanumeric. So it maps "data" → "_contract_input".
        // Then the mapped expr: "_contract_input.len() > 0 && !_contract_input.is_empty()"
        // has_unbound_vars check on mapped → no unbound → assertion = mapped. count += 1.
        let pres = vec!["data.len() > 0 && !data.is_empty()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-SM", "eq_sm", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_input.len() > 0"));
    }

    #[test]
    fn test_emit_precondition_generic_skips_still_unbound() {
        // After all substitutions, "custom_var > 0" still unbound → skipped
        // No dot → no domain trigger. No input/x → no initial substitution.
        let pres = vec!["custom_var > 0".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-SK", "eq_sk", &pres);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_emit_precondition_empty() {
        let pres: Vec<String> = vec![];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-E", "eq_e", &pres);
        assert_eq!(count, 0);
        assert!(rust.is_empty());
    }

    #[test]
    fn test_emit_precondition_multiple_domain() {
        let pres = vec![
            "x.len() == 10".to_string(),
            "x.iter().all(|v| v.is_finite())".to_string(),
        ];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-MD", "eq_md", &pres);
        assert_eq!(count, 2);
        assert!(rust.contains("x.len() == 10"));
        assert!(rust.contains("x.iter().all(|v| v.is_finite())"));
    }

    #[test]
    fn test_emit_precondition_generic_with_input_var() {
        // "input.len() > 0" — no ==, but has '.' without is_empty → domain!
        // Actually: '.' present and no "is_empty" → domain.
        // Let me use something that's purely generic:
        // "input > 0" — no dot, no domain triggers
        let pres = vec!["input > 0".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-IN", "eq_in", &pres);
        assert_eq!(count, 1);
        // "input" replaced with "_contract_input"
        assert!(rust.contains("_contract_input > 0"));
    }

    #[test]
    fn test_emit_precondition_domain_quote_escaping() {
        let pres = vec!["x.len() == 10".to_string()];
        let mut rust = String::new();
        emit_precondition_macro(&mut rust, "EQ-QE", "eq_qe", &pres);
        // The escaped message should contain the original precondition text
        assert!(rust.contains("precondition violated"));
    }

    // ---------------------------------------------------------------
    // emit_postcondition_macro
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_postcondition_empty() {
        let posts: Vec<String> = vec![];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-PE", "eq_pe", &posts);
        assert_eq!(count, 0);
        assert!(rust.is_empty());
    }

    #[test]
    fn test_emit_postcondition_scalar_result() {
        // "result >= 0.0" — no "result." or "result)" → scalar path
        // "result" → "*_contract_result"
        let posts = vec!["result >= 0.0".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-PS", "eq_ps", &posts);
        assert_eq!(count, 1);
        assert!(rust.contains("*_contract_result >= 0.0"));
        assert!(rust.contains("postcondition violated"));
    }

    #[test]
    fn test_emit_postcondition_method_call_on_result() {
        // "result.len() > 0" — contains "result." → method path
        // "result" → "_contract_result"
        let posts = vec!["result.len() > 0".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-PM", "eq_pm", &posts);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_result.len() > 0"));
        // Should NOT have the dereference *
        assert!(!rust.contains("*_contract_result.len()"));
    }

    #[test]
    fn test_emit_postcondition_result_paren() {
        // "result)" — contains "result)" → method path, "result" → "_contract_result"
        // But "foo" is an unbound var → skipped (count 0)
        let posts = vec!["foo(result)".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-RP", "eq_rp", &posts);
        assert_eq!(count, 0); // skipped: "foo" is unbound

        // Use a safe expression with result) that has no unbound vars
        let posts2 = vec!["result.iter().all(|v| v.is_finite())".to_string()];
        let mut rust2 = String::new();
        let count2 = emit_postcondition_macro(&mut rust2, "EQ-RP2", "eq_rp2", &posts2);
        assert_eq!(count2, 1);
        assert!(rust2.contains("_contract_result.iter()"));
    }

    #[test]
    fn test_emit_postcondition_skips_unbound() {
        // "result.len() == expected_len" — "expected_len" is unbound
        let posts = vec!["result.len() == expected_len".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-PU", "eq_pu", &posts);
        assert_eq!(count, 0); // skipped
    }

    #[test]
    fn test_emit_postcondition_multiple() {
        let posts = vec!["result >= 0.0".to_string(), "result <= 1.0".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-MU", "eq_mu", &posts);
        assert_eq!(count, 2);
        assert!(rust.contains("*_contract_result >= 0.0"));
        assert!(rust.contains("*_contract_result <= 1.0"));
    }

    #[test]
    fn test_emit_postcondition_is_finite() {
        let posts = vec!["result.is_finite()".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-FI", "eq_fi", &posts);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_result.is_finite()"));
    }

    // ---------------------------------------------------------------
    // emit_combined_macro
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_combined_macro_both_present() {
        let pres = vec!["x > 0".to_string()];
        let posts = vec!["result > 0".to_string()];
        let mut rust = String::new();
        emit_combined_macro(&mut rust, "EQ-C", "eq_c", &pres, &posts);
        assert!(rust.contains("macro_rules! contract_eq_c"));
        assert!(rust.contains("contract_pre_eq_c!($input)"));
        assert!(rust.contains("contract_post_eq_c!(_contract_result)"));
        assert!(rust.contains("_contract_result"));
    }

    #[test]
    fn test_emit_combined_macro_empty_pres() {
        let pres: Vec<String> = vec![];
        let posts = vec!["result > 0".to_string()];
        let mut rust = String::new();
        emit_combined_macro(&mut rust, "EQ-CP", "eq_cp", &pres, &posts);
        assert!(rust.is_empty()); // not emitted
    }

    #[test]
    fn test_emit_combined_macro_empty_posts() {
        let pres = vec!["x > 0".to_string()];
        let posts: Vec<String> = vec![];
        let mut rust = String::new();
        emit_combined_macro(&mut rust, "EQ-CE", "eq_ce", &pres, &posts);
        assert!(rust.is_empty()); // not emitted
    }

    #[test]
    fn test_emit_combined_macro_both_empty() {
        let pres: Vec<String> = vec![];
        let posts: Vec<String> = vec![];
        let mut rust = String::new();
        emit_combined_macro(&mut rust, "EQ-BE", "eq_be", &pres, &posts);
        assert!(rust.is_empty());
    }

    // ---------------------------------------------------------------
    // write_rust_module
    // ---------------------------------------------------------------

    #[test]
    fn test_write_rust_module() {
        let contracts = vec![
            GeneratedContract {
                name: "alpha".to_string(),
                rust_assertions: "// alpha code\n".to_string(),
                lean_stubs: String::new(),
                precondition_count: 2,
                postcondition_count: 1,
                lean_theorem_count: 0,
            },
            GeneratedContract {
                name: "beta".to_string(),
                rust_assertions: "// beta code\n".to_string(),
                lean_stubs: String::new(),
                precondition_count: 0,
                postcondition_count: 3,
                lean_theorem_count: 0,
            },
        ];
        let dir = std::env::temp_dir().join("codegen_test_write");
        std::fs::create_dir_all(&dir).unwrap();
        let out_path = dir.join("generated.rs");
        write_rust_module(&contracts, &out_path).unwrap();
        let content = std::fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("Auto-generated"));
        assert!(content.contains("// alpha code"));
        assert!(content.contains("// beta code"));
        assert!(content.contains("Total: 2 preconditions, 4 postconditions from 2 contracts"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_rust_module_empty() {
        let contracts: Vec<GeneratedContract> = vec![];
        let dir = std::env::temp_dir().join("codegen_test_write_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let out_path = dir.join("empty.rs");
        write_rust_module(&contracts, &out_path).unwrap();
        let content = std::fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("Total: 0 preconditions, 0 postconditions from 0 contracts"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ---------------------------------------------------------------
    // collect_yaml_files
    // ---------------------------------------------------------------

    #[test]
    fn test_collect_yaml_files_basic() {
        let dir = std::env::temp_dir().join("codegen_test_collect");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("foo-v1.yaml"), "metadata: {}").unwrap();
        std::fs::write(dir.join("bar-v1.yaml"), "metadata: {}").unwrap();
        std::fs::write(dir.join("not-a-yaml.txt"), "nope").unwrap();
        let mut files = Vec::new();
        collect_yaml_files(&dir, &mut files);
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.extension().unwrap() == "yaml"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_collect_yaml_files_skips_binding() {
        let dir = std::env::temp_dir().join("codegen_test_binding");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("binding.yaml"), "bindings: []").unwrap();
        std::fs::write(dir.join("real-v1.yaml"), "metadata: {}").unwrap();
        let mut files = Vec::new();
        collect_yaml_files(&dir, &mut files);
        assert_eq!(files.len(), 1);
        assert!(files[0].file_name().unwrap() != "binding.yaml");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_collect_yaml_files_skips_special_dirs() {
        let dir = std::env::temp_dir().join("codegen_test_skipdir");
        std::fs::create_dir_all(dir.join("kaizen")).unwrap();
        std::fs::create_dir_all(dir.join("legacy")).unwrap();
        std::fs::create_dir_all(dir.join("pipelines")).unwrap();
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::write(dir.join("kaizen/k.yaml"), "").unwrap();
        std::fs::write(dir.join("legacy/l.yaml"), "").unwrap();
        std::fs::write(dir.join("pipelines/p.yaml"), "").unwrap();
        std::fs::write(dir.join("models/m.yaml"), "").unwrap();
        std::fs::write(dir.join("top.yaml"), "").unwrap();
        let mut files = Vec::new();
        collect_yaml_files(&dir, &mut files);
        // Only top.yaml and models/m.yaml — kaizen, legacy, pipelines skipped
        assert_eq!(files.len(), 2);
        let names: Vec<_> = files
            .iter()
            .map(|f| f.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert!(names.contains(&"top.yaml".to_string()));
        assert!(names.contains(&"m.yaml".to_string()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_collect_yaml_files_nonexistent_dir() {
        let mut files = Vec::new();
        collect_yaml_files(Path::new("/tmp/nonexistent_dir_codegen_test"), &mut files);
        assert!(files.is_empty());
    }

    // ---------------------------------------------------------------
    // generate_all — integration
    // ---------------------------------------------------------------

    #[test]
    fn test_generate_all_with_valid_contract() {
        let dir = std::env::temp_dir().join("codegen_test_gen_all");
        std::fs::create_dir_all(&dir).unwrap();
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "test contract"
equations:
  EQ-01:
    formula: "y = x + 1"
    preconditions:
      - "!input.is_empty()"
    postconditions:
      - "result >= 0.0"
proof_obligations: []
falsification_tests: []
kani_harnesses: []
"#;
        std::fs::write(dir.join("test-contract-v1.yaml"), yaml).unwrap();
        let results = generate_all(&dir);
        assert!(!results.is_empty());
        let out = &results[0];
        assert_eq!(out.name, "test-contract-v1");
        assert!(out.precondition_count > 0 || out.postcondition_count > 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_generate_all_skips_invalid_yaml() {
        let dir = std::env::temp_dir().join("codegen_test_gen_all_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("bad.yaml"), "not: valid: yaml: [[[").unwrap();
        let results = generate_all(&dir);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_generate_all_skips_no_assertions_contract() {
        let dir = std::env::temp_dir().join("codegen_test_gen_all_empty_eq");
        std::fs::create_dir_all(&dir).unwrap();
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "empty contract"
equations: {}
proof_obligations: []
falsification_tests: []
kani_harnesses: []
"#;
        std::fs::write(dir.join("empty-v1.yaml"), yaml).unwrap();
        let results = generate_all(&dir);
        // Contract with no preconditions/postconditions/lean theorems is skipped
        assert!(results.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_generate_all_sorted_by_name() {
        let dir = std::env::temp_dir().join("codegen_test_gen_all_sort");
        std::fs::create_dir_all(&dir).unwrap();
        let yaml_template = |name: &str| {
            format!(
                r#"
metadata:
  version: "1.0.0"
  description: "{name}"
equations:
  EQ-01:
    formula: "y = x"
    preconditions:
      - "!input.is_empty()"
proof_obligations: []
falsification_tests: []
kani_harnesses: []
"#
            )
        };
        std::fs::write(dir.join("zebra-v1.yaml"), yaml_template("zebra")).unwrap();
        std::fs::write(dir.join("alpha-v1.yaml"), yaml_template("alpha")).unwrap();
        std::fs::write(dir.join("middle-v1.yaml"), yaml_template("middle")).unwrap();
        let results = generate_all(&dir);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].name, "alpha-v1");
        assert_eq!(results[1].name, "middle-v1");
        assert_eq!(results[2].name, "zebra-v1");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ---------------------------------------------------------------
    // Lean theorem linkage (multiline formula)
    // ---------------------------------------------------------------

    #[test]
    fn test_lean_stub_multiline_formula() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-ML".to_string(),
            make_equation(
                "line1\nline2\nline3",
                vec![],
                vec![],
                Some("Theorem.Multiline"),
            ),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("ml-test", &contract);
        assert_eq!(out.lean_theorem_count, 1);
        // Only first line of formula should appear
        assert!(out.lean_stubs.contains("Formula: line1"));
        assert!(!out.lean_stubs.contains("line2"));
    }

    #[test]
    fn test_lean_stub_empty_formula() {
        let mut eqs = BTreeMap::new();
        eqs.insert(
            "EQ-EF".to_string(),
            make_equation("", vec![], vec![], Some("Theorem.Empty")),
        );
        let contract = make_contract(eqs);
        let out = generate_from_contract("ef-test", &contract);
        assert_eq!(out.lean_theorem_count, 1);
        assert!(out.lean_stubs.contains("Formula: "));
    }

    // ---------------------------------------------------------------
    // Edge cases: postcondition escaping
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_postcondition_quote_escaping() {
        // "result == \"ok\"" — scalar path: result → *_contract_result
        // The quoted "ok" is not purely alphanumeric (has "), so not flagged unbound → emitted
        let posts = vec!["result == \"ok\"".to_string()];
        let mut rust = String::new();
        let count = emit_postcondition_macro(&mut rust, "EQ-QP", "eq_qp", &posts);
        assert_eq!(count, 1);
        // The error message should escape the quotes
        assert!(rust.contains("postcondition violated"));
    }

    // ---------------------------------------------------------------
    // GeneratedContract struct
    // ---------------------------------------------------------------

    #[test]
    fn test_generated_contract_clone_debug() {
        let gc = GeneratedContract {
            name: "test".to_string(),
            rust_assertions: "code".to_string(),
            lean_stubs: "lean".to_string(),
            precondition_count: 1,
            postcondition_count: 2,
            lean_theorem_count: 3,
        };
        let cloned = gc.clone();
        assert_eq!(cloned.name, gc.name);
        assert_eq!(cloned.precondition_count, gc.precondition_count);
        // Debug derive
        let dbg = format!("{:?}", gc);
        assert!(dbg.contains("GeneratedContract"));
    }

    // ---------------------------------------------------------------
    // Domain detection edge cases
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_precondition_domain_dot_with_is_empty_not_domain() {
        // ".is_empty" suppresses the dot domain rule
        let pres = vec!["!x.is_empty()".to_string()];
        let mut rust = String::new();
        emit_precondition_macro(&mut rust, "EQ-IE", "eq_ie", &pres);
        // Should be generic path, not domain
        assert!(!rust.contains("Domain-specific"));
    }

    #[test]
    fn test_emit_precondition_mixed_domain_triggers() {
        // One pre has == → domain for the whole block
        let pres = vec![
            "!input.is_empty()".to_string(),
            "input.len() == 10".to_string(),
        ];
        let mut rust = String::new();
        emit_precondition_macro(&mut rust, "EQ-MIX", "eq_mix", &pres);
        assert!(rust.contains("Domain-specific"));
    }

    // ---------------------------------------------------------------
    // Generic path: iter/is_finite safe method rewrite
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_precondition_generic_iter_safe_method() {
        // "arr.iter().all(|v| v.is_finite()) && !arr.is_empty()"
        // Domain check: has '.' but also has is_empty → dot rule suppressed per this pre.
        // No ==, eps, weight, freqs, scale → generic path.
        // After initial sub: "arr" not replaced → has_unbound_vars.
        // Safe method check: method starts with "iter()" → true.
        // Maps arr → _contract_input.
        let pres = vec!["arr.iter().all(|v| v.is_finite()) && !arr.is_empty()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-IT", "eq_it", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_input.iter()"));
    }

    #[test]
    fn test_emit_precondition_generic_is_finite_safe_method() {
        // "val.is_finite() && !val.is_empty()"
        let pres = vec!["val.is_finite() && !val.is_empty()".to_string()];
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-IF", "eq_if", &pres);
        assert_eq!(count, 1);
        assert!(rust.contains("_contract_input.is_finite()"));
    }

    // ---------------------------------------------------------------
    // Generic path: not remapped when var has special chars
    // ---------------------------------------------------------------

    #[test]
    fn test_emit_precondition_generic_non_ident_var_not_remapped() {
        // "a+b.len() > 0 && !x.is_empty()" — the var before first dot is "a+b" which
        // contains '+', not purely alphanumeric → safe method rewrite fails.
        // But: "a+b.len()" has dot without is_empty → domain trigger!
        // Actually the "!x.is_empty()" also has is_empty, but the domain check uses .any()
        // so "a+b.len() > 0" triggers domain (has dot, no is_empty in THAT substring).
        // Wait, the check is per-precondition in pres.iter().any(|p| ...).
        // "a+b.len() > 0" has '.' and does NOT contain "is_empty" → true → domain.
        // So this is domain path. Let me construct a purely-generic case.
        //
        // For a non-remappable var in generic path, we need:
        // pres where no pre triggers domain, and a var before dot that isn't simple.
        // Impossible since any dot without is_empty triggers domain.
        // Skip this test scenario — it's unreachable in the generic path.
        // Instead test that generic path skips when safe_method doesn't match.
        let pres = vec!["buf.is_empty()".to_string()]; // dot + is_empty → no domain
        let mut rust = String::new();
        let count = emit_precondition_macro(&mut rust, "EQ-NR", "eq_nr", &pres);
        // "buf" after initial sub still unbound. safe_method check:
        // method = "is_empty()" which does NOT start with len(), iter(), is_finite()
        // → safe_method = false → not remapped → still unbound → skipped
        assert_eq!(count, 0);
    }
}
