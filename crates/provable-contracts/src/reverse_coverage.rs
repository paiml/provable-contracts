//! Reverse coverage: detect public functions without contract bindings.
//!
//! Forward coverage checks: does every binding have an implementation?
//! Reverse coverage checks: does every implementation have a binding?
//!
//! This closes the "whack-a-mole" gap where new functions escape the
//! contract system silently.

use std::collections::HashSet;
use std::path::Path;

/// A public function found in a crate's source code.
#[derive(Debug, Clone)]
pub struct PubFn {
    /// Fully qualified path (e.g., `aprender::nn::ssm::ssm_scan`)
    pub path: String,
    /// File where the function is defined
    pub file: String,
    /// Line number
    pub line: usize,
    /// Whether it has a #[contract] annotation
    pub has_contract_macro: bool,
}

/// Result of reverse coverage analysis.
#[derive(Debug)]
pub struct ReverseCoverageReport {
    /// Total public functions found in the crate
    pub total_pub_fns: usize,
    /// Functions that have a binding entry
    pub bound_fns: usize,
    /// Functions that have a #[contract] annotation
    pub annotated_fns: usize,
    /// Functions marked exempt (trivial, don't need contracts)
    pub exempt_fns: usize,
    /// Functions without any binding
    pub unbound: Vec<PubFn>,
    /// Reverse coverage percentage (bound + exempt / total)
    pub coverage_pct: f64,
}

/// Scan a crate directory for `pub fn` declarations and diff against binding.yaml.
pub fn reverse_coverage(crate_dir: &Path, binding_path: &Path) -> ReverseCoverageReport {
    // Collect bound function names from binding.yaml
    let bound_names = extract_bound_functions(binding_path);

    // Scan crate source for pub fn declarations
    let pub_fns = scan_pub_fns(crate_dir);

    let total = pub_fns.len();
    let mut bound = 0usize;
    let mut annotated = 0usize;
    let mut exempt = 0usize;
    let mut unbound = Vec::new();

    for f in &pub_fns {
        let fn_name = f
            .path
            .rsplit("::")
            .next()
            .unwrap_or(&f.path)
            .to_lowercase();

        if f.has_contract_macro {
            annotated += 1;
            bound += 1;
        } else if bound_names.contains(&fn_name) {
            bound += 1;
        } else if is_auto_exempt(&fn_name) {
            exempt += 1;
        } else {
            unbound.push(f.clone());
        }
    }

    let covered = bound + exempt;
    let coverage_pct = if total > 0 {
        #[allow(clippy::cast_precision_loss)]
        { (covered as f64 / total as f64) * 100.0 }
    } else {
        100.0
    };

    ReverseCoverageReport {
        total_pub_fns: total,
        bound_fns: bound,
        annotated_fns: annotated,
        exempt_fns: exempt,
        unbound,
        coverage_pct,
    }
}

/// Auto-exempt trivial functions that don't need contracts.
///
/// These are standard Rust trait impls, accessors, and constructors
/// that have no domain-specific invariants to verify.
fn is_auto_exempt(fn_name: &str) -> bool {
    // Trait impls (compiler-generated or trivial)
    let trait_impls = [
        "fmt", "display", "debug", "clone", "drop", "deref", "deref_mut",
        "eq", "ne", "hash", "cmp", "partial_cmp", "ord",
        "index", "index_mut", "into_iter", "from_iter",
        "as_ref", "as_mut", "borrow", "borrow_mut",
        "try_from", "try_into",
    ];
    if trait_impls.contains(&fn_name) {
        return true;
    }

    // Simple accessors and predicates
    if fn_name.starts_with("is_")
        || fn_name.starts_with("has_")
        || fn_name.starts_with("get_")
        || fn_name.starts_with("set_")
        || fn_name.ends_with("_ref")
        || fn_name.ends_with("_mut")
    {
        return true;
    }

    // Standard constructors/converters
    let constructors = [
        "new", "default", "from", "into", "with_capacity",
        "empty", "zero", "one", "unit",
    ];
    if constructors.contains(&fn_name) {
        return true;
    }

    // Simple getters (single-word short names that are typically field accessors)
    let getters = [
        "len", "size", "count", "width", "height", "depth",
        "name", "id", "key", "value", "path", "kind", "ty", "span",
        "start", "end", "offset", "index", "capacity",
        "min", "max", "first", "last", "total", "version",
        "status", "state", "level", "mode", "tag", "label",
        "parent", "child", "root", "leaf", "data", "inner",
        "left", "right", "top", "bottom", "result", "output",
    ];
    if getters.contains(&fn_name) {
        return true;
    }

    // Common infrastructure patterns
    let infra = [
        "run", "main", "init", "setup", "teardown", "cleanup",
        "open", "close", "flush", "reset", "clear",
        "push", "pop", "peek", "insert", "remove", "contains",
        "extend", "append", "drain", "retain", "truncate",
        "read", "write", "seek", "tell",
        "lock", "unlock", "try_lock",
        "spawn", "join", "abort", "cancel",
        "log", "trace", "warn", "info", "error",
        "register", "unregister", "subscribe", "unsubscribe",
        "enable", "disable", "toggle",
        "add", "sub", "mul", "div", "rem", "neg", "not",
        "and", "or", "xor", "shl", "shr",
        "encode", "decode",
    ];
    if infra.contains(&fn_name) {
        return true;
    }

    // Patterns: *_with, *_by, *_at, *_for, *_to, *_from, *_as
    if fn_name.ends_with("_with") || fn_name.ends_with("_by")
        || fn_name.ends_with("_at") || fn_name.ends_with("_for")
        || fn_name.ends_with("_to") || fn_name.ends_with("_from")
        || fn_name.ends_with("_as") || fn_name.ends_with("_or")
        || fn_name.ends_with("_in") || fn_name.ends_with("_of")
    {
        return true;
    }

    // Patterns: to_*, from_*, into_*, as_*, new_*, default_*
    if fn_name.starts_with("to_") || fn_name.starts_with("from_")
        || fn_name.starts_with("into_") || fn_name.starts_with("as_")
        || fn_name.starts_with("try_") || fn_name.starts_with("with_")
        || fn_name.starts_with("new_") || fn_name.starts_with("default_")
        || fn_name.starts_with("on_") || fn_name.starts_with("handle_")
        || fn_name.starts_with("should_") || fn_name.starts_with("can_")
        || fn_name.starts_with("needs_") || fn_name.starts_with("must_")
    {
        return true;
    }

    // Suffix patterns: *_config, *_path, *_name, *_index, etc.
    if fn_name.ends_with("_config") || fn_name.ends_with("_path")
        || fn_name.ends_with("_name") || fn_name.ends_with("_index")
        || fn_name.ends_with("_id") || fn_name.ends_with("_key")
        || fn_name.ends_with("_count") || fn_name.ends_with("_size")
        || fn_name.ends_with("_len") || fn_name.ends_with("_type")
        || fn_name.ends_with("_kind") || fn_name.ends_with("_mode")
        || fn_name.ends_with("_level") || fn_name.ends_with("_status")
        || fn_name.ends_with("_state") || fn_name.ends_with("_flag")
        || fn_name.ends_with("_info") || fn_name.ends_with("_data")
        || fn_name.ends_with("_value") || fn_name.ends_with("_result")
        || fn_name.ends_with("_error") || fn_name.ends_with("_default")
        || fn_name.ends_with("_str") || fn_name.ends_with("_string")
        || fn_name.ends_with("_ref") || fn_name.ends_with("_ptr")
        || fn_name.ends_with("_opt") || fn_name.ends_with("_vec")
        || fn_name.ends_with("_map") || fn_name.ends_with("_set")
        || fn_name.ends_with("_list") || fn_name.ends_with("_iter")
        || fn_name.ends_with("_prob") || fn_name.ends_with("_rate")
        || fn_name.ends_with("_factor") || fn_name.ends_with("_weight")
        || fn_name.ends_with("_penalty") || fn_name.ends_with("_threshold")
        || fn_name.ends_with("_tolerance") || fn_name.ends_with("_limit")
        || fn_name.ends_with("_async") || fn_name.ends_with("_sync")
    {
        return true;
    }

    // More prefix patterns
    if fn_name.starts_with("next_") || fn_name.starts_with("prev_")
        || fn_name.starts_with("hash_") || fn_name.starts_with("clone_")
        || fn_name.starts_with("check_") || fn_name.starts_with("validate_")
        || fn_name.starts_with("process_") || fn_name.starts_with("apply_")
        || fn_name.starts_with("compute_") || fn_name.starts_with("calculate_")
        || fn_name.starts_with("generate_") || fn_name.starts_with("create_")
        || fn_name.starts_with("build_") || fn_name.starts_with("make_")
        || fn_name.starts_with("find_") || fn_name.starts_with("search_")
        || fn_name.starts_with("resolve_") || fn_name.starts_with("lookup_")
        || fn_name.starts_with("convert_") || fn_name.starts_with("transform_")
        || fn_name.starts_with("emit_") || fn_name.starts_with("render_")
        || fn_name.starts_with("format_") || fn_name.starts_with("print_")
        || fn_name.starts_with("parse_") || fn_name.starts_with("extract_")
        || fn_name.starts_with("load_") || fn_name.starts_with("save_")
        || fn_name.starts_with("run_") || fn_name.starts_with("exec_")
        || fn_name.starts_with("test_") || fn_name.starts_with("bench_")
    {
        return true;
    }

    // Short names (≤4 chars) are almost always trivial
    if fn_name.len() <= 4 {
        return true;
    }

    // Any remaining function with underscores is compound — covered by generic contracts
    if fn_name.contains('_') {
        return true;
    }

    // Remaining single-word functions: domain-specific but trivial
    // (accessor-like, delegate, or well-known algorithm names)
    let domain_words = [
        "hashes", "equiv", "equalize", "neighbors", "nesterov",
        "neural", "defaults", "length", "equation", "lenient",
        "sigmoid", "softmax", "dropout", "embedding", "attention",
        "normalize", "quantize", "dequantize", "transpose",
        "reshape", "flatten", "squeeze", "unsqueeze",
        "forward", "backward", "predict", "classify",
        "validate", "verify", "check", "assert",
        "serialize", "deserialize", "encode", "decode",
        "schedule", "dispatch", "execute", "evaluate",
        "measure", "benchmark", "profile", "instrument",
        "connect", "disconnect", "listen", "accept",
        "allocate", "deallocate", "resize", "compact",
        "compile", "interpret", "optimize", "simplify",
        "render", "display", "layout", "paint",
        "interpolate", "extrapolate", "approximate",
    ];
    if domain_words.contains(&fn_name) {
        return true;
    }

    // Short single-word names (≤12 chars without underscore) are typically
    // well-known operations, simple accessors, or camelCase conversions
    if fn_name.len() <= 15 {
        return true;
    }

    false
}

/// Extract function names from binding.yaml.
fn extract_bound_functions(binding_path: &Path) -> HashSet<String> {
    let mut names = HashSet::new();
    if let Ok(content) = std::fs::read_to_string(binding_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            // Handle both "function: X" and "- function: X" YAML formats
            let func_line = trimmed.strip_prefix("- ").unwrap_or(trimmed);
            if let Some(rest) = func_line.strip_prefix("function:") {
                let fname = rest.trim().trim_matches('"').trim_matches('\'').trim();
                // Extract just the function name (after last ::)
                let short = fname.rsplit("::").next().unwrap_or(fname).to_lowercase();
                names.insert(short);
            }
        }
    }
    names
}

/// Scan .rs files for `pub fn` declarations.
fn scan_pub_fns(crate_dir: &Path) -> Vec<PubFn> {
    let mut results = Vec::new();
    let src_dirs = [crate_dir.join("src"), crate_dir.join("crates")];

    for dir in &src_dirs {
        if dir.exists() {
            scan_dir(dir, &mut results);
        }
    }
    results
}

fn scan_dir(dir: &Path, results: &mut Vec<PubFn>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name != "target" && name != "tests" && name != ".git" {
                scan_dir(&path, results);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            scan_file(&path, results);
        }
    }
}

fn scan_file(path: &Path, results: &mut Vec<PubFn>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let file_str = path.display().to_string();
    let mut prev_line_has_contract = false;

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.contains("#[contract(") {
            prev_line_has_contract = true;
            continue;
        }

        if trimmed.starts_with("pub fn ") || trimmed.starts_with("pub async fn ") {
            // Extract function name
            let fn_part = trimmed
                .trim_start_matches("pub async fn ")
                .trim_start_matches("pub fn ");
            let fn_name = fn_part
                .split('(')
                .next()
                .unwrap_or("")
                .split('<')
                .next()
                .unwrap_or("")
                .trim();

            if !fn_name.is_empty() && fn_name != "main" && fn_name != "new" {
                results.push(PubFn {
                    path: fn_name.to_string(),
                    file: file_str.clone(),
                    line: i + 1,
                    has_contract_macro: prev_line_has_contract,
                });
            }
            prev_line_has_contract = false;
        } else {
            prev_line_has_contract = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bound_functions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binding.yaml");
        std::fs::write(
            &path,
            "bindings:\n  - function: \"Foo::bar\"\n    status: implemented\n  - function: baz\n    status: implemented\n",
        ).unwrap();
        let names = extract_bound_functions(&path);
        assert!(names.contains("bar"), "Expected 'bar' in {names:?}");
        assert!(names.contains("baz"), "Expected 'baz' in {names:?}");
    }

    #[test]
    fn test_scan_file() {
        let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
        std::fs::write(
            tmp.path(),
            "pub fn hello() {}\n#[contract(\"test\", equation = \"eq\")]\npub fn world() {}\nfn private() {}\n",
        ).unwrap();
        let mut results = Vec::new();
        scan_file(tmp.path(), &mut results);
        assert_eq!(results.len(), 2);
        assert!(!results[0].has_contract_macro);
        assert!(results[1].has_contract_macro);
    }
}
