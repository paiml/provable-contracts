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
    /// Functions without any binding
    pub unbound: Vec<PubFn>,
    /// Reverse coverage percentage
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
        } else {
            unbound.push(f.clone());
        }
    }

    let coverage_pct = if total > 0 {
        #[allow(clippy::cast_precision_loss)]
        { (bound as f64 / total as f64) * 100.0 }
    } else {
        100.0
    };

    ReverseCoverageReport {
        total_pub_fns: total,
        bound_fns: bound,
        annotated_fns: annotated,
        unbound,
        coverage_pct,
    }
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
