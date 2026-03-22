//! Extended gate implementations: verify, enforce.
//!
//! Split from `gates.rs` to keep file sizes under the 500-line limit.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::schema::Contract;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateResult};

/// Gate 4: Source verification — do referenced test functions exist in source?
///
/// Resolves `test:` fields in `falsification_tests` against `fn test_*` / `fn prop_*`
/// in the project's `src/` directory. Any missing test = ERROR (unfalsifiable claim).
pub(crate) fn run_verify_gate(
    contracts: &[(String, Contract)],
    project_root: &Path,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let src_dir = project_root.join("src");
    let mut findings = Vec::new();
    let mut total_refs = 0usize;
    let mut missing = 0usize;

    // Build set of all test function names in src/
    let mut source_tests = HashSet::new();
    if src_dir.exists() {
        collect_test_fns(&src_dir, &mut source_tests);
    }

    for (stem, contract) in contracts {
        for ft in &contract.falsification_tests {
            if let Some(ref test_name) = ft.test {
                let raw = test_name.trim().trim_matches('"');
                // Extract function name from module path (e.g., "mod::tests::test_foo" → "test_foo")
                let name = raw.rsplit("::").next().unwrap_or(raw);
                if name.starts_with("test_") || name.starts_with("prop_") {
                    total_refs += 1;
                    if !source_tests.contains(name) {
                        missing += 1;
                        findings.push(LintFinding {
                            rule_id: "PV-VER-001".into(),
                            severity: RuleSeverity::Error,
                            message: format!(
                                "Unfalsifiable: test `{name}` referenced but not found in src/"
                            ),
                            file: format!("contracts/{stem}.yaml"),
                            line: None,
                            contract_stem: Some(stem.clone()),
                            suppressed: false,
                            suppression_reason: None,
                        });
                    }
                }
            }
        }
    }

    let passed = missing == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "verify".into(),
            passed,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Verify {
                total_refs,
                existing: total_refs - missing,
                missing,
            },
        },
        findings,
    )
}

fn collect_test_fns(dir: &Path, tests: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_test_fns(&path, tests);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    if let Some(pos) = line.find("fn test_").or_else(|| line.find("fn prop_")) {
                        let rest = &line[pos + 3..];
                        let name: String = rest
                            .chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect();
                        if !name.is_empty() {
                            tests.insert(name);
                        }
                    }
                }
            }
        }
    }
}

/// Gate 5: Enforce — equations MUST have preconditions, postconditions, and `lean_theorem`.
/// An equation without enforceable contracts is an unverifiable claim.
pub(crate) fn run_enforce_gate(contracts: &[(String, Contract)]) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut total_eqs = 0usize;
    let mut with_pre = 0usize;
    let mut with_post = 0usize;
    let mut with_lean = 0usize;

    for (stem, contract) in contracts {
        if contract.is_registry() {
            continue;
        }
        for (eq_name, eq) in &contract.equations {
            total_eqs += 1;
            if eq.preconditions.is_empty() {
                findings.push(LintFinding {
                    rule_id: "PV-ENF-001".into(),
                    severity: RuleSeverity::Warning,
                    message: format!("Equation `{eq_name}` has no preconditions"),
                    file: format!("contracts/{stem}.yaml"),
                    line: None,
                    contract_stem: Some(stem.clone()),
                    suppressed: false,
                    suppression_reason: None,
                });
            } else {
                with_pre += 1;
            }
            if !eq.postconditions.is_empty() {
                with_post += 1;
            }
            if eq.lean_theorem.is_some() {
                with_lean += 1;
            } else {
                findings.push(LintFinding {
                    rule_id: "PV-ENF-002".into(),
                    severity: RuleSeverity::Warning,
                    message: format!(
                        "Equation `{eq_name}` has no lean_theorem — proof recommended"
                    ),
                    file: format!("contracts/{stem}.yaml"),
                    line: None,
                    contract_stem: Some(stem.clone()),
                    suppressed: false,
                    suppression_reason: None,
                });
            }
        }
    }

    let has_errors = findings.iter().any(|f| f.severity == RuleSeverity::Error);
    let passed = !has_errors || total_eqs == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "enforce".into(),
            passed,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Enforce {
                equations_total: total_eqs,
                equations_with_pre: with_pre,
                equations_with_post: with_post,
                equations_with_lean: with_lean,
            },
        },
        findings,
    )
}
