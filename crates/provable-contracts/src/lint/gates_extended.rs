//! Extended gate implementations: verify, enforce, enforcement-level, level-lock.
//!
//! Split from `gates.rs` to keep file sizes under the 500-line limit.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::schema::Contract;
use crate::schema::EnforcementLevel;

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

/// Gate 6: Enforcement level — check contracts meet minimum level.
///
/// `min_level` defaults to `Standard` if not specified. A contract's actual
/// level is derived from its content: Basic (schema only), Standard
/// (has falsification + kani), Strict (all bindings implemented), Proven (Lean).
pub(crate) fn run_enforcement_level_gate(
    contracts: &[(String, Contract)],
    min_level: EnforcementLevel,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut below = 0usize;

    for (stem, contract) in contracts {
        if contract.is_registry() {
            continue;
        }
        let declared = contract
            .metadata
            .enforcement_level
            .unwrap_or(EnforcementLevel::Standard);
        let actual = compute_actual_level(contract);

        // Check: actual level must meet declared level
        if actual < declared {
            findings.push(LintFinding {
                rule_id: "PV-ENF-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Contract `{stem}` declares enforcement_level={declared:?} but only achieves {actual:?}"
                ),
                file: format!("contracts/{stem}.yaml"),
                line: None,
                contract_stem: Some(stem.clone()),
                suppressed: false,
                suppression_reason: None,
            });
            below += 1;
        }

        // Check: actual level must meet --min-level
        if actual < min_level {
            findings.push(LintFinding {
                rule_id: "PV-ENF-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Contract `{stem}` at level {actual:?}, below required {min_level:?}"
                ),
                file: format!("contracts/{stem}.yaml"),
                line: None,
                contract_stem: Some(stem.clone()),
                suppressed: false,
                suppression_reason: None,
            });
            below += 1;
        }

        // Gate 7: Level lock — cannot regress below locked_level
        if let Some(ref locked) = contract.metadata.locked_level {
            let locked_level = match locked.to_lowercase().as_str() {
                "basic" | "l1" => EnforcementLevel::Basic,
                "strict" | "l4" => EnforcementLevel::Strict,
                "proven" | "l5" => EnforcementLevel::Proven,
                // "standard", "l2", "l3", or anything unrecognized
                _ => EnforcementLevel::Standard,
            };
            if actual < locked_level {
                findings.push(LintFinding {
                    rule_id: "PV-LCK-001".into(),
                    severity: RuleSeverity::Error,
                    message: format!(
                        "Contract `{stem}` locked at {locked} but regressed to {actual:?}. Use `pv unlock` to release."
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
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "enforcement-level".into(),
            passed: !has_errors,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Skipped {
                reason: format!("{} contracts, {} below level", contracts.len(), below),
            },
        },
        findings,
    )
}

/// Derive a contract's actual enforcement level from its content.
fn compute_actual_level(contract: &Contract) -> EnforcementLevel {
    let has_falsification = !contract.falsification_tests.is_empty();
    let has_kani = !contract.kani_harnesses.is_empty();
    let has_lean = contract
        .verification_summary
        .as_ref()
        .is_some_and(|v| v.l4_lean_proved > 0 && v.l4_sorry_count == 0);

    if has_lean {
        EnforcementLevel::Proven
    } else if has_falsification && has_kani {
        EnforcementLevel::Standard
    } else {
        EnforcementLevel::Basic
    }
}

/// Detect stale suppressions: rules/findings that were suppressed but no
/// longer fire. Returns PV-SUP-001 findings for each stale suppression.
pub(crate) fn check_stale_suppressions(
    findings: &[LintFinding],
    suppressed_rules: &[String],
    suppressed_findings: &[String],
) -> Vec<LintFinding> {
    let mut stale = Vec::new();

    // Check rule suppressions: does the suppressed rule still fire?
    let active_rules: HashSet<&str> = findings.iter().map(|f| f.rule_id.as_str()).collect();
    for rule in suppressed_rules {
        if !active_rules.contains(rule.as_str()) {
            stale.push(LintFinding {
                rule_id: "PV-SUP-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Suppression for rule `{rule}` is stale — the rule no longer fires. Remove the suppression."
                ),
                file: String::new(),
                line: None,
                contract_stem: None,
                suppressed: false,
                suppression_reason: None,
            });
        }
    }

    // Check finding suppressions: does the suppressed contract still have findings?
    let active_stems: HashSet<&str> = findings
        .iter()
        .filter_map(|f| f.contract_stem.as_deref())
        .collect();
    for stem in suppressed_findings {
        if !active_stems.contains(stem.as_str()) {
            stale.push(LintFinding {
                rule_id: "PV-SUP-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Suppression for `{stem}` is stale — no findings exist for this contract. Remove the suppression."
                ),
                file: String::new(),
                line: None,
                contract_stem: Some(stem.clone()),
                suppressed: false,
                suppression_reason: None,
            });
        }
    }

    stale
}
