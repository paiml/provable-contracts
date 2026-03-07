//! Contract quality gate: validate + audit + score in one pass.
//!
//! Runs three sequential gates across all contracts in a directory:
//! 1. **validate** — schema completeness (SCHEMA-001..013, PROVABILITY-001)
//! 2. **audit** — traceability chain (paper→equation→obligation→test→proof)
//! 3. **score** — 5-dimension quality score vs threshold
//!
//! Extended with SARIF output, rule catalog, config file, and findings.
//! Spec: `docs/specifications/sub/lint.md`

pub mod config;
pub mod finding;
mod gates;
pub mod rules;
pub mod sarif;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use self::finding::LintFinding;
use self::gates::{load_binding, load_contracts, run_audit_gate, run_score_gate, run_validate_gate};
use self::rules::RuleSeverity;

/// Result of a single gate execution.
#[derive(Debug, Clone, Serialize)]
pub struct GateResult {
    pub name: String,
    pub passed: bool,
    pub skipped: bool,
    pub duration_ms: u64,
    pub detail: GateDetail,
}

/// Gate-specific detail payload.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum GateDetail {
    #[serde(rename = "validate")]
    Validate {
        contracts: usize,
        errors: usize,
        warnings: usize,
        error_messages: Vec<String>,
    },
    #[serde(rename = "audit")]
    Audit {
        contracts: usize,
        findings: usize,
        finding_messages: Vec<String>,
    },
    #[serde(rename = "score")]
    Score {
        contracts: usize,
        min_score: f64,
        mean_score: f64,
        threshold: f64,
        below_threshold: Vec<String>,
    },
    #[serde(rename = "skipped")]
    Skipped { reason: String },
}

/// Overall lint report.
#[derive(Debug, Clone, Serialize)]
pub struct LintReport {
    pub passed: bool,
    pub gates: Vec<GateResult>,
    pub total_duration_ms: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<LintFinding>,
}

/// Configuration for `pv lint`.
pub struct LintConfig<'a> {
    pub contract_dir: &'a Path,
    pub binding_path: Option<&'a Path>,
    pub min_score: f64,
    pub severity_filter: Option<RuleSeverity>,
    pub severity_overrides: HashMap<String, RuleSeverity>,
    pub suppressed_findings: Vec<String>,
    pub suppressed_rules: Vec<String>,
    pub suppressed_files: Vec<String>,
    pub strict: bool,
}

impl<'a> LintConfig<'a> {
    /// Create a basic config (backward compatible).
    pub fn new(contract_dir: &'a Path, binding_path: Option<&'a Path>, min_score: f64) -> Self {
        Self {
            contract_dir,
            binding_path,
            min_score,
            severity_filter: None,
            severity_overrides: HashMap::new(),
            suppressed_findings: Vec::new(),
            suppressed_rules: Vec::new(),
            suppressed_files: Vec::new(),
            strict: false,
        }
    }
}

/// Run all lint gates across a contract directory.
pub fn run_lint(config: &LintConfig) -> LintReport {
    let overall_start = Instant::now();
    let mut gates = Vec::with_capacity(3);
    let mut all_findings = Vec::new();

    let contracts = load_contracts(config.contract_dir);
    let binding = load_binding(config.binding_path);

    // Gate 1: validate
    let (validate_result, mut validate_findings) = run_validate_gate(&contracts);
    let validation_passed = validate_result.passed;
    gates.push(validate_result);

    // Gate 2: audit (skip if validation failed)
    if validation_passed {
        let (audit_result, mut audit_findings) = run_audit_gate(&contracts);
        gates.push(audit_result);
        all_findings.append(&mut audit_findings);
    } else {
        gates.push(skipped_gate("audit", "validation failed"));
    }

    // Gate 3: score (skip if validation failed)
    if validation_passed {
        let (score_result, mut score_findings) =
            run_score_gate(&contracts, binding.as_ref(), config.min_score);
        gates.push(score_result);
        all_findings.append(&mut score_findings);
    } else {
        gates.push(skipped_gate("score", "validation failed"));
    }

    all_findings.append(&mut validate_findings);

    // Apply suppressions, severity overrides, strict mode, and severity filter
    apply_suppressions(&mut all_findings, config);
    apply_severity_overrides(&mut all_findings, config);
    if let Some(min_sev) = config.severity_filter {
        all_findings.retain(|f| f.severity >= min_sev);
    }

    let passed = gates.iter().all(|g| g.passed);

    LintReport {
        passed,
        gates,
        total_duration_ms: u64::try_from(overall_start.elapsed().as_millis())
            .unwrap_or(u64::MAX),
        findings: all_findings,
    }
}

fn skipped_gate(name: &str, reason: &str) -> GateResult {
    GateResult {
        name: name.into(),
        passed: false,
        skipped: true,
        duration_ms: 0,
        detail: GateDetail::Skipped {
            reason: reason.into(),
        },
    }
}

fn apply_suppressions(findings: &mut [LintFinding], config: &LintConfig) {
    for f in findings.iter_mut() {
        if config.suppressed_rules.iter().any(|r| r == &f.rule_id) {
            f.suppressed = true;
            f.suppression_reason = Some("Suppressed by --suppress-rule".into());
        }
        if let Some(ref stem) = f.contract_stem {
            if config.suppressed_findings.iter().any(|s| s == stem) {
                f.suppressed = true;
                f.suppression_reason = Some("Suppressed by --suppress".into());
            }
        }
        if config.suppressed_files.iter().any(|p| f.file.contains(p)) {
            f.suppressed = true;
            f.suppression_reason = Some("Suppressed by --suppress-file".into());
        }
    }
}

fn apply_severity_overrides(findings: &mut [LintFinding], config: &LintConfig) {
    for f in findings.iter_mut() {
        if let Some(&sev) = config.severity_overrides.get(&f.rule_id) {
            f.severity = sev;
        }
    }
    if config.strict {
        for f in findings.iter_mut() {
            if f.severity == RuleSeverity::Warning {
                f.severity = RuleSeverity::Error;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contracts_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
    }

    #[test]
    fn lint_passes_on_real_contracts() {
        let dir = contracts_dir();
        let config = LintConfig::new(&dir, None, 0.0);
        let report = run_lint(&config);
        assert!(report.passed, "lint should pass: {report:?}");
        assert_eq!(report.gates.len(), 3);
    }

    #[test]
    fn lint_score_gate_fails_with_high_threshold() {
        let dir = contracts_dir();
        let config = LintConfig::new(&dir, None, 0.99);
        let report = run_lint(&config);
        assert!(!report.passed);
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn lint_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let config = LintConfig::new(tmp.path(), None, 0.0);
        let report = run_lint(&config);
        assert!(report.passed);
    }

    #[test]
    fn lint_report_serializes_to_json() {
        let dir = contracts_dir();
        let config = LintConfig::new(&dir, None, 0.0);
        let report = run_lint(&config);
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("\"passed\""));
    }

    #[test]
    fn gate_detail_variants() {
        let skipped = GateDetail::Skipped {
            reason: "test".into(),
        };
        let json = serde_json::to_string(&skipped).unwrap();
        assert!(json.contains("skipped"));
    }

    #[test]
    fn lint_findings_on_failure() {
        let dir = contracts_dir();
        let config = LintConfig::new(&dir, None, 0.99);
        let report = run_lint(&config);
        assert!(report.findings.iter().any(|f| f.rule_id == "PV-SCR-001"));
    }

    #[test]
    fn lint_severity_filter() {
        let dir = contracts_dir();
        let mut config = LintConfig::new(&dir, None, 0.99);
        config.severity_filter = Some(RuleSeverity::Error);
        let report = run_lint(&config);
        assert!(report.findings.iter().all(|f| f.severity >= RuleSeverity::Error));
    }

    #[test]
    fn lint_suppression_by_rule() {
        let dir = contracts_dir();
        let mut config = LintConfig::new(&dir, None, 0.99);
        config.suppressed_rules = vec!["PV-SCR-001".into()];
        let report = run_lint(&config);
        for f in &report.findings {
            if f.rule_id == "PV-SCR-001" {
                assert!(f.suppressed);
            }
        }
    }

    #[test]
    fn lint_strict_mode() {
        let dir = contracts_dir();
        let mut config = LintConfig::new(&dir, None, 0.0);
        config.strict = true;
        let report = run_lint(&config);
        for f in &report.findings {
            assert_ne!(f.severity, RuleSeverity::Warning);
        }
    }

    #[test]
    fn lint_sarif_output() {
        let dir = contracts_dir();
        let config = LintConfig::new(&dir, None, 0.99);
        let report = run_lint(&config);
        let sarif_log = sarif::findings_to_sarif(&report.findings, "0.1.0");
        let json = sarif::sarif_to_json(&sarif_log, true);
        assert!(json.contains("sarif-schema-2.1.0"));
        assert!(json.contains("PV-SCR-001"));
    }

    #[test]
    fn skipped_gate_creates_correct_result() {
        let g = skipped_gate("test", "reason");
        assert_eq!(g.name, "test");
        assert!(!g.passed);
        assert!(g.skipped);
    }
}
