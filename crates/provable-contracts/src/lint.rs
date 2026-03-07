//! Contract quality gate: validate + audit + score in one pass.
//!
//! Runs three sequential gates across all contracts in a directory:
//! 1. **validate** — schema completeness (SCHEMA-001..013, PROVABILITY-001)
//! 2. **audit** — traceability chain (paper→equation→obligation→test→proof)
//! 3. **score** — 5-dimension quality score vs threshold
//!
//! Spec: `docs/specifications/sub/scoring.md` Section 5

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::audit::audit_contract;
use crate::binding::{parse_binding, BindingRegistry};
use crate::error::Severity;
use crate::schema::{parse_contract, validate_contract, Contract};
use crate::scoring::{score_contract, ContractScore};

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
}

/// Configuration for `pv lint`.
pub struct LintConfig<'a> {
    pub contract_dir: &'a Path,
    pub binding_path: Option<&'a Path>,
    pub min_score: f64,
}

/// Run all lint gates across a contract directory.
#[allow(clippy::cast_precision_loss)]
pub fn run_lint(config: &LintConfig) -> LintReport {
    let overall_start = Instant::now();
    let mut gates = Vec::with_capacity(3);

    // Load all contracts
    let contracts = load_contracts(config.contract_dir);
    let binding = config.binding_path.and_then(|p| parse_binding(p).ok());

    // Gate 1: validate
    let validate_result = run_validate_gate(&contracts);
    let validation_passed = validate_result.passed;
    gates.push(validate_result);

    // Gate 2: audit (skip if validation failed)
    if validation_passed {
        gates.push(run_audit_gate(&contracts));
    } else {
        gates.push(GateResult {
            name: "audit".into(),
            passed: false,
            skipped: true,
            duration_ms: 0,
            detail: GateDetail::Skipped {
                reason: "validation failed".into(),
            },
        });
    }

    // Gate 3: score (skip if validation failed)
    if validation_passed {
        gates.push(run_score_gate(
            &contracts,
            binding.as_ref(),
            config.min_score,
        ));
    } else {
        gates.push(GateResult {
            name: "score".into(),
            passed: false,
            skipped: true,
            duration_ms: 0,
            detail: GateDetail::Skipped {
                reason: "validation failed".into(),
            },
        });
    }

    let passed = gates.iter().all(|g| g.passed);

    LintReport {
        passed,
        gates,
        total_duration_ms: u64::try_from(overall_start.elapsed().as_millis()).unwrap_or(u64::MAX),
    }
}

/// Load and parse all YAML contracts from a directory.
fn load_contracts(dir: &Path) -> Vec<(String, Contract)> {
    let mut contracts = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return contracts;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("yaml") {
            continue;
        }
        if path.is_dir() {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        match parse_contract(&path) {
            Ok(c) => contracts.push((stem, c)),
            Err(_) => continue,
        }
    }
    contracts.sort_by(|a, b| a.0.cmp(&b.0));
    contracts
}

fn run_validate_gate(contracts: &[(String, Contract)]) -> GateResult {
    let start = Instant::now();
    let mut total_errors = 0usize;
    let mut total_warnings = 0usize;
    let mut error_messages = Vec::new();

    for (stem, contract) in contracts {
        let violations = validate_contract(contract);
        for v in &violations {
            match v.severity {
                Severity::Error => {
                    total_errors += 1;
                    error_messages.push(format!("{v} ({stem})"));
                }
                Severity::Warning => total_warnings += 1,
                Severity::Info => {}
            }
        }
    }

    GateResult {
        name: "validate".into(),
        passed: total_errors == 0,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: contracts.len(),
            errors: total_errors,
            warnings: total_warnings,
            error_messages,
        },
    }
}

fn run_audit_gate(contracts: &[(String, Contract)]) -> GateResult {
    let start = Instant::now();
    let mut total_findings = 0usize;
    let mut finding_messages = Vec::new();

    for (stem, contract) in contracts {
        let report = audit_contract(contract);
        for v in &report.violations {
            if v.severity == Severity::Error {
                total_findings += 1;
                finding_messages.push(format!("{v} ({stem})"));
            }
        }
    }

    GateResult {
        name: "audit".into(),
        passed: total_findings == 0,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Audit {
            contracts: contracts.len(),
            findings: total_findings,
            finding_messages,
        },
    }
}

#[allow(clippy::cast_precision_loss)]
fn run_score_gate(
    contracts: &[(String, Contract)],
    binding: Option<&BindingRegistry>,
    threshold: f64,
) -> GateResult {
    let start = Instant::now();
    let mut scores: Vec<ContractScore> = Vec::new();
    let mut below_threshold = Vec::new();

    for (stem, contract) in contracts {
        let s = score_contract(contract, binding, stem);
        if s.composite < threshold {
            below_threshold.push(format!(
                "{} — {:.2} (Grade {}, threshold {:.2})",
                stem, s.composite, s.grade, threshold
            ));
        }
        scores.push(s);
    }

    let min_score = scores
        .iter()
        .map(|s| s.composite)
        .fold(f64::INFINITY, f64::min);
    let mean_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64
    };

    // If threshold is 0.0, score gate always passes (no minimum enforced)
    let passed = below_threshold.is_empty();

    GateResult {
        name: "score".into(),
        passed,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Score {
            contracts: contracts.len(),
            min_score: if scores.is_empty() { 0.0 } else { min_score },
            mean_score,
            threshold,
            below_threshold,
        },
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
        let config = LintConfig {
            contract_dir: &contracts_dir(),
            binding_path: None,
            min_score: 0.0,
        };
        let report = run_lint(&config);
        assert!(report.passed, "lint should pass: {report:?}");
        assert_eq!(report.gates.len(), 3);
        assert!(report.gates[0].passed, "validate gate");
        assert!(report.gates[1].passed, "audit gate");
        assert!(report.gates[2].passed, "score gate");
    }

    #[test]
    fn lint_score_gate_fails_with_high_threshold() {
        let config = LintConfig {
            contract_dir: &contracts_dir(),
            binding_path: None,
            min_score: 0.99,
        };
        let report = run_lint(&config);
        assert!(!report.passed);
        assert!(!report.gates[2].passed, "score gate should fail at 0.99");
    }

    #[test]
    fn lint_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let config = LintConfig {
            contract_dir: tmp.path(),
            binding_path: None,
            min_score: 0.0,
        };
        let report = run_lint(&config);
        // Empty dir = 0 contracts, all gates pass vacuously
        assert!(report.passed);
    }

    #[test]
    fn lint_report_serializes_to_json() {
        let config = LintConfig {
            contract_dir: &contracts_dir(),
            binding_path: None,
            min_score: 0.0,
        };
        let report = run_lint(&config);
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("\"passed\""));
        assert!(json.contains("\"validate\""));
        assert!(json.contains("\"audit\""));
        assert!(json.contains("\"score\""));
    }

    #[test]
    fn gate_detail_variants() {
        let skipped = GateDetail::Skipped {
            reason: "test".into(),
        };
        let json = serde_json::to_string(&skipped).unwrap();
        assert!(json.contains("skipped"));
    }
}
