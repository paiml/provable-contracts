use std::path::Path;

use provable_contracts::lint::config::{find_config, load_config};
use provable_contracts::lint::finding::LintFinding;
use provable_contracts::lint::rules::RuleSeverity;
use provable_contracts::lint::sarif::{findings_to_sarif, sarif_to_json};
use provable_contracts::lint::{GateDetail, LintConfig, LintReport, run_lint};

#[allow(clippy::too_many_arguments)]
pub fn run(
    contract_dir: &Path,
    binding_path: Option<&Path>,
    min_score: f64,
    format: &str,
    severity: Option<&str>,
    strict: bool,
    suppress: Option<&str>,
    suppress_rule: Option<&str>,
    suppress_file: Option<&str>,
    rule_overrides: &[String],
    config_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load config file
    let pv_config = if let Some(cp) = config_path {
        load_config(cp).unwrap_or_default()
    } else {
        find_config(contract_dir)
            .and_then(|p| load_config(&p).ok())
            .unwrap_or_default()
    };

    // Merge CLI flags with config
    let effective_min_score = if min_score > 0.0 {
        min_score
    } else {
        pv_config.lint.min_score.unwrap_or(0.0)
    };

    let effective_format = if format != "text" {
        format.to_string()
    } else {
        pv_config.output.format.clone().unwrap_or_else(|| "text".into())
    };

    let severity_filter = severity
        .or(pv_config.lint.severity.as_deref())
        .and_then(RuleSeverity::from_str_opt);

    let effective_strict = strict || pv_config.lint.strict;

    let effective_binding = binding_path.map(|p| p.to_path_buf()).or_else(|| {
        pv_config.lint.binding.as_ref().map(std::path::PathBuf::from)
    });

    // Parse suppressions
    let suppressed_findings = parse_csv(suppress)
        .into_iter()
        .chain(pv_config.lint.suppress.findings.iter().cloned())
        .collect();
    let suppressed_rules = parse_csv(suppress_rule)
        .into_iter()
        .chain(pv_config.lint.suppress.rules.iter().cloned())
        .collect();
    let suppressed_files = parse_csv(suppress_file)
        .into_iter()
        .chain(pv_config.lint.suppress.files.iter().cloned())
        .collect();

    // Parse rule severity overrides
    let mut severity_overrides = std::collections::HashMap::new();
    for entry in &pv_config.lint.rules {
        if let Some(sev) = RuleSeverity::from_str_opt(entry.1) {
            severity_overrides.insert(entry.0.clone(), sev);
        }
    }
    for r in rule_overrides {
        if let Some((id, sev_str)) = r.split_once('=') {
            if let Some(sev) = RuleSeverity::from_str_opt(sev_str) {
                severity_overrides.insert(id.to_string(), sev);
            }
        }
    }

    let config = LintConfig {
        contract_dir,
        binding_path: effective_binding.as_deref(),
        min_score: effective_min_score,
        severity_filter,
        severity_overrides,
        suppressed_findings,
        suppressed_rules,
        suppressed_files,
        strict: effective_strict,
    };

    let report = run_lint(&config);

    match effective_format.as_str() {
        "json" => print_json(&report)?,
        "sarif" => print_sarif(&report)?,
        "github" => print_github(&report),
        _ => print_text(&report),
    }

    if report.passed {
        Ok(())
    } else {
        let passed_count = report.gates.iter().filter(|g| g.passed).count();
        Err(format!(
            "lint failed ({}/{} gates passed)",
            passed_count,
            report.gates.len()
        )
        .into())
    }
}

fn parse_csv(s: Option<&str>) -> Vec<String> {
    s.map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default()
}

fn gate_summary(detail: &GateDetail) -> String {
    match detail {
        GateDetail::Validate {
            contracts,
            errors,
            warnings,
            ..
        } => format!("{contracts} contracts, {errors} errors, {warnings} warnings"),
        GateDetail::Audit {
            contracts,
            findings,
            ..
        } => format!("{contracts} contracts, {findings} findings"),
        GateDetail::Score {
            contracts,
            min_score,
            mean_score,
            threshold,
            ..
        } => format!(
            "{contracts} contracts, min={min_score:.2}, mean={mean_score:.2}, threshold={threshold:.2}"
        ),
        GateDetail::Skipped { reason } => reason.clone(),
    }
}

fn print_gate_errors(detail: &GateDetail) {
    match detail {
        GateDetail::Validate { error_messages, .. } => {
            for msg in error_messages {
                println!("  {msg}");
            }
        }
        GateDetail::Audit {
            finding_messages, ..
        } => {
            for msg in finding_messages {
                println!("  {msg}");
            }
        }
        GateDetail::Score {
            below_threshold, ..
        } => {
            for msg in below_threshold.iter().take(10) {
                println!("  {msg}");
            }
            let remaining = below_threshold.len().saturating_sub(10);
            if remaining > 0 {
                println!("  ... and {remaining} more");
            }
        }
        GateDetail::Skipped { .. } => {}
    }
}

fn print_text(report: &LintReport) {
    println!("pv lint — contract quality gate");
    println!("================================");
    println!();

    for (i, gate) in report.gates.iter().enumerate() {
        let num = i + 1;
        let status = if gate.skipped {
            "SKIP"
        } else if gate.passed {
            "PASS"
        } else {
            "FAIL"
        };

        let detail_summary = gate_summary(&gate.detail);
        let dots = 30usize.saturating_sub(gate.name.len());
        let dot_str = ".".repeat(dots);
        println!(
            "Gate {num}: {} {dot_str} {status} ({detail_summary}) [{}ms]",
            gate.name, gate.duration_ms
        );

        if !gate.passed && !gate.skipped {
            print_gate_errors(&gate.detail);
        }
    }

    // Print findings summary
    if !report.findings.is_empty() {
        let unsuppressed: Vec<&LintFinding> =
            report.findings.iter().filter(|f| !f.suppressed).collect();
        let suppressed_count = report.findings.len() - unsuppressed.len();
        println!();
        println!("Findings: {} total ({} suppressed)", report.findings.len(), suppressed_count);
        for f in &unsuppressed {
            println!("  {f}");
        }
    }

    let passed_count = report.gates.iter().filter(|g| g.passed).count();
    let total = report.gates.len();
    let result = if report.passed { "PASS" } else { "FAIL" };
    println!();
    println!(
        "Result: {result} ({passed_count}/{total} gates passed) [{}ms]",
        report.total_duration_ms
    );
}

fn print_json(report: &LintReport) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

fn print_sarif(report: &LintReport) -> Result<(), Box<dyn std::error::Error>> {
    let version = env!("CARGO_PKG_VERSION");
    let sarif = findings_to_sarif(&report.findings, version);
    println!("{}", sarif_to_json(&sarif, true));
    Ok(())
}

fn print_github(report: &LintReport) {
    for finding in &report.findings {
        if !finding.suppressed {
            println!("{}", finding.to_github_annotation());
        }
    }
}
