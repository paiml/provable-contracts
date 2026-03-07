use std::path::Path;

use provable_contracts::lint::config::{find_config, load_config};
use provable_contracts::lint::finding::LintFinding;
use provable_contracts::lint::rules::RuleSeverity;
use provable_contracts::lint::sarif::{findings_to_sarif, sarif_to_json};
use provable_contracts::lint::trend;
use provable_contracts::lint::{GateDetail, LintConfig, LintReport, run_lint};

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
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
    diff_ref: Option<&str>,
    do_trend: bool,
    show_trend: bool,
    _no_cache: bool,
    _cache_stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Show trend history (no lint run needed)
    if show_trend {
        let trend_root = trend::trend_dir(contract_dir);
        let snapshots = trend::load_snapshots(&trend_root);
        if snapshots.is_empty() {
            println!("No trend data. Run `pv lint --trend` to record snapshots.");
        } else {
            println!("{}", trend::format_trend(&snapshots, 30));
        }
        return Ok(());
    }

    // Diff-aware: report changed contracts
    if let Some(base) = diff_ref {
        match provable_contracts::lint::diff::changed_contracts(contract_dir, base) {
            Ok(changed) if changed.is_empty() => {
                println!("No contracts changed since {base}. Nothing to lint.");
                return Ok(());
            }
            Ok(changed) => {
                println!(
                    "Diff-aware: {} contracts changed since {base}",
                    changed.len()
                );
                for stem in &changed {
                    println!("  {stem}");
                }
                println!();
            }
            Err(e) => {
                eprintln!("Warning: diff-aware mode failed ({e}), linting all contracts");
            }
        }
    }

    let config = build_config(
        contract_dir,
        binding_path,
        min_score,
        format,
        severity,
        strict,
        suppress,
        suppress_rule,
        suppress_file,
        rule_overrides,
        config_path,
    );

    let report = run_lint(&config);

    // Record trend snapshot
    if do_trend {
        let trend_root = trend::trend_dir(contract_dir);
        let contracts_count = count_contracts(&report);
        match trend::record_snapshot(&trend_root, &report, contracts_count) {
            Ok(path) => eprintln!("Trend snapshot saved: {}", path.display()),
            Err(e) => eprintln!("Warning: failed to save trend snapshot: {e}"),
        }

        // Check for drift
        let snapshots = trend::load_snapshots(&trend_root);
        if let Some(drop) = trend::detect_drift(&snapshots, 0.05) {
            eprintln!("Warning: quality drift detected (score dropped {drop:.3})");
        }
    }

    let effective_format = resolve_format(format, config_path, contract_dir);
    match effective_format.as_str() {
        "json" => print_json(&report)?,
        "sarif" => print_sarif(&report),
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

fn count_contracts(report: &LintReport) -> usize {
    for gate in &report.gates {
        match &gate.detail {
            GateDetail::Validate { contracts, .. }
            | GateDetail::Audit { contracts, .. }
            | GateDetail::Score { contracts, .. } => return *contracts,
            GateDetail::Skipped { .. } => {}
        }
    }
    0
}

fn resolve_format(format: &str, config_path: Option<&Path>, contract_dir: &Path) -> String {
    if format != "text" {
        return format.to_string();
    }
    let pv_config = config_path
        .and_then(|cp| load_config(cp).ok())
        .or_else(|| find_config(contract_dir).and_then(|p| load_config(&p).ok()))
        .unwrap_or_default();
    pv_config
        .output
        .format
        .unwrap_or_else(|| "text".into())
}

#[allow(clippy::too_many_arguments)]
fn build_config<'a>(
    contract_dir: &'a Path,
    binding_path: Option<&'a Path>,
    min_score: f64,
    _format: &str,
    severity: Option<&str>,
    strict: bool,
    suppress: Option<&str>,
    suppress_rule: Option<&str>,
    suppress_file: Option<&str>,
    rule_overrides: &[String],
    config_path: Option<&Path>,
) -> LintConfig<'a> {
    let pv_config = config_path
        .and_then(|cp| load_config(cp).ok())
        .or_else(|| find_config(contract_dir).and_then(|p| load_config(&p).ok()))
        .unwrap_or_default();

    let effective_min_score = if min_score > 0.0 {
        min_score
    } else {
        pv_config.lint.min_score.unwrap_or(0.0)
    };

    let severity_filter = severity
        .or(pv_config.lint.severity.as_deref())
        .and_then(RuleSeverity::from_str_opt);

    let effective_strict = strict || pv_config.lint.strict;

    let suppressed_findings: Vec<String> = parse_csv(suppress)
        .into_iter()
        .chain(pv_config.lint.suppress.findings.iter().cloned())
        .collect();
    let suppressed_rules: Vec<String> = parse_csv(suppress_rule)
        .into_iter()
        .chain(pv_config.lint.suppress.rules.iter().cloned())
        .collect();
    let suppressed_files: Vec<String> = parse_csv(suppress_file)
        .into_iter()
        .chain(pv_config.lint.suppress.files.iter().cloned())
        .collect();

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

    LintConfig {
        contract_dir,
        binding_path,
        min_score: effective_min_score,
        severity_filter,
        severity_overrides,
        suppressed_findings,
        suppressed_rules,
        suppressed_files,
        strict: effective_strict,
    }
}

fn parse_csv(s: Option<&str>) -> Vec<String> {
    s.map(|v| {
        v.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    })
    .unwrap_or_default()
}

fn print_text(report: &LintReport) {
    println!("pv lint — contract quality gate");
    println!("================================\n");
    for (i, gate) in report.gates.iter().enumerate() {
        print_gate(i + 1, gate);
    }
    print_findings(report);
    print_summary(report);
}

fn print_gate(num: usize, gate: &provable_contracts::lint::GateResult) {
    let status = if gate.skipped {
        "SKIP"
    } else if gate.passed {
        "PASS"
    } else {
        "FAIL"
    };
    let summary = gate_summary(&gate.detail);
    let dots = 30usize.saturating_sub(gate.name.len());
    let dot_str = ".".repeat(dots);
    println!(
        "Gate {num}: {} {dot_str} {status} ({summary}) [{}ms]",
        gate.name, gate.duration_ms
    );
    if !gate.passed && !gate.skipped {
        print_gate_errors(&gate.detail);
    }
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
    let messages: &[String] = match detail {
        GateDetail::Validate { error_messages, .. } => error_messages,
        GateDetail::Audit {
            finding_messages, ..
        } => finding_messages,
        GateDetail::Score {
            below_threshold, ..
        } => below_threshold,
        GateDetail::Skipped { .. } => return,
    };
    for msg in messages.iter().take(10) {
        println!("  {msg}");
    }
    let remaining = messages.len().saturating_sub(10);
    if remaining > 0 {
        println!("  ... and {remaining} more");
    }
}

fn print_findings(report: &LintReport) {
    if report.findings.is_empty() {
        return;
    }
    let unsuppressed: Vec<&LintFinding> =
        report.findings.iter().filter(|f| !f.suppressed).collect();
    let suppressed_count = report.findings.len() - unsuppressed.len();
    println!(
        "\nFindings: {} total ({} suppressed)",
        report.findings.len(),
        suppressed_count
    );
    for f in &unsuppressed {
        println!("  {f}");
    }
}

fn print_summary(report: &LintReport) {
    let passed_count = report.gates.iter().filter(|g| g.passed).count();
    let total = report.gates.len();
    let result = if report.passed { "PASS" } else { "FAIL" };
    println!(
        "\nResult: {result} ({passed_count}/{total} gates passed) [{}ms]",
        report.total_duration_ms
    );
}

fn print_json(report: &LintReport) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

fn print_sarif(report: &LintReport) {
    let version = env!("CARGO_PKG_VERSION");
    let sarif = findings_to_sarif(&report.findings, version);
    println!("{}", sarif_to_json(&sarif, true));
}

fn print_github(report: &LintReport) {
    for finding in &report.findings {
        if !finding.suppressed {
            println!("{}", finding.to_github_annotation());
        }
    }
}
