//! Lint output rendering (text, JSON, SARIF, GitHub).
//! Split from lint.rs to stay under the 500-line limit.

use provable_contracts::lint::finding::LintFinding;
use provable_contracts::lint::rules::RuleSeverity;
use provable_contracts::lint::sarif::{findings_to_sarif, sarif_to_json};
use provable_contracts::lint::{GateDetail, LintReport};

pub fn print_text(report: &LintReport) {
    println!("pv lint — contract quality gate");
    println!("================================\n");
    for (i, gate) in report.gates.iter().enumerate() {
        print_gate(i + 1, gate);
    }
    println!();
    print_findings(report);
    print_summary(report);
}

pub fn print_gate(num: usize, gate: &provable_contracts::lint::GateResult) {
    let icon = if gate.skipped {
        "⏭"
    } else if gate.passed {
        "✅"
    } else {
        "❌"
    };
    let summary = gate_summary(&gate.detail);
    println!(
        "  Gate {num}: {:<20} {icon}  ({summary}) [{:.0}ms]",
        gate.name, gate.duration_ms
    );
}

pub fn gate_summary(detail: &GateDetail) -> String {
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
            mean_score,
            threshold,
            ..
        } => format!("{contracts} contracts, mean={mean_score:.2}, threshold={threshold:.2}"),
        GateDetail::Verify {
            total_refs,
            existing,
            missing,
        } => format!("{total_refs} refs, {existing} found, {missing} missing"),
        GateDetail::Enforce {
            equations_total,
            equations_with_pre,
            equations_with_post,
            ..
        } => format!("{equations_total} eqs, {equations_with_pre} pre, {equations_with_post} post"),
        GateDetail::ReverseCoverage {
            total_pub_fns,
            bound_fns,
            coverage_pct,
            ..
        } => format!("{bound_fns}/{total_pub_fns} bound ({coverage_pct:.1}%)"),
        GateDetail::Skipped { reason } => format!("skipped: {reason}"),
    }
}

#[allow(dead_code)]
pub fn print_gate_errors(detail: &GateDetail) {
    match detail {
        GateDetail::Validate { error_messages, .. } => {
            for msg in error_messages {
                println!("    {msg}");
            }
        }
        GateDetail::Audit {
            finding_messages, ..
        } => {
            for msg in finding_messages {
                println!("    {msg}");
            }
        }
        GateDetail::Score {
            below_threshold, ..
        } => {
            for c in below_threshold {
                println!("    {c}");
            }
        }
        _ => {}
    }
}

pub fn print_findings(report: &LintReport) {
    let active: Vec<&LintFinding> = report.findings.iter().filter(|f| !f.suppressed).collect();
    if active.is_empty() {
        return;
    }
    println!("Findings:");
    for f in &active {
        let sev = match f.severity {
            RuleSeverity::Error => "ERROR",
            RuleSeverity::Warning => "WARN",
            RuleSeverity::Info => "INFO",
            RuleSeverity::Off => "OFF",
        };
        println!("  [{sev}] {} — {}", f.rule_id, f.message);
    }
    println!();
}

pub fn print_summary(report: &LintReport) {
    let total = report.findings.len();
    let active = report.findings.iter().filter(|f| !f.suppressed).count();
    let suppressed = total - active;
    let errors = report
        .findings
        .iter()
        .filter(|f| !f.suppressed && f.severity == RuleSeverity::Error)
        .count();
    let warnings = report
        .findings
        .iter()
        .filter(|f| !f.suppressed && f.severity == RuleSeverity::Warning)
        .count();
    println!("Summary: {errors} errors, {warnings} warnings, {suppressed} suppressed");
    println!("Result: {}", if report.passed { "PASS" } else { "FAIL" });
}

pub fn print_json(report: &LintReport) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    Ok(())
}

pub fn print_sarif(report: &LintReport) {
    let sarif = findings_to_sarif(&report.findings, env!("CARGO_PKG_VERSION"));
    println!("{}", sarif_to_json(&sarif, true));
}

pub fn print_github(report: &LintReport) {
    for f in &report.findings {
        if f.suppressed {
            continue;
        }
        let level = match f.severity {
            RuleSeverity::Error => "error",
            RuleSeverity::Warning => "warning",
            _ => "notice",
        };
        println!(
            "::{level} file={},title={}::{}",
            f.file, f.rule_id, f.message
        );
    }
}
