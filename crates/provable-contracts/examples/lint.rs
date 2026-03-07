//! Run the contract quality gate (validate + audit + score) on a directory.
//!
//! Usage:
//!   cargo run --example lint -- contracts/
//!   cargo run --example lint -- contracts/ 0.60
//!   cargo run --example lint -- contracts/ 0.0 sarif

use std::path::PathBuf;
use std::process;

use provable_contracts::lint::sarif::{findings_to_sarif, sarif_to_json};
use provable_contracts::lint::{GateDetail, LintConfig, LintReport, run_lint};

fn main() {
    let dir = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: lint <contracts-dir/> [min-score] [format]");
            process::exit(1);
        },
        PathBuf::from,
    );

    let min_score: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let format = std::env::args().nth(3).unwrap_or_else(|| "text".into());

    let config = LintConfig::new(&dir, None, min_score);
    let report = run_lint(&config);

    match format.as_str() {
        "sarif" => print_sarif(&report),
        "json" => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        "github" => print_github(&report),
        _ => print_text(&report),
    }

    if !report.passed {
        process::exit(1);
    }
}

fn print_sarif(report: &LintReport) {
    let sarif = findings_to_sarif(&report.findings, env!("CARGO_PKG_VERSION"));
    println!("{}", sarif_to_json(&sarif, true));
}

fn print_github(report: &LintReport) {
    for f in &report.findings {
        if !f.suppressed {
            println!("{}", f.to_github_annotation());
        }
    }
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
    println!(
        "Gate {num}: {:.<30} {status} ({summary}) [{}ms]",
        gate.name, gate.duration_ms
    );
}

fn gate_summary(detail: &GateDetail) -> String {
    match detail {
        GateDetail::Validate { contracts, errors, warnings, .. } => {
            format!("{contracts} contracts, {errors} errors, {warnings} warnings")
        }
        GateDetail::Audit { contracts, findings, .. } => {
            format!("{contracts} contracts, {findings} findings")
        }
        GateDetail::Score { contracts, min_score, mean_score, threshold, .. } => {
            format!("{contracts} contracts, min={min_score:.2}, mean={mean_score:.2}, threshold={threshold:.2}")
        }
        GateDetail::Skipped { reason } => reason.clone(),
    }
}

fn print_findings(report: &LintReport) {
    if report.findings.is_empty() {
        return;
    }
    let unsuppressed = report.findings.iter().filter(|f| !f.suppressed).count();
    let suppressed = report.findings.len() - unsuppressed;
    println!("\nFindings: {} total ({suppressed} suppressed)", report.findings.len());
    for f in report.findings.iter().filter(|f| !f.suppressed) {
        println!("  {f}");
    }
}

fn print_summary(report: &LintReport) {
    let passed = report.gates.iter().filter(|g| g.passed).count();
    let total = report.gates.len();
    let result = if report.passed { "PASS" } else { "FAIL" };
    println!("\nResult: {result} ({passed}/{total} gates passed) [{}ms]", report.total_duration_ms);
}
