use std::path::Path;

use provable_contracts::lint::{run_lint, GateDetail, LintConfig, LintReport};

pub fn run(
    contract_dir: &Path,
    binding_path: Option<&Path>,
    min_score: f64,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = LintConfig {
        contract_dir,
        binding_path,
        min_score,
    };

    let report = run_lint(&config);

    match format {
        "json" => print_json(&report)?,
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
