//! Cross-contract obligation coverage report.
//!
//! Usage:
//!   cargo run --example coverage -- contracts/
//!   cargo run --example coverage -- contracts/ contracts/aprender/binding.yaml

use std::path::Path;

use provable_contracts::coverage::{coverage_report, overall_percentage};
use provable_contracts::schema::parse_contract;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map_or_else(
        || {
            eprintln!("Usage: coverage <contracts-dir/> [binding.yaml]");
            std::process::exit(1);
        },
        Path::new,
    );

    let binding = args
        .get(2)
        .and_then(|bp| provable_contracts::binding::parse_binding(Path::new(bp)).ok());

    let mut contracts = Vec::new();
    let entries = std::fs::read_dir(dir).expect("cannot read dir");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            if let Ok(c) = parse_contract(&path) {
                contracts.push((stem, c));
            }
        }
    }
    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let report = coverage_report(&refs, binding.as_ref());
    let pct = overall_percentage(&report);

    println!("=== Obligation Coverage Report ===\n");
    println!("{:<40} eq  ob  ft  kani  impl", "Contract");
    println!("{}", "-".repeat(70));
    for cc in &report.contracts {
        println!(
            "{:<40} {:>3} {:>3} {:>3} {:>4}  {}/{}",
            cc.stem,
            cc.equations,
            cc.obligations,
            cc.falsification_covered,
            cc.kani_covered,
            cc.binding_implemented,
            cc.equations,
        );
    }

    println!("\n=== Totals ===");
    println!("  Contracts:            {}", report.totals.contracts);
    println!("  Equations:            {}", report.totals.equations);
    println!("  Obligations:          {}", report.totals.obligations);
    println!(
        "  Falsification tests:  {}",
        report.totals.falsification_tests
    );
    println!("  Kani harnesses:       {}", report.totals.kani_harnesses);
    if binding.is_some() {
        println!(
            "  Binding implemented:  {}",
            report.totals.binding_implemented
        );
        println!("  Binding partial:      {}", report.totals.binding_partial);
        println!("  Binding missing:      {}", report.totals.binding_missing);
    }
    println!("\n  Overall obligation coverage: {pct:.1}%");

    // Gap analysis
    let gaps: Vec<_> = report
        .contracts
        .iter()
        .filter(|c| c.obligations > 0 && c.kani_covered == 0)
        .collect();
    if !gaps.is_empty() {
        println!("\n=== Contracts with obligations but no Kani harnesses ===");
        for g in &gaps {
            println!("  {} ({} obligations)", g.stem, g.obligations);
        }
    }
}
