//! Report hierarchical proof levels (L1-L5) across all contracts.
//!
//! Usage:
//!   cargo run --example `proof_status` -- contracts/

use provable_contracts::proof_status::proof_status_report;
use provable_contracts::schema::parse_contract;

fn main() {
    let dir = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: proof_status <contracts-dir/>");
            std::process::exit(1);
        },
        std::path::PathBuf::from,
    );

    let mut contracts = Vec::new();
    let entries = std::fs::read_dir(&dir).expect("cannot read dir");
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
    let report = proof_status_report(&refs, None, true);

    println!("=== Proof Status Report ===\n");
    println!("{:<40} oblg  fals  kani  lean", "Contract");
    println!("{}", "-".repeat(70));
    for entry in &report.contracts {
        println!(
            "{:<40} {:>4}  {:>4}  {:>4}  {:>4}",
            entry.stem,
            entry.obligations,
            entry.falsification_tests,
            entry.kani_harnesses,
            entry.lean_proved,
        );
    }

    println!("{}", "-".repeat(70));
    println!(
        "{:<40} {:>4}  {:>4}  {:>4}  {:>4}",
        "TOTAL",
        report.totals.obligations,
        report.totals.falsification_tests,
        report.totals.kani_harnesses,
        report.totals.lean_proved,
    );

    println!("\n=== Verification Ladder ===");
    println!("  L1: Type system (rustc)");
    println!("  L2: Falsification tests (#[test])");
    println!("  L3: Property-based tests (probar/proptest)");
    println!("  L4: Bounded model checking (Kani)");
    println!("  L5: Theorem proving (Lean 4)");

    println!("\n=== Kernel Equivalence Classes ===");
    for class in &report.kernel_classes {
        println!(
            "  Class {}: {} — {} contracts",
            class.label,
            class.description,
            class.contract_stems.len()
        );
    }

    println!(
        "\n  Contracts: {} | Obligations: {} | Kani: {} | Lean: {}",
        report.totals.contracts,
        report.totals.obligations,
        report.totals.kani_harnesses,
        report.totals.lean_proved,
    );
}
