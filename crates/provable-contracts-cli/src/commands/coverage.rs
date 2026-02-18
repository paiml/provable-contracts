use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::coverage::{coverage_report, overall_percentage};
use provable_contracts::schema::parse_contract;

pub fn run(
    contract_dir: &Path,
    binding_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    // Collect all .yaml contracts from the directory
    let mut contracts = Vec::new();
    let entries = std::fs::read_dir(contract_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            match parse_contract(&path) {
                Ok(c) => contracts.push((stem, c)),
                Err(e) => {
                    eprintln!("warning: skipping {}: {e}", path.display());
                }
            }
        }
    }

    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let report = coverage_report(&refs, binding.as_ref());
    let pct = overall_percentage(&report);

    println!("Obligation Coverage Report");
    println!("==========================");
    println!();

    for cc in &report.contracts {
        println!(
            "  {:<35} eq={} ob={} ft={} kani={} impl={}/{}",
            cc.stem,
            cc.equations,
            cc.obligations,
            cc.falsification_covered,
            cc.kani_covered,
            cc.binding_implemented,
            cc.equations,
        );
    }

    println!();
    println!("Totals:");
    println!("  Contracts:            {}", report.totals.contracts);
    println!("  Equations:            {}", report.totals.equations);
    println!("  Obligations:          {}", report.totals.obligations);
    println!(
        "  Falsification tests:  {}",
        report.totals.falsification_tests
    );
    println!("  Kani harnesses:       {}", report.totals.kani_harnesses);
    if binding_path.is_some() {
        println!(
            "  Binding implemented:  {}",
            report.totals.binding_implemented
        );
        println!("  Binding partial:      {}", report.totals.binding_partial);
        println!("  Binding missing:      {}", report.totals.binding_missing);
    }
    println!();
    println!("Overall obligation coverage: {pct:.1}%");

    Ok(())
}
