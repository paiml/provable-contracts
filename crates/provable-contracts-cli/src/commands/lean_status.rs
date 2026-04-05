use std::path::Path;

use provable_contracts::lean_gen::{format_status_report, lean_status};
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let reports = if path.is_dir() {
        let mut reports = Vec::new();
        walk_contracts(path, &mut reports);
        reports
    } else {
        let contract = parse_contract(path)?;
        vec![lean_status(&contract)]
    };

    if reports.is_empty() {
        println!("No Lean proof metadata found in any contracts.");
    } else {
        print!("{}", format_status_report(&reports));
    }

    Ok(())
}

fn walk_contracts(dir: &Path, out: &mut Vec<provable_contracts::lean_gen::LeanStatusReport>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk_contracts(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if stem == "binding" || stem == "playbook.schema" || stem.contains("playbook") {
                continue;
            }
            if let Ok(contract) = parse_contract(&path) {
                let report = lean_status(&contract);
                if report.with_lean > 0 {
                    out.push(report);
                }
            }
        }
    }
}
