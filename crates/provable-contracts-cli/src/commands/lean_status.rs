use std::path::Path;

use provable_contracts::lean_gen::{format_status_report, lean_status};
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let reports = if path.is_dir() {
        let mut reports = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(path)?
            .filter_map(Result::ok)
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "yaml" || ext == "yml")
            })
            .collect();
        entries.sort_by_key(std::fs::DirEntry::path);
        for entry in entries {
            match parse_contract(&entry.path()) {
                Ok(contract) => {
                    let report = lean_status(&contract);
                    if report.with_lean > 0 {
                        reports.push(report);
                    }
                }
                Err(e) => {
                    eprintln!("warning: skipping {}: {e}", entry.path().display());
                }
            }
        }
        reports
    } else {
        let contract = parse_contract(path)?;
        let report = lean_status(&contract);
        vec![report]
    };

    if reports.is_empty() {
        println!("No Lean proof metadata found in any contracts.");
    } else {
        print!("{}", format_status_report(&reports));
    }

    Ok(())
}
