use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::proof_status::{format_text, proof_status_report};
use provable_contracts::schema::parse_contract;

pub fn run(
    path: &Path,
    binding_path: Option<&Path>,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    // Collect contracts (single file or directory)
    let mut contracts = Vec::new();
    if path.is_dir() {
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
            let stem = entry
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            match parse_contract(&entry.path()) {
                Ok(c) => contracts.push((stem, c)),
                Err(e) => {
                    eprintln!("warning: skipping {}: {e}", entry.path().display());
                }
            }
        }
    } else {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let c = parse_contract(path)?;
        contracts.push((stem, c));
    };

    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let include_classes = contracts.len() > 1;
    let report = proof_status_report(&refs, binding.as_ref(), include_classes);

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        _ => {
            print!("{}", format_text(&report));
        }
    }

    Ok(())
}
