use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::obligation_matrix::{format_obligation_table, obligation_matrix};
use provable_contracts::proof_status::{format_text, proof_status_report};
use provable_contracts::schema::{Contract, parse_contract};

pub fn run(
    path: &Path,
    binding_path: Option<&Path>,
    format: &str,
    table: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    // Collect contracts (single file or directory tree)
    let mut contracts = Vec::new();
    if path.is_dir() {
        collect_contracts_recursive(path, &mut contracts);
    } else {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let c = parse_contract(path)?;
        contracts.push((stem, c));
    }

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

    if table {
        let matrices = obligation_matrix(&refs);
        print!("{}", format_obligation_table(&matrices));
    }

    Ok(())
}

/// Walk `dir` recursively and collect parseable `.yaml` contracts.
///
/// Skips binding/playbook sidecar files (matching `pv verify-pipeline`
/// behavior) and silently drops unparseable entries.
fn collect_contracts_recursive(dir: &Path, out: &mut Vec<(String, Contract)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_contracts_recursive(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            if stem == "binding" || stem == "playbook.schema" || stem.contains("playbook") {
                continue;
            }
            if let Ok(c) = parse_contract(&path) {
                out.push((stem, c));
            }
        }
    }
}
