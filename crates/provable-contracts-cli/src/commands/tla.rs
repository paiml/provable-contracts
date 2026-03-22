use std::path::Path;

use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{parse_contract, Contract};
use provable_contracts::tla_gen::generate_tla_module;

pub fn run(
    contract_dir: &Path,
    output: Option<&Path>,
    check: bool,
    alloy: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if alloy {
        eprintln!("note: Alloy (.als) output not yet implemented — generating TLA+ instead");
    }

    let mut contracts: Vec<(String, Contract)> = Vec::new();

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

    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    let module_name = contract_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("Contracts");
    let module_name = module_name
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>();
    let module_name = if module_name.is_empty() {
        "Contracts".to_string()
    } else {
        let mut chars = module_name.chars();
        match chars.next() {
            None => "Contracts".to_string(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    };

    let tla_output = generate_tla_module(&module_name, &refs, &graph);

    if let Some(out_path) = output {
        std::fs::write(out_path, &tla_output)?;
        eprintln!("Wrote TLA+ spec to {}", out_path.display());
    } else {
        print!("{tla_output}");
    }

    if check {
        eprintln!();
        eprintln!("Running TLC model checker...");
        let status = std::process::Command::new("tlc").arg("-workers").arg("auto").status();
        match status {
            Ok(s) if s.success() => eprintln!("TLC verification: PASS"),
            Ok(s) => {
                eprintln!("TLC verification: FAIL (exit {})", s.code().unwrap_or(-1));
                return Err("TLC model checking failed".into());
            }
            Err(e) => {
                eprintln!("Could not run `tlc`: {e}");
                eprintln!("Install TLC: https://lamport.azurewebsites.net/tla/tools.html");
                return Err("tlc not found".into());
            }
        }
    }

    Ok(())
}
