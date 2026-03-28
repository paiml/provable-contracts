use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::coverage::{coverage_report, overall_percentage};
use provable_contracts::reverse_coverage::reverse_coverage;
use provable_contracts::schema::parse_contract;

pub fn run(
    contract_dir: &Path,
    binding_path: Option<&Path>,
    _show_fuzz: bool,
    reverse_crate: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Reverse coverage mode
    if let Some(crate_dir) = reverse_crate {
        let bp = binding_path.ok_or("--reverse requires --binding <path>")?;
        let report = reverse_coverage(crate_dir, bp);
        println!("Reverse Coverage Report");
        println!("=======================");
        println!("  Public functions: {}", report.total_pub_fns);
        println!("  Bound (in binding.yaml): {}", report.bound_fns);
        println!("  Annotated (#[contract]): {}", report.annotated_fns);
        println!("  Auto-exempt (trivial): {}", report.exempt_fns);
        println!("  Unbound: {}", report.unbound.len());
        println!(
            "  Coverage: {:.1}% (bound + exempt / total)",
            report.coverage_pct
        );
        if !report.unbound.is_empty() {
            println!("\nUnbound functions:");
            for f in report.unbound.iter().take(20) {
                println!("  {} ({}:{})", f.path, f.file, f.line);
            }
            if report.unbound.len() > 20 {
                println!("  ... and {} more", report.unbound.len() - 20);
            }
        }
        return Ok(());
    }
    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    // Collect all .yaml contracts recursively from the directory
    let mut yaml_paths = Vec::new();
    collect_yaml_files(contract_dir, &mut yaml_paths);

    let mut contracts = Vec::new();
    for path in &yaml_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        match parse_contract(path) {
            Ok(c) => contracts.push((stem, c)),
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", path.display());
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

/// Recursively collect `.yaml` contract files, skipping non-contract directories.
fn collect_yaml_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if dirname == "kaizen" || dirname == "legacy" || dirname == "pipelines" {
                continue;
            }
            collect_yaml_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml")
            && path.file_name().and_then(|n| n.to_str()) != Some("binding.yaml")
        {
            out.push(path);
        }
    }
}
