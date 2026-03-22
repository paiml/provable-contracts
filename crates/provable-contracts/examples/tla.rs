//! Generate TLA+ system-level specs from the contract dependency DAG.
//!
//! Usage:
//!   cargo run --example tla -- contracts/

use std::path::Path;

use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{Contract, parse_contract};
use provable_contracts::tla_gen::generate_tla_module;

fn main() {
    let dir = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: tla <contracts-dir/>");
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

    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    println!("=== TLA+ Module ===");
    println!("Contracts: {}", contracts.len());
    println!("Dependency edges: {}", graph.edges.len());
    println!();

    let tla = generate_tla_module("InferencePipeline", &refs, &graph);

    // Print first 40 lines
    for (i, line) in tla.lines().enumerate() {
        if i >= 40 {
            println!("... ({} more lines)", tla.lines().count() - 40);
            break;
        }
        println!("{line}");
    }

    // Show top dependencies
    println!("\n=== Top Dependencies (by in-degree) ===");
    let mut in_degree: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for deps in graph.edges.values() {
        for dep in deps {
            *in_degree.entry(dep.as_str()).or_default() += 1;
        }
    }
    let mut sorted: Vec<_> = in_degree.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (stem, count) in sorted.iter().take(5) {
        println!("  {stem}: {count} dependents");
    }

    println!("\n=== Usage ===");
    println!(
        "  pv tla {dir} --output pipeline.tla",
        dir = Path::new(&dir).display()
    );
    println!("  tlc pipeline.tla  # Model check with TLC");
}
