use std::path::Path;

use provable_contracts::graph::{dependency_graph, graph_nodes};
use provable_contracts::schema::parse_contract;

pub fn run(contract_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
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

    let graph = dependency_graph(&refs);
    let nodes = graph_nodes(&graph);

    println!("Contract Dependency Graph");
    println!("=========================");
    println!("Nodes: {}", graph.nodes.len());
    println!();

    // Print each node with its deps
    for node in &nodes {
        let deps = graph.edges.get(&node.stem).cloned().unwrap_or_default();
        if deps.is_empty() {
            println!(
                "  {} (dependents: {}, deps: 0)",
                node.stem, node.dependents
            );
        } else {
            println!(
                "  {} → [{}] (dependents: {})",
                node.stem,
                deps.join(", "),
                node.dependents,
            );
        }
    }

    if !graph.cycles.is_empty() {
        println!();
        println!("CYCLES DETECTED:");
        for cycle in &graph.cycles {
            println!("  {}", cycle.join(" → "));
        }
        return Err("Dependency graph contains cycles".into());
    }

    if !graph.topo_order.is_empty() {
        println!();
        println!("Topological order:");
        for (i, node) in graph.topo_order.iter().enumerate() {
            println!("  {}: {node}", i + 1);
        }
    }

    Ok(())
}
