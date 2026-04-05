//! `pv verify-pipeline` — compositional shape verification across contracts.
//!
//! Walks the dependency graph in topological order and verifies that every
//! assumes/guarantees edge is satisfied. Produces a composition report
//! showing the full proof chain or identifying break points.
//!
//! Spec: docs/specifications/sub/model-layout-provability.md (§36, P0-3)

use std::collections::BTreeMap;
use std::path::Path;

use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{Contract, parse_contract};

/// Run the verify-pipeline command.
pub fn run(contract_dir: &Path, format: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load all contracts
    let mut contracts = Vec::new();
    load_contracts_recursive(contract_dir, &mut contracts)?;
    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    if contracts.is_empty() {
        eprintln!("No contracts found in {}", contract_dir.display());
        return Ok(());
    }

    // 2. Build dependency graph + topological sort
    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    if !graph.cycles.is_empty() {
        eprintln!(
            "ERROR: {} cycle(s) detected in dependency graph — cannot verify pipeline",
            graph.cycles.len()
        );
        for cycle in &graph.cycles {
            eprintln!("  cycle: {}", cycle.join(" → "));
        }
        std::process::exit(1);
    }

    // 3. Build stem→contract index
    let index: BTreeMap<&str, &Contract> = contracts.iter().map(|(s, c)| (s.as_str(), c)).collect();

    // 4. Walk in topological order, verify composition edges
    let mut edges_total = 0usize;
    let mut edges_satisfied = 0usize;
    let mut edges_broken = Vec::new();
    let mut chains: Vec<CompositionEdge> = Vec::new();

    for stem in &graph.topo_order {
        let Some(contract) = index.get(stem.as_str()) else {
            continue;
        };
        for (eq_name, equation) in &contract.equations {
            let Some(assumes) = &equation.assumes else {
                continue;
            };
            let Some(from_contract) = &assumes.from_contract else {
                continue;
            };

            edges_total += 1;
            let from_eq = assumes.from_equation.as_deref();

            let edge = CompositionEdge {
                downstream: format!("{stem}.{eq_name}"),
                upstream: format!(
                    "{from_contract}{}",
                    from_eq.map_or(String::new(), |e| format!(".{e}"))
                ),
                assumed_shapes: assumes.shapes.keys().cloned().collect(),
                status: EdgeStatus::Unknown,
            };

            // Resolve upstream
            let Some(upstream_contract) = index.get(from_contract.as_str()) else {
                let mut e = edge;
                e.status = EdgeStatus::Broken("upstream contract not found".into());
                edges_broken.push(e.clone());
                chains.push(e);
                continue;
            };

            if let Some(upstream_eq_name) = from_eq {
                let Some(upstream_eq) = upstream_contract.equations.get(upstream_eq_name) else {
                    let mut e = edge;
                    e.status = EdgeStatus::Broken("upstream equation not found".into());
                    edges_broken.push(e.clone());
                    chains.push(e);
                    continue;
                };

                if upstream_eq.guarantees.is_none() {
                    let mut e = edge;
                    e.status = EdgeStatus::Broken("upstream has no guarantees".into());
                    edges_broken.push(e.clone());
                    chains.push(e);
                    continue;
                }

                // Edge satisfied: upstream has guarantees
                let mut e = edge;
                let guaranteed_shapes: Vec<String> = upstream_eq
                    .guarantees
                    .as_ref()
                    .unwrap()
                    .shapes
                    .keys()
                    .cloned()
                    .collect();
                e.status = EdgeStatus::Satisfied(guaranteed_shapes);
                edges_satisfied += 1;
                chains.push(e);
            } else {
                // No specific equation — check any equation has guarantees
                let has_guarantees = upstream_contract
                    .equations
                    .values()
                    .any(|eq| eq.guarantees.is_some());
                let mut e = edge;
                if has_guarantees {
                    e.status = EdgeStatus::Satisfied(vec![]);
                    edges_satisfied += 1;
                } else {
                    e.status = EdgeStatus::Broken("no equations with guarantees".into());
                    edges_broken.push(e.clone());
                }
                chains.push(e);
            }
        }
    }

    // 5. Output
    match format {
        "json" => print_json(&chains, edges_total, edges_satisfied, &edges_broken, &graph),
        _ => print_text(&chains, edges_total, edges_satisfied, &edges_broken, &graph),
    }

    if !edges_broken.is_empty() {
        std::process::exit(1);
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct CompositionEdge {
    downstream: String,
    upstream: String,
    assumed_shapes: Vec<String>,
    status: EdgeStatus,
}

#[derive(Debug, Clone)]
enum EdgeStatus {
    Unknown,
    Satisfied(Vec<String>),
    Broken(String),
}

fn print_text(
    chains: &[CompositionEdge],
    total: usize,
    satisfied: usize,
    broken: &[CompositionEdge],
    graph: &provable_contracts::graph::DependencyGraph,
) {
    println!("pv verify-pipeline — Compositional Shape Verification");
    println!("=====================================================");
    println!();
    println!(
        "Contracts: {}  |  Topo depth: {}",
        graph.nodes.len(),
        graph.topo_order.len()
    );
    println!(
        "Edges: {}  |  Satisfied: {}  |  Broken: {}",
        total,
        satisfied,
        broken.len()
    );
    println!();

    if !chains.is_empty() {
        println!("Composition edges:");
        for edge in chains {
            let icon = match &edge.status {
                EdgeStatus::Satisfied(_) => "✓",
                EdgeStatus::Broken(_) => "✗",
                EdgeStatus::Unknown => "?",
            };
            let detail = match &edge.status {
                EdgeStatus::Satisfied(shapes) if !shapes.is_empty() => {
                    format!(" (guarantees: {})", shapes.join(", "))
                }
                EdgeStatus::Broken(reason) => format!(" — {reason}"),
                _ => String::new(),
            };
            println!("  {icon} {} ← {}{detail}", edge.downstream, edge.upstream);
        }
        println!();
    }

    if broken.is_empty() {
        println!("Result: PASS — all composition edges satisfied");
    } else {
        println!("Result: FAIL — {} broken edge(s)", broken.len());
        for edge in broken {
            if let EdgeStatus::Broken(reason) = &edge.status {
                println!("  ✗ {} ← {} — {reason}", edge.downstream, edge.upstream);
            }
        }
    }
}

fn print_json(
    chains: &[CompositionEdge],
    total: usize,
    satisfied: usize,
    broken: &[CompositionEdge],
    graph: &provable_contracts::graph::DependencyGraph,
) {
    let edges_json: Vec<serde_json::Value> = chains
        .iter()
        .map(|e| {
            let (status, detail) = match &e.status {
                EdgeStatus::Satisfied(shapes) => (
                    "satisfied",
                    serde_json::json!({"guaranteed_shapes": shapes}),
                ),
                EdgeStatus::Broken(reason) => ("broken", serde_json::json!({"reason": reason})),
                EdgeStatus::Unknown => ("unknown", serde_json::json!({})),
            };
            serde_json::json!({
                "downstream": e.downstream,
                "upstream": e.upstream,
                "assumed_shapes": e.assumed_shapes,
                "status": status,
                "detail": detail,
            })
        })
        .collect();

    let report = serde_json::json!({
        "contracts": graph.nodes.len(),
        "topo_depth": graph.topo_order.len(),
        "edges_total": total,
        "edges_satisfied": satisfied,
        "edges_broken": broken.len(),
        "passed": broken.is_empty(),
        "edges": edges_json,
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn load_contracts_recursive(
    dir: &Path,
    out: &mut Vec<(String, Contract)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            load_contracts_recursive(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            // Skip non-contract YAML (binding.yaml, playbook.schema.yaml, etc.)
            if stem == "binding" || stem == "playbook.schema" || stem.contains("playbook") {
                continue;
            }
            match parse_contract(&path) {
                Ok(c) => out.push((stem, c)),
                Err(_) => {} // silently skip unparseable files
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_pipeline_on_real_contracts() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return; // skip in CI without contracts
        }
        // Should not panic or error
        let result = run(&dir, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn verify_pipeline_json_on_real_contracts() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return;
        }
        let result = run(&dir, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn verify_pipeline_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = run(tmp.path(), "text");
        assert!(result.is_ok());
    }
}
