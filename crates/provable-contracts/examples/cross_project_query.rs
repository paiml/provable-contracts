//! Example: Cross-project query with violations and coverage map.
//!
//! Demonstrates the Phase 3 cross-project search features:
//! - `--call-sites`: Where contracts are used across sibling projects
//! - `--violations`: Binding gaps and unproven obligations
//! - `--coverage-map`: Per-project coverage matrix
//!
//! Run from the workspace root:
//!   cargo run --example cross_project_query

use std::path::Path;

use provable_contracts::query::{self, ContractIndex, QueryParams};

fn main() {
    let contracts_dir = Path::new("contracts");
    let index = ContractIndex::from_directory(contracts_dir)
        .expect("contracts/ directory must exist");

    // 1. Query with all cross-project enrichment
    println!("=== Cross-Project Query: \"softmax\" ===\n");
    let params = QueryParams {
        query: "softmax".to_string(),
        show_call_sites: true,
        show_violations: true,
        show_coverage_map: true,
        show_score: true,
        limit: 3,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 2. Query for contracts with binding gaps
    println!("\n=== Contracts with Violations: \"metrics\" ===\n");
    let params = QueryParams {
        query: "metrics".to_string(),
        show_violations: true,
        show_coverage_map: true,
        limit: 5,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 3. JSON output for CI integration
    println!("\n=== JSON Output (first result) ===\n");
    let params = QueryParams {
        query: "attention".to_string(),
        show_call_sites: true,
        show_coverage_map: true,
        limit: 1,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    println!("{}", serde_json::to_string_pretty(&output).unwrap());

    // 4. Markdown output
    println!("\n=== Markdown Output ===\n");
    let params = QueryParams {
        query: "rmsnorm".to_string(),
        show_call_sites: true,
        show_violations: true,
        show_coverage_map: true,
        limit: 2,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{}", output.to_markdown());
}
