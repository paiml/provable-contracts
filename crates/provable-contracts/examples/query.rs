//! Example: Query contracts using BM25 semantic search.
//!
//! ```bash
//! cargo run --example query
//! ```

use std::path::Path;

use provable_contracts::query::{self, ContractIndex, QueryParams};

fn main() {
    let contracts_dir = Path::new("contracts");
    let index = ContractIndex::from_directory(contracts_dir)
        .expect("contracts/ directory must exist");

    println!("Indexed {} contracts\n", index.entries.len());

    // Semantic search with score + proof-status enrichment
    let params = QueryParams {
        query: "softmax numerical stability".to_string(),
        limit: 3,
        show_score: true,
        show_proof_status: true,
        show_paper: true,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // Literal search
    println!("\n--- Literal search for 'RMSNorm' ---\n");
    let params = QueryParams {
        query: "RMSNorm".to_string(),
        mode: query::SearchMode::Literal,
        limit: 3,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // Filtered search with obligation type
    println!("\n--- Invariant obligations only ---\n");
    let params = QueryParams {
        query: "kernel".to_string(),
        obligation_filter: Some("invariant".to_string()),
        limit: 5,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // Markdown output
    println!("\n--- Markdown format ---\n");
    let params = QueryParams {
        query: "rmsnorm".to_string(),
        show_score: true,
        show_paper: true,
        limit: 2,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{}", output.to_markdown());

    // Binding enrichment
    let binding_path = Path::new("contracts/aprender/binding.yaml");
    if binding_path.exists() {
        println!("\n--- Binding enrichment ---\n");
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 2,
            ..Default::default()
        };
        let output = query::execute(&index, &params);
        print!("{output}");
    }
}
