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

    // Semantic search
    let params = QueryParams {
        query: "softmax numerical stability".to_string(),
        limit: 5,
        show_score: true,
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

    // Filtered search
    println!("\n--- Invariant obligations only ---\n");
    let params = QueryParams {
        query: "kernel".to_string(),
        obligation_filter: Some("invariant".to_string()),
        limit: 5,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");
}
