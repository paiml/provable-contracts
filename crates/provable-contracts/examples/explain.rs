//! Explain a contract in natural language — narrative walkthrough.
//!
//! Usage:
//!   cargo run --example explain -- contracts/softmax-kernel-v1.yaml

use provable_contracts::explain::{explain_contract, explain_contract_markdown};
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: explain <contract.yaml>");
            std::process::exit(1);
        },
        std::path::PathBuf::from,
    );

    let contract = parse_contract(&path).unwrap_or_else(|e| {
        eprintln!("Failed to parse: {e}");
        std::process::exit(1);
    });

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    println!("=== Text Format ===\n");
    let text = explain_contract(&contract, stem, None);
    println!("{text}");

    println!("\n=== Markdown Format ===\n");
    let md = explain_contract_markdown(&contract, stem, None);
    // Print first 20 lines of markdown
    for (i, line) in md.lines().enumerate() {
        if i >= 20 {
            println!("... ({} more lines)", md.lines().count() - 20);
            break;
        }
        println!("{line}");
    }
}
