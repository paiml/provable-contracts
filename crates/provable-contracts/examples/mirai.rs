//! Generate MIRAI abstract interpretation annotations from contracts.
//!
//! Usage:
//!   cargo run --example mirai -- contracts/softmax-kernel-v1.yaml

use provable_contracts::mirai_gen::generate_mirai_annotations;
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: mirai <contract.yaml>");
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

    println!("=== MIRAI Annotations: {stem} ===\n");
    let output = generate_mirai_annotations(&contract, stem);
    println!("{output}");

    println!("=== Usage ===");
    println!("  cargo install --git https://github.com/facebookexperimental/MIRAI mirai");
    println!("  cargo mirai -- --diag=default");
}
