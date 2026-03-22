//! Generate Coq theorem stubs from YAML contracts.
//!
//! Usage:
//!   cargo run --example coq -- contracts/softmax-kernel-v1.yaml

use provable_contracts::coq_gen::generate_coq_spec;
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: coq <contract.yaml>");
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

    println!("=== Coq Spec: {stem} ===\n");
    let output = generate_coq_spec(&contract, stem);
    println!("{output}");

    // Summary
    let ob_count = contract.proof_obligations.len();
    let coq_count = contract
        .coq_spec
        .as_ref()
        .map_or(0, |s| s.obligations.len());
    println!("=== Summary ===");
    println!("  Proof obligations: {ob_count}");
    println!("  Coq obligation links: {coq_count}");
    if coq_count > 0 {
        let proved = contract
            .coq_spec
            .as_ref()
            .unwrap()
            .obligations
            .iter()
            .filter(|o| o.status == "proved")
            .count();
        println!("  Proved: {proved}/{coq_count}");
    }
    println!("\n=== Usage ===");
    println!(
        "  pv coq {path} > generated/coq/{stem}.v",
        path = path.display()
    );
    println!("  coqc generated/coq/{stem}.v");
}
