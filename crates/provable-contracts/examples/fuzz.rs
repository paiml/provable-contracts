//! Generate coverage-guided fuzz targets from YAML contracts.
//!
//! Usage:
//!   cargo run --example fuzz -- contracts/softmax-kernel-v1.yaml

use provable_contracts::fuzz_gen::{generate_fuzz_cargo_toml, generate_fuzz_target};
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: fuzz <contract.yaml>");
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

    println!("=== Fuzz Target: {stem} ===\n");
    let target = generate_fuzz_target(&contract, stem);
    println!("{target}");

    println!("=== fuzz/Cargo.toml ===\n");
    let cargo = generate_fuzz_cargo_toml(stem);
    println!("{cargo}");

    println!("=== Usage ===");
    println!("  mkdir -p fuzz/fuzz_targets");
    println!(
        "  pv fuzz {path} > fuzz/fuzz_targets/{stem}.rs",
        path = path.display()
    );
    println!("  cargo fuzz run {stem}");
}
