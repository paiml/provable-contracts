//! Generate Flux refinement type annotations from contracts.
//!
//! Usage:
//!   cargo run --example flux -- contracts/softmax-kernel-v1.yaml
//!   cargo run --example flux -- contracts/tensor-shape-flow-v1.yaml

use provable_contracts::flux_gen::generate_flux_annotations;
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: flux <contract.yaml>");
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

    println!("=== Flux Annotations: {stem} ===\n");
    let output = generate_flux_annotations(&contract, stem);
    println!("{output}");

    // Show shape detection
    let has_shapes = contract.equations.iter().any(|(_, eq)| {
        let f = eq.formula.to_lowercase();
        f.contains("shape") || f.contains("dim") || f.contains("len") || f.contains("product")
    });
    println!("=== Shape Detection ===");
    println!(
        "  Shape-related equations: {}",
        if has_shapes {
            "yes (RefinedVec)"
        } else {
            "no (generic)"
        }
    );
    println!("  Equations: {}", contract.equations.len());
    println!("\n=== Usage ===");
    println!("  cargo flux  # Verify refinement types via SMT");
}
