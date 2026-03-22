//! Generate type invariant trait + Kani preservation harnesses.
//!
//! Usage:
//!   cargo run --example invariants -- contracts/validated-tensor-v1.yaml

use provable_contracts::invariant_gen::generate_invariants;
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: invariants <contract.yaml>");
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

    let inv_count = contract.type_invariants.len();
    println!("=== Type Invariants: {stem} ===");
    println!("  type_invariants defined: {inv_count}");

    if inv_count == 0 {
        println!("\n  No type_invariants in this contract.");
        println!("  Add a `type_invariants:` section to define them.");
        println!("  Example:");
        println!("    type_invariants:");
        println!("      - name: tensor_valid");
        println!("        type: ValidatedTensor");
        println!("        predicate: \"!self.dims.is_empty()\"");
        return;
    }

    println!();
    let output = generate_invariants(&contract);
    println!("{output}");

    // Group by type
    let mut types: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for inv in &contract.type_invariants {
        types.insert(&inv.type_name);
    }

    println!("=== Summary ===");
    println!("  Types with invariants: {}", types.len());
    for t in &types {
        let count = contract
            .type_invariants
            .iter()
            .filter(|i| i.type_name == *t)
            .count();
        println!("    {t}: {count} invariant(s)");
    }
    println!("\n=== Usage ===");
    println!("  pv invariants {path}", path = path.display());
    println!(
        "  pv invariants {path} --harnesses  # Also emit Kani proofs",
        path = path.display()
    );
}
