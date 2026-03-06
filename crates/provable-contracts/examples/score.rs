//! Example: Score contracts using the scoring module.
//!
//! ```bash
//! cargo run --example score
//! ```

use std::path::Path;

fn main() {
    let contracts_dir = Path::new("contracts");

    let mut entries: Vec<_> = std::fs::read_dir(contracts_dir)
        .expect("contracts/ directory must exist")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                == Some("yaml")
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut scores = Vec::new();
    for entry in &entries {
        let path = entry.path();
        let contract = match provable_contracts::schema::parse_contract(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let score = provable_contracts::scoring::score_contract(&contract, None, stem);
        scores.push(score);
    }

    // Sort by composite score descending
    scores.sort_by(|a, b| b.composite.partial_cmp(&a.composite).unwrap());

    println!("=== Top 10 Contracts by Score ===\n");
    for s in scores.iter().take(10) {
        print!("{s}");
    }

    println!("\n=== Bottom 5 Contracts (improvement targets) ===\n");
    for s in scores.iter().rev().take(5) {
        print!("{s}");
    }

    let mean: f64 = scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64;
    println!(
        "\nOverall: {} contracts, mean {:.2} (Grade {})",
        scores.len(),
        mean,
        provable_contracts::scoring::Grade::from_score(mean)
    );
}
