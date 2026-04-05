//! `pv verify-structure` — verify model architecture matches contracts.
//!
//! Two modes:
//! 1. Contract-only (no model file): enumerate expected tensors from
//!    apr-architecture-schema-v1 equations for a given config.
//! 2. With model file (future): compare expected vs actual tensors.
//!
//! Spec: docs/specifications/sub/model-layout-provability.md (§36, P0-4)

use std::path::Path;

/// Run the verify-structure command.
pub fn run(
    contract_dir: &Path,
    config_json: Option<&Path>,
    model_file: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("pv verify-structure — Architecture Structure Verification");
    println!("=========================================================");
    println!();

    // Load architecture schema contract
    let arch_path = find_contract(contract_dir, "apr-architecture-schema-v1");
    let config_path = find_contract(contract_dir, "model-config-algebra-v1");
    let shape_path = find_contract(contract_dir, "tensor-shape-flow-v1");

    let mut found = 0;
    let mut missing = Vec::new();

    for (name, path) in [
        ("apr-architecture-schema-v1", &arch_path),
        ("model-config-algebra-v1", &config_path),
        ("tensor-shape-flow-v1", &shape_path),
    ] {
        if let Some(p) = path {
            let contract = provable_contracts::schema::parse_contract(p)?;
            let eq_count = contract.equations.len();
            let has_assumes = contract
                .equations
                .values()
                .filter(|e| e.assumes.is_some())
                .count();
            let has_guarantees = contract
                .equations
                .values()
                .filter(|e| e.guarantees.is_some())
                .count();
            println!(
                "  ✓ {name}: {eq_count} equations, {has_assumes} assumes, {has_guarantees} guarantees"
            );
            found += 1;
        } else {
            println!("  ✗ {name}: not found");
            missing.push(name);
        }
    }
    println!();

    // Config analysis
    if let Some(cfg) = config_json {
        if cfg.exists() {
            println!("Config: {}", cfg.display());
            let content = std::fs::read_to_string(cfg)?;
            // Extract key parameters
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                let params = [
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                    "vocab_size",
                ];
                for param in &params {
                    if let Some(val) = json.get(param) {
                        println!("  {param}: {val}");
                    }
                }
                println!();

                // Compute expected tensor count
                let hidden = json.get("hidden_size").and_then(|v| v.as_u64());
                let layers = json.get("num_hidden_layers").and_then(|v| v.as_u64());
                let heads = json.get("num_attention_heads").and_then(|v| v.as_u64());
                let vocab = json.get("vocab_size").and_then(|v| v.as_u64());

                if let (Some(h), Some(l), Some(nh), Some(v)) = (hidden, layers, heads, vocab) {
                    let head_dim = h / nh;
                    println!("Derived: head_dim = {h} / {nh} = {head_dim}");
                    let expected = 1 + l * 9 + 2; // embed + layers*(4attn+3ffn+2norm) + final_norm + lm_head
                    println!("Expected tensors (standard): {expected}");
                    println!("  = 1 (embed) + {l} × 9 (per-layer) + 2 (final_norm + lm_head)");

                    // Divisibility checks from model-config-algebra-v1
                    let kv_heads = json
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(nh);
                    println!();
                    println!("Config algebra checks:");
                    check("hidden_size % num_heads == 0", h % nh == 0);
                    check("num_heads % num_kv_heads == 0", nh % kv_heads == 0);
                    check("head_dim % 2 == 0 (RoPE)", head_dim % 2 == 0);
                    check("hidden_size > 0", h > 0);
                    check("num_layers > 0", l > 0);
                    check("vocab_size > 0", v > 0);
                    println!();
                }
            }
        } else {
            println!("Config file not found: {}", cfg.display());
        }
    } else {
        println!("No --config provided. Use --config path/to/config.json for structural analysis.");
    }

    // Model file analysis (future)
    if let Some(mf) = model_file {
        println!("Model file: {}", mf.display());
        println!("  ⚠ Model file parsing not yet implemented (P0-4 phase 2)");
        println!("  Planned: enumerate actual tensors, compare shapes to arch-schema");
    }

    println!();
    if missing.is_empty() && found == 3 {
        println!("Result: PASS — all 3 architecture contracts found with composition data");
    } else {
        println!(
            "Result: PARTIAL — {found}/3 contracts found, {} missing",
            missing.len()
        );
    }

    Ok(())
}

fn check(label: &str, ok: bool) {
    let icon = if ok { "✓" } else { "✗" };
    println!("  {icon} {label}");
}

fn find_contract(dir: &Path, stem: &str) -> Option<std::path::PathBuf> {
    let direct = dir.join(format!("{stem}.yaml"));
    if direct.exists() {
        return Some(direct);
    }
    // Search subdirectories
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let sub = path.join(format!("{stem}.yaml"));
                if sub.exists() {
                    return Some(sub);
                }
            }
        }
    }
    None
}
