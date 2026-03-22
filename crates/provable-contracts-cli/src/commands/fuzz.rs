use std::path::Path;

use provable_contracts::fuzz_gen::{generate_fuzz_cargo_toml, generate_fuzz_target};
use provable_contracts::schema::parse_contract;

pub fn run(
    path: &Path,
    sanitizer: Option<&str>,
    max_len: Option<usize>,
    timeout: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let mut output = generate_fuzz_target(&contract, stem);

    // Append sanitizer/max_len/timeout as comments if specified
    if sanitizer.is_some() || max_len.is_some() || timeout.is_some() {
        output.push_str("\n// Fuzz configuration:\n");
        if let Some(san) = sanitizer {
            output.push_str(&format!("// sanitizer: {san}\n"));
        }
        if let Some(ml) = max_len {
            output.push_str(&format!("// max_len: {ml}\n"));
        }
        if let Some(t) = timeout {
            output.push_str(&format!("// timeout: {t}s\n"));
        }
    }

    print!("{output}");

    if crate::verbosity::is_verbose() {
        eprintln!();
        eprintln!("--- fuzz/Cargo.toml ---");
        eprint!("{}", generate_fuzz_cargo_toml(stem));
    }

    Ok(())
}
