use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::generate::generate_all;
use provable_contracts::schema::parse_contract;

pub fn run(
    contract: &Path,
    output_dir: &Path,
    binding_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let c = parse_contract(contract)?;

    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    let stem = contract
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("contract");

    let result = generate_all(&c, stem, output_dir, binding.as_ref())?;

    println!(
        "Generated {} files in {}:",
        result.files.len(),
        output_dir.display()
    );
    for f in &result.files {
        println!(
            "  {} ({}, {} bytes)",
            f.relative_path.display(),
            f.kind,
            f.bytes
        );
    }

    Ok(())
}
