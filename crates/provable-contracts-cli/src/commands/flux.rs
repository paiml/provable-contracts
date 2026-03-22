use std::path::Path;

use provable_contracts::flux_gen::generate_flux_annotations;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path, verify: bool) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let output = generate_flux_annotations(&contract, stem);
    print!("{output}");

    if verify {
        eprintln!();
        eprintln!("Running `cargo flux` to verify annotations...");
        let status = std::process::Command::new("cargo").arg("flux").status();
        match status {
            Ok(s) if s.success() => eprintln!("Flux verification: PASS"),
            Ok(s) => {
                eprintln!("Flux verification: FAIL (exit {})", s.code().unwrap_or(-1));
                return Err("Flux verification failed".into());
            }
            Err(e) => {
                eprintln!("Could not run `cargo flux`: {e}");
                eprintln!("Install Flux: https://flux-rs.github.io");
                return Err("cargo flux not found".into());
            }
        }
    }

    Ok(())
}
