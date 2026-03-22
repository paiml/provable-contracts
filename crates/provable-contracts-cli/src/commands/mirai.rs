use std::path::Path;

use provable_contracts::mirai_gen::generate_mirai_annotations;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path, emit_tags: bool) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let output = generate_mirai_annotations(&contract, stem);
    print!("{output}");

    if emit_tags {
        println!();
        println!("// --- MIRAI Tag Structs (--emit-tags) ---");
        for eq_name in contract.equations.keys() {
            let tag = eq_name.replace('-', "_");
            println!("pub struct Tag{} {{}}", to_pascal_case(&tag));
        }
    }

    Ok(())
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}
