use std::path::Path;

use provable_contracts::scaffold::{
    generate_contract_tests, generate_trait,
};
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    println!("// === Trait Definition ===\n");
    print!("{}", generate_trait(&contract));
    println!("\n// === Contract Tests ===\n");
    print!("{}", generate_contract_tests(&contract));

    Ok(())
}
