use std::path::Path;

use provable_contracts::probar_gen::generate_probar_tests;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    print!("{}", generate_probar_tests(&contract));
    Ok(())
}
