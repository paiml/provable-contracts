use std::path::Path;

use provable_contracts::audit::audit_contract;
use provable_contracts::error::Severity;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let report = audit_contract(&contract);

    println!("Traceability Audit");
    println!("==================");
    println!("Equations:          {}", report.equations);
    println!("Proof obligations:  {}", report.obligations);
    println!("Falsification tests: {}", report.falsification_tests);
    println!("Kani harnesses:     {}", report.kani_harnesses);
    println!();

    if report.violations.is_empty() {
        println!("No audit findings.");
    } else {
        for v in &report.violations {
            println!("{v}");
        }
        let errors = report
            .violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .count();
        if errors > 0 {
            return Err(format!(
                "Audit found {errors} error(s)"
            )
            .into());
        }
    }

    Ok(())
}
