use std::path::Path;

use provable_contracts::audit::{audit_binding, audit_contract};
use provable_contracts::binding::parse_binding;
use provable_contracts::error::Severity;
use provable_contracts::schema::{Contract, parse_contract};

pub fn run(
    path: &Path,
    binding_path: Option<&Path>,
    show_coq_tiers: bool,
    show_flux_coverage: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    // Standard traceability audit
    let report = audit_contract(&contract);

    println!("Traceability Audit");
    println!("==================");
    println!("Equations:          {}", report.equations);
    println!("Proof obligations:  {}", report.obligations);
    println!("Falsification tests: {}", report.falsification_tests);
    println!("Kani harnesses:     {}", report.kani_harnesses);
    println!("Type invariants:    {}", contract.type_invariants.len());

    // Lean status
    let lean_proved = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.l4_lean_proved);
    if lean_proved > 0 {
        println!(
            "Lean proved:        {}/{}",
            lean_proved,
            contract
                .verification_summary
                .as_ref()
                .map_or(0, |vs| vs.total_obligations)
        );
    }

    // Coq status
    if let Some(ref spec) = contract.coq_spec {
        let total = spec.obligations.len();
        let proved = spec
            .obligations
            .iter()
            .filter(|o| o.status == "proved")
            .count();
        let admitted = spec
            .obligations
            .iter()
            .filter(|o| o.status == "admitted")
            .count();
        let stubs = total - proved - admitted;
        println!(
            "Coq ({}):{} {proved} proved, {admitted} admitted, {stubs} stub",
            spec.module,
            if total > 0 {
                format!("  {total} obligations —")
            } else {
                " no obligation links".to_string()
            }
        );
    }

    if show_coq_tiers {
        print_coq_tiers(&contract);
    }
    if show_flux_coverage {
        print_flux_coverage(&contract);
    }

    println!();

    if report.violations.is_empty() {
        println!("No audit findings.");
    } else {
        for v in &report.violations {
            println!("{v}");
        }
    }

    let errors = report
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count();

    // Binding audit (if --binding provided)
    let binding_errors = if let Some(bp) = binding_path {
        print_binding_audit(path, &contract, bp)?
    } else {
        0
    };

    let total_errors = errors + binding_errors;
    if total_errors > 0 {
        return Err(format!("Audit found {total_errors} error(s)").into());
    }

    Ok(())
}

fn print_binding_audit(
    path: &Path,
    contract: &Contract,
    bp: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    let binding = parse_binding(bp)?;
    let contract_file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    let report = audit_binding(&[(contract_file, contract)], &binding);

    println!();
    println!("Binding Audit");
    println!("=============");
    println!("Total equations:    {}", report.total_equations);
    println!("Bound equations:    {}", report.bound_equations);
    println!("Implemented:        {}", report.implemented);
    println!("Partial:            {}", report.partial);
    println!("Not implemented:    {}", report.not_implemented);
    println!("Obligations total:  {}", report.total_obligations);
    println!("Obligations covered: {}", report.covered_obligations);
    println!();

    if report.violations.is_empty() {
        println!("No binding gaps found.");
    } else {
        for v in &report.violations {
            println!("{v}");
        }
    }

    Ok(report
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count())
}

fn print_coq_tiers(contract: &Contract) {
    println!();
    println!("Coq Proof Tiers");
    println!("===============");
    for ob in &contract.proof_obligations {
        let has_kani = !contract.kani_harnesses.is_empty();
        let has_lean = ob
            .lean
            .as_ref()
            .is_some_and(|l| l.status == provable_contracts::schema::LeanStatus::Proved);
        let coq_status = contract.coq_spec.as_ref().and_then(|spec| {
            spec.obligations
                .iter()
                .find(|co| co.links_to == ob.property)
                .map(|co| co.status.as_str())
        });
        let tier = match (has_lean, coq_status) {
            (_, Some("proved")) => "coq-proved",
            (_, Some("admit")) => "coq-admit",
            (true, _) => "lean-proved",
            _ if has_kani => "kani-only",
            _ => "unverified",
        };
        println!("  [{tier}] {}", ob.property);
    }
}

fn print_flux_coverage(contract: &Contract) {
    println!();
    println!("Flux Shape Coverage");
    println!("===================");
    let shape_keywords = ["shape", "dim", "len", "size", "rows", "cols"];
    for ob in &contract.proof_obligations {
        let is_shape = shape_keywords
            .iter()
            .any(|kw| ob.property.to_lowercase().contains(kw));
        let status = if is_shape {
            "flux-dischargeable"
        } else {
            "kani-needed"
        };
        println!("  [{status}] {}", ob.property);
    }
}
