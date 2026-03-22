//! Contract explanation — chain-of-thought narrative for any contract.
//!
//! Unlike `book_gen` (reference tables), explain produces prose that walks
//! through the contract section by section, explaining the *why* behind
//! each element.

use std::fmt::Write;

use crate::binding::BindingRegistry;
use crate::proof_status::compute_proof_level;
use crate::schema::{Contract, ObligationType};

/// Return a mathematical pattern description for the given obligation type.
pub fn obligation_pattern(ot: ObligationType) -> &'static str {
    match ot {
        ObligationType::Invariant => "∀x ∈ Domain: P(f(x)) — property holds for all inputs",
        ObligationType::Equivalence => "∀x: |f(x) - g(x)| < ε — two implementations agree",
        ObligationType::Bound => "∀x: a ≤ f(x)_i ≤ b — output range bounded",
        ObligationType::Monotonicity => "x_i > x_j → f(x)_i > f(x)_j — order preserved",
        ObligationType::Idempotency => "f(f(x)) = f(x) — applying twice gives same result",
        ObligationType::Linearity => "f(αx) = α·f(x) — homogeneous scaling",
        ObligationType::Symmetry => "f(permute(x)) related to f(x) — permutation property",
        ObligationType::Associativity => "(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) — grouping invariant",
        ObligationType::Conservation => "Q(before) = Q(after) — conserved quantity",
        ObligationType::Ordering => "a ≤ b → f(a) ≤ f(b) — order relation maintained",
        ObligationType::Completeness => "∀ required elements present — completeness verified",
        ObligationType::Soundness => "∀x: P(x) → Q(f(x)) — soundness of transformation",
        ObligationType::Involution => "f(f(x)) = x — involution (self-inverse)",
        ObligationType::Determinism => "f(x) = f(x) — deterministic output for same input",
        ObligationType::Roundtrip => "decode(encode(x)) = x — roundtrip fidelity",
        ObligationType::StateMachine => "S × A → S — valid state transitions",
        ObligationType::Classification => "f(x) ∈ C — output belongs to valid class set",
        ObligationType::Independence => "P(A∩B) = P(A)·P(B) — statistical independence",
        ObligationType::Termination => "algorithm terminates in finite steps",
    }
}

/// Kani strategy explanation for the explain output.
fn strategy_explanation(strategy: &str) -> &str {
    match strategy {
        "exhaustive" => "verify for ALL inputs within bound",
        "stub_float" => {
            "assume Lean-proved postconditions on transcendentals, verify surrounding code"
        }
        "compositional" => "verify sub-kernels separately, compose proofs",
        "bounded_int" => "integer-only verification within bound",
        _ => "bounded model check",
    }
}

/// Generate a chain-of-thought narrative explanation for a contract.
///
/// `stem` is the contract filename without `.yaml`.
/// `binding` is an optional binding registry for binding status.
pub fn explain_contract(
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) -> String {
    let mut out = String::with_capacity(4096);

    write_header(&mut out, contract, stem);
    write_what(&mut out, contract);
    write_equations(&mut out, contract);
    write_obligations(&mut out, contract);
    write_verification_ladder(&mut out, contract, binding);
    write_falsification_tests(&mut out, contract);
    write_kani_harnesses(&mut out, contract);
    write_kernel_phases(&mut out, contract);
    write_simd_dispatch(&mut out, contract);
    write_enforcement(&mut out, contract);
    write_qa_gate(&mut out, contract);
    write_binding_status(&mut out, contract, stem, binding);

    out
}

fn write_header(out: &mut String, contract: &Contract, stem: &str) {
    let _ = writeln!(out, "{stem} (v{})", contract.metadata.version);
    let _ = writeln!(out, "{}", contract.metadata.description);
    let _ = writeln!(out);
}

fn write_what(out: &mut String, contract: &Contract) {
    let _ = writeln!(out, "What this contract specifies");

    let refs = &contract.metadata.references;
    if refs.is_empty() {
        let _ = writeln!(
            out,
            "  This contract specifies {}.",
            contract.metadata.description
        );
    } else {
        let _ = write!(
            out,
            "  This contract specifies {}. It derives from",
            contract.metadata.description
        );
        for (i, r) in refs.iter().enumerate() {
            if i == 0 {
                let _ = write!(out, " {r}");
            } else {
                let _ = write!(out, " and {r}");
            }
        }
        let _ = writeln!(out, ".");
    }

    if !contract.metadata.depends_on.is_empty() {
        let _ = write!(out, "  Depends on:");
        for dep in &contract.metadata.depends_on {
            let _ = write!(out, " {dep}");
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(out);
}

fn write_equations(out: &mut String, contract: &Contract) {
    if contract.equations.is_empty() {
        return;
    }
    let _ = writeln!(out, "Governing equations");
    let _ = writeln!(out);

    for (name, eq) in &contract.equations {
        let _ = writeln!(out, "  {name}");
        let _ = writeln!(out, "    {}", eq.formula);

        if let Some(ref dom) = eq.domain {
            let _ = writeln!(out, "    Domain: {dom}");
        }
        if let Some(ref cod) = eq.codomain {
            let _ = writeln!(out, "    Range:  {cod}");
        }

        if !eq.invariants.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "    Invariants:");
            for (i, inv) in eq.invariants.iter().enumerate() {
                let _ = writeln!(out, "      {}. {inv}", i + 1);
            }
        }

        if !eq.preconditions.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "    Preconditions (caller must guarantee):");
            for pre in &eq.preconditions {
                let _ = writeln!(out, "      - {pre}");
            }
        }

        if !eq.postconditions.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "    Postconditions (kernel guarantees):");
            for post in &eq.postconditions {
                let _ = writeln!(out, "      - {post}");
            }
        }

        if let Some(ref theorem) = eq.lean_theorem {
            let _ = writeln!(out, "    Lean theorem: {theorem}");
        }

        let _ = writeln!(out);
    }
}

fn write_obligations(out: &mut String, contract: &Contract) {
    if contract.proof_obligations.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "Proof obligations ({})",
        contract.proof_obligations.len()
    );
    let _ = writeln!(out);

    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        let _ = writeln!(out, "  {}. [{}] {}", i + 1, ob.obligation_type, ob.property);
        let _ = writeln!(
            out,
            "     Pattern: {}",
            obligation_pattern(ob.obligation_type)
        );

        if let Some(ref formal) = ob.formal {
            let _ = writeln!(out, "     Formal:  {formal}");
        }
        if let Some(tol) = ob.tolerance {
            let _ = writeln!(out, "     Tolerance: {tol:e}");
        }

        // Lean proof status
        if let Some(ref lean) = ob.lean {
            let module = lean.module.as_deref().unwrap_or("?");
            let _ = writeln!(
                out,
                "     Lean: {module}.{} ({})",
                lean.theorem, lean.status
            );
            if !lean.depends_on.is_empty() {
                let _ = writeln!(out, "       Depends: {}", lean.depends_on.join(", "));
            }
            if let Some(ref notes) = lean.notes {
                let _ = writeln!(out, "       Note: {notes}");
            }
        }

        // Cross-reference to falsification tests
        let matching_ft: Vec<&str> = contract
            .falsification_tests
            .iter()
            .filter(|ft| {
                ob.property.to_lowercase().contains(&ft.rule.to_lowercase())
                    || ft.rule.to_lowercase().contains(&ob.property.to_lowercase())
            })
            .map(|ft| ft.id.as_str())
            .collect();

        // Cross-reference to Kani harnesses
        let matching_kh: Vec<&str> = contract
            .kani_harnesses
            .iter()
            .filter(|kh| {
                kh.property
                    .as_ref()
                    .is_some_and(|p| p.to_lowercase().contains(&ob.property.to_lowercase()))
            })
            .map(|kh| kh.id.as_str())
            .collect();

        if !matching_ft.is_empty() || !matching_kh.is_empty() {
            let mut parts = Vec::new();
            for ft in &matching_ft {
                parts.push(format!("L2 ({ft})"));
            }
            for kh in &matching_kh {
                parts.push(format!("L4 ({kh})"));
            }
            if ob
                .lean
                .as_ref()
                .is_some_and(|l| l.status.to_string() == "proved")
            {
                parts.push("L5 (Lean)".to_string());
            }
            if !parts.is_empty() {
                let _ = writeln!(out, "     Verified at: {}", parts.join(", "));
            }
        }

        let _ = writeln!(out);
    }
}

fn write_verification_ladder(
    out: &mut String,
    contract: &Contract,
    binding: Option<&BindingRegistry>,
) {
    let total = contract.proof_obligations.len();
    if total == 0 {
        return;
    }

    let lean_proved = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.l4_lean_proved as usize);
    let kani_count = contract.kani_harnesses.len();
    let ft_count = contract.falsification_tests.len();
    let level = compute_proof_level(contract, None);

    let _ = writeln!(out, "Verification ladder");
    if lean_proved > 0 {
        let pct = lean_proved * 100 / total;
        let _ = writeln!(out, "  L5 (Lean):  {lean_proved}/{total} proved ({pct}%)");
    }
    if kani_count > 0 {
        // Summarize strategies
        let mut strategies: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for kh in &contract.kani_harnesses {
            let s = kh
                .strategy
                .map_or_else(|| "default".to_string(), |s| s.to_string());
            *strategies.entry(s).or_default() += 1;
        }
        let strat_summary: Vec<String> = strategies
            .iter()
            .map(|(s, n)| format!("{n}× {s}"))
            .collect();
        let _ = writeln!(
            out,
            "  L4 (Kani):  {kani_count} harnesses ({})",
            strat_summary.join(", ")
        );
    }
    if ft_count > 0 {
        let _ = writeln!(out, "  L2 (Tests): {ft_count} falsification tests");
    }

    let _ = writeln!(out, "  Level: {level}");

    if let Some(b) = binding {
        let stem_for_binding = format!("{}.yaml", ""); // binding needs contract filename
        let _ = b; // suppress unused — binding status shown in write_binding_status
        let _ = stem_for_binding;
    }
    let _ = writeln!(out);
}

fn write_falsification_tests(out: &mut String, contract: &Contract) {
    if contract.falsification_tests.is_empty() {
        return;
    }
    let _ = writeln!(out, "Falsification tests (Popperian)");
    let _ = writeln!(
        out,
        "  Each test tries to refute the contract. Survival = evidence"
    );
    let _ = writeln!(out, "  for, not proof of, correctness.");
    let _ = writeln!(out);

    for ft in &contract.falsification_tests {
        let _ = writeln!(out, "  {}: {}", ft.id, ft.rule);
        let _ = writeln!(out, "    Predicts: {}", ft.prediction);
        if let Some(ref test) = ft.test {
            let _ = writeln!(out, "    Method:   {test}");
        }
        let _ = writeln!(out, "    Catches:  {}", ft.if_fails);
        let _ = writeln!(out);
    }
}

fn write_kani_harnesses(out: &mut String, contract: &Contract) {
    if contract.kani_harnesses.is_empty() {
        return;
    }
    let _ = writeln!(out, "Kani bounded model checking");

    for kh in &contract.kani_harnesses {
        let bound = kh.bound.map_or_else(|| "-".to_string(), |b| b.to_string());
        let strategy = kh
            .strategy
            .map_or_else(|| "default".to_string(), |s| s.to_string());
        let harness = kh.harness.as_deref().unwrap_or(&kh.id);

        let _ = writeln!(
            out,
            "  {}: {} (bound: {}, {})",
            kh.id, harness, bound, strategy
        );
        let _ = writeln!(out, "    {}", strategy_explanation(&strategy));
        if let Some(ref prop) = kh.property {
            let _ = writeln!(out, "    Property: {prop}");
        }
        let _ = writeln!(out);
    }
}

fn write_kernel_phases(out: &mut String, contract: &Contract) {
    let Some(ref ks) = contract.kernel_structure else {
        return;
    };
    let _ = writeln!(out, "Kernel phases");
    for (i, phase) in ks.phases.iter().enumerate() {
        let inv = phase
            .invariant
            .as_deref()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  {}. {} — {}{}",
            i + 1,
            phase.name,
            phase.description,
            inv
        );
    }
    let _ = writeln!(out);
}

fn write_simd_dispatch(out: &mut String, contract: &Contract) {
    if contract.simd_dispatch.is_empty() {
        return;
    }
    let _ = writeln!(out, "SIMD dispatch");
    for (kernel, dispatch) in &contract.simd_dispatch {
        let targets: Vec<String> = dispatch
            .iter()
            .map(|(isa, target)| format!("{isa} → {target}"))
            .collect();
        let _ = writeln!(out, "  {kernel}: {}", targets.join(" | "));
    }
    let _ = writeln!(out);
}

fn write_enforcement(out: &mut String, contract: &Contract) {
    if contract.enforcement.is_empty() {
        return;
    }
    let _ = writeln!(out, "Enforcement");
    for (name, rule) in &contract.enforcement {
        let severity = rule.severity.as_deref().unwrap_or("ERROR");
        let check = rule.check.as_deref().unwrap_or("-");
        let _ = writeln!(
            out,
            "  {name} — {} ({severity}) → {check}",
            rule.description
        );
    }
    let _ = writeln!(out);
}

fn write_qa_gate(out: &mut String, contract: &Contract) {
    let Some(ref qa) = contract.qa_gate else {
        return;
    };
    let _ = writeln!(out, "Quality gate: {} {}", qa.id, qa.name);
    if let Some(ref desc) = qa.description {
        let _ = writeln!(out, "  {desc}");
    }
    if let Some(ref criteria) = qa.pass_criteria {
        let _ = writeln!(out, "  Pass: {criteria}");
    }
    if let Some(ref falsification) = qa.falsification {
        let _ = writeln!(out, "  Mutation: {falsification}");
    }
    let _ = writeln!(out);
}

fn write_binding_status(
    out: &mut String,
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) {
    let Some(registry) = binding else {
        return;
    };

    let contract_file = format!("{stem}.yaml");
    let mut found = false;

    for eq_name in contract.equations.keys() {
        let status = registry
            .bindings
            .iter()
            .find(|b| b.contract == contract_file && b.equation == *eq_name)
            .map_or_else(|| "missing".to_string(), |b| b.status.to_string());

        if !found {
            let _ = writeln!(out, "Binding status ({})", registry.target_crate);
            found = true;
        }
        let _ = writeln!(out, "  {eq_name}: {status}");
    }

    if found {
        let _ = writeln!(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn explain_minimal_contract() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test kernel"
  references: ["Paper A"]
equations:
  f:
    formula: "f(x) = x"
    domain: "x ∈ ℝ^n"
proof_obligations:
  - type: invariant
    property: "output finite"
    formal: "|f(x)| < ∞"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    test: "proptest"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
        )
        .unwrap();

        let output = explain_contract(&contract, "test-kernel-v1", None);
        assert!(output.contains("test-kernel-v1 (v1.0.0)"));
        assert!(output.contains("Test kernel"));
        assert!(output.contains("Paper A"));
        assert!(output.contains("f(x) = x"));
        assert!(output.contains("[invariant] output finite"));
        assert!(output.contains("FALSIFY-001"));
        assert!(output.contains("KANI-001"));
        assert!(output.contains("Verification ladder"));
    }

    #[test]
    fn explain_with_preconditions() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Pre/post test"
equations:
  eq:
    formula: "f(x) = x + 1"
    preconditions:
      - "x > 0"
    postconditions:
      - "ret > 1"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = explain_contract(&contract, "prepost-v1", None);
        assert!(output.contains("Preconditions (caller must guarantee)"));
        assert!(output.contains("x > 0"));
        assert!(output.contains("Postconditions (kernel guarantees)"));
        assert!(output.contains("ret > 1"));
    }

    #[test]
    fn explain_with_lean_proof() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Lean test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
    lean:
      theorem: test_theorem
      module: Test.Module
      status: proved
      depends_on: [dep1]
      notes: "Proved over reals"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = explain_contract(&contract, "lean-v1", None);
        assert!(output.contains("Test.Module.test_theorem (proved)"));
        assert!(output.contains("Depends: dep1"));
        assert!(output.contains("Note: Proved over reals"));
    }

    #[test]
    fn obligation_pattern_coverage() {
        // Verify all obligation types have a pattern
        let types = [
            ObligationType::Invariant,
            ObligationType::Equivalence,
            ObligationType::Bound,
            ObligationType::Monotonicity,
            ObligationType::Idempotency,
            ObligationType::Linearity,
            ObligationType::Symmetry,
            ObligationType::Associativity,
            ObligationType::Conservation,
            ObligationType::Ordering,
            ObligationType::Completeness,
            ObligationType::Soundness,
            ObligationType::Involution,
            ObligationType::Determinism,
            ObligationType::Roundtrip,
            ObligationType::StateMachine,
            ObligationType::Classification,
            ObligationType::Independence,
            ObligationType::Termination,
        ];
        for t in types {
            let pattern = obligation_pattern(t);
            assert!(!pattern.is_empty(), "empty pattern for {t}");
        }
    }
}
