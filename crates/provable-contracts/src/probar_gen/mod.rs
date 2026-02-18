//! Probar property-test generator — Phase 5 of the pipeline.
//!
//! Generates property-based test source code from both the
//! `falsification_tests` and `proof_obligations` sections of
//! YAML contracts. Maps obligation types to test patterns:
//!
//! | Obligation | Pattern |
//! |---|---|
//! | invariant | `#[probar::property]` with random inputs |
//! | equivalence | comparison of two implementations |
//! | bound | range checking on all outputs |
//! | monotonicity | ordered pair metamorphic test |
//! | idempotency | `f(f(x)) == f(x)` |
//! | linearity | `f(αx) == α·f(x)` metamorphic |
//! | symmetry | permutation invariance |
//! | associativity | `(a⊕b)⊕c == a⊕(b⊕c)` |
//! | conservation | `Q(before) == Q(after)` |

use crate::schema::{Contract, ObligationType, ProofObligation};

/// Generate probar property-based tests from a contract.
///
/// Produces two sections:
/// 1. Obligation-derived property tests (from `proof_obligations`)
/// 2. Falsification test stubs (from `falsification_tests`)
pub fn generate_probar_tests(contract: &Contract) -> String {
    if contract.proof_obligations.is_empty()
        && contract.falsification_tests.is_empty()
    {
        return String::from(
            "// No probar tests defined in this contract.\n",
        );
    }

    let mut out = String::new();

    out.push_str("#[cfg(test)]\nmod probar_tests {\n");
    out.push_str("    use super::*;\n\n");

    // Section 1: Property tests from proof obligations
    if !contract.proof_obligations.is_empty() {
        out.push_str(
            "    // === Property tests derived from proof \
             obligations ===\n\n",
        );
        for (i, ob) in
            contract.proof_obligations.iter().enumerate()
        {
            generate_obligation_test(&mut out, ob, i);
        }
    }

    // Section 2: Falsification test stubs
    if !contract.falsification_tests.is_empty() {
        out.push_str(
            "    // === Falsification test stubs ===\n\n",
        );
        for test in &contract.falsification_tests {
            generate_falsification_stub(&mut out, test);
        }
    }

    out.push_str("}\n");
    out
}

fn generate_obligation_test(
    out: &mut String,
    ob: &ProofObligation,
    index: usize,
) {
    let fn_name = obligation_fn_name(ob, index);
    let pattern = obligation_pattern(ob.obligation_type);

    // Doc comment
    out.push_str(&format!(
        "    /// Obligation: {} ({})\n",
        ob.property,
        ob.obligation_type
    ));
    if let Some(ref formal) = ob.formal {
        out.push_str(&format!("    /// Formal: {formal}\n"));
    }
    out.push_str(&format!("    /// Pattern: {pattern}\n"));
    if let Some(tol) = ob.tolerance {
        out.push_str(&format!("    /// Tolerance: {tol}\n"));
    }

    out.push_str("    #[test]\n");
    out.push_str(&format!("    fn {fn_name}() {{\n"));

    // Body based on obligation type
    match ob.obligation_type {
        ObligationType::Invariant => {
            generate_invariant_body(out, ob);
        }
        ObligationType::Equivalence => {
            generate_equivalence_body(out, ob);
        }
        ObligationType::Bound => {
            generate_bound_body(out, ob);
        }
        ObligationType::Monotonicity => {
            generate_monotonicity_body(out, ob);
        }
        ObligationType::Idempotency => {
            generate_idempotency_body(out, ob);
        }
        ObligationType::Linearity => {
            generate_linearity_body(out, ob);
        }
        ObligationType::Symmetry => {
            generate_symmetry_body(out, ob);
        }
        ObligationType::Associativity => {
            generate_associativity_body(out, ob);
        }
        ObligationType::Conservation => {
            generate_conservation_body(out, ob);
        }
    }

    out.push_str("    }\n\n");
}

fn obligation_fn_name(ob: &ProofObligation, index: usize) -> String {
    let base = ob
        .property
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric(), "_")
        .trim_matches('_')
        .to_string();
    if base.is_empty() {
        format!("prop_obligation_{index}")
    } else {
        format!("prop_{base}")
    }
}

fn obligation_pattern(ot: ObligationType) -> &'static str {
    match ot {
        ObligationType::Invariant => {
            "∀x ∈ Domain: P(f(x)) — property holds for all inputs"
        }
        ObligationType::Equivalence => {
            "∀x: |f(x) - g(x)| < ε — two implementations agree"
        }
        ObligationType::Bound => {
            "∀x: a ≤ f(x)_i ≤ b — output range bounded"
        }
        ObligationType::Monotonicity => {
            "x_i > x_j → f(x)_i > f(x)_j — order preserved"
        }
        ObligationType::Idempotency => {
            "f(f(x)) = f(x) — applying twice gives same result"
        }
        ObligationType::Linearity => {
            "f(αx) = α·f(x) — homogeneous scaling"
        }
        ObligationType::Symmetry => {
            "f(permute(x)) related to f(x) — permutation property"
        }
        ObligationType::Associativity => {
            "(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) — grouping invariant"
        }
        ObligationType::Conservation => {
            "Q(before) = Q(after) — conserved quantity"
        }
    }
}

fn tolerance_str(ob: &ProofObligation) -> String {
    match ob.tolerance {
        Some(tol) => format!("{tol:e}"),
        None => "1e-6".to_string(),
    }
}

fn generate_invariant_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: invariant — property holds \
         for all inputs.\n",
    );
    out.push_str(
        "        // Generate random inputs and check \
         postcondition.\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let output = kernel(&input);\n",
    );
    out.push_str(&format!(
        "            // assert!(postcondition(&output), \
         \"Invariant violated: {}\");\n",
        ob.property
    ));
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        let _ = {tol}; // tolerance\n"
    ));
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_equivalence_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: equivalence — two implementations \
         must agree.\n",
    );
    out.push_str(
        "        // Compare reference vs optimized within tolerance.\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let ref_out = reference_impl(&input);\n",
    );
    out.push_str(
        "            // let opt_out = optimized_impl(&input);\n",
    );
    out.push_str(&format!(
        "            // assert!(max_ulp_diff(&ref_out, &opt_out) \
         <= {tol});\n"
    ));
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_bound_body(out: &mut String, ob: &ProofObligation) {
    out.push_str(
        "        // Pattern: bound — all outputs within range.\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let output = kernel(&input);\n",
    );
    out.push_str(
        "            // for val in &output {\n",
    );
    out.push_str(
        "            //     assert!(lo <= *val && *val <= hi);\n",
    );
    out.push_str(
        "            // }\n",
    );
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_monotonicity_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    out.push_str(
        "        // Pattern: monotonicity — order preserved \
         in output.\n",
    );
    out.push_str(
        "        // Metamorphic: if x_i > x_j then f(x)_i > f(x)_j.\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let output = kernel(&input);\n",
    );
    out.push_str(
        "            // for i in 0..input.len() {\n",
    );
    out.push_str(
        "            //     for j in 0..input.len() {\n",
    );
    out.push_str(
        "            //         if input[i] > input[j] {\n",
    );
    out.push_str(
        "            //             assert!(output[i] > output[j]);\n",
    );
    out.push_str(
        "            //         }\n",
    );
    out.push_str(
        "            //     }\n",
    );
    out.push_str(
        "            // }\n",
    );
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_idempotency_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: idempotency — f(f(x)) == f(x).\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let once = kernel(&input);\n",
    );
    out.push_str(
        "            // let twice = kernel(&once);\n",
    );
    out.push_str(&format!(
        "            // assert!(max_diff(&once, &twice) < {tol});\n"
    ));
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_linearity_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: linearity — f(α·x) == α·f(x).\n",
    );
    out.push_str(
        "        // Metamorphic: scale input, compare scaled output.\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let alpha: f32 = random_nonzero();\n",
    );
    out.push_str(
        "            // let scaled: Vec<f32> = input.iter()\
         .map(|x| alpha * x).collect();\n",
    );
    out.push_str(
        "            // let f_scaled = kernel(&scaled);\n",
    );
    out.push_str(
        "            // let scaled_f: Vec<f32> = kernel(&input).iter()\
         .map(|x| alpha * x).collect();\n",
    );
    out.push_str(&format!(
        "            // assert!(max_diff(&f_scaled, &scaled_f) \
         < {tol});\n"
    ));
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_symmetry_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    out.push_str(
        "        // Pattern: symmetry — permutation invariance.\n",
    );
    out.push_str(
        "        // f(permute(x)) is related to permute(f(x)).\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let input = generate_random_input();\n",
    );
    out.push_str(
        "            // let perm = random_permutation(input.len());\n",
    );
    out.push_str(
        "            // let permuted_input = apply_perm(&input, &perm);\n",
    );
    out.push_str(
        "            // let out_orig = kernel(&input);\n",
    );
    out.push_str(
        "            // let out_perm = kernel(&permuted_input);\n",
    );
    out.push_str(
        "            // assert!(check_symmetry(&out_orig, \
         &out_perm, &perm));\n",
    );
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_associativity_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: associativity — \
         (a ⊕ b) ⊕ c == a ⊕ (b ⊕ c).\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let a = generate_random_input();\n",
    );
    out.push_str(
        "            // let b = generate_random_input();\n",
    );
    out.push_str(
        "            // let c = generate_random_input();\n",
    );
    out.push_str(
        "            // let left = op(&op(&a, &b), &c);\n",
    );
    out.push_str(
        "            // let right = op(&a, &op(&b, &c));\n",
    );
    out.push_str(&format!(
        "            // assert!(max_diff(&left, &right) < {tol});\n"
    ));
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_conservation_body(
    out: &mut String,
    ob: &ProofObligation,
) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: conservation — conserved \
         quantity unchanged.\n",
    );
    out.push_str(
        "        // Q(state_before) == Q(state_after).\n",
    );
    out.push_str(
        "        for _ in 0..1000 {\n",
    );
    out.push_str(
        "            // let state = generate_random_state();\n",
    );
    out.push_str(
        "            // let q_before = conserved_quantity(&state);\n",
    );
    out.push_str(
        "            // let new_state = transform(&state);\n",
    );
    out.push_str(
        "            // let q_after = conserved_quantity(&new_state);\n",
    );
    out.push_str(&format!(
        "            // assert!((q_before - q_after).abs() \
         < {tol});\n"
    ));
    out.push_str(
        "        }\n",
    );
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

fn generate_falsification_stub(
    out: &mut String,
    test: &crate::schema::FalsificationTest,
) {
    // Doc comment
    out.push_str(&format!(
        "    /// {}: {}\n",
        test.id, test.rule
    ));
    out.push_str(&format!(
        "    /// Prediction: {}\n",
        test.prediction
    ));
    out.push_str(&format!(
        "    /// If fails: {}\n",
        test.if_fails
    ));

    let fn_name = test.id.to_lowercase().replace('-', "_");

    out.push_str(&format!(
        "    #[test]\n    fn prop_{fn_name}() {{\n"
    ));

    if let Some(ref method) = test.test {
        out.push_str(&format!(
            "        // Method: {method}\n"
        ));
    }

    out.push_str(&format!(
        "        unimplemented!(\"Implement falsification \
         test for {}\")\n",
        test.id
    ));
    out.push_str("    }\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn generate_empty_probar_tests() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No tests"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("No probar tests"));
    }

    #[test]
    fn generate_falsification_stubs() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Probar test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "normalization"
    prediction: "sum(output) ≈ 1.0"
    if_fails: "missing max subtraction"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("mod probar_tests"));
        assert!(code.contains("fn prop_falsify_001()"));
        assert!(code.contains("Falsification test stubs"));
    }

    #[test]
    fn generate_invariant_property_test() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Invariant test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|sum(f(x)) - 1.0| < ε"
    tolerance: 1.0e-6
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("fn prop_output_sums_to_1()"));
        assert!(code.contains("invariant"));
        assert!(code.contains("postcondition"));
        assert!(code.contains("1e-6"));
    }

    #[test]
    fn generate_equivalence_property_test() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Equivalence"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: equivalence
    property: "SIMD matches scalar"
    tolerance: 8.0
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("fn prop_simd_matches_scalar()"));
        assert!(code.contains("equivalence"));
        assert!(code.contains("reference_impl"));
        assert!(code.contains("optimized_impl"));
    }

    #[test]
    fn generate_monotonicity_test() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Monotonicity"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: monotonicity
    property: "Order preservation"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("fn prop_order_preservation()"));
        assert!(code.contains("monotonicity"));
        assert!(code.contains("input[i] > input[j]"));
    }

    #[test]
    fn generate_all_obligation_types() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "All types"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "invariant test"
  - type: equivalence
    property: "equivalence test"
  - type: bound
    property: "bound test"
  - type: monotonicity
    property: "monotonicity test"
  - type: idempotency
    property: "idempotency test"
  - type: linearity
    property: "linearity test"
  - type: symmetry
    property: "symmetry test"
  - type: associativity
    property: "associativity test"
  - type: conservation
    property: "conservation test"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("invariant"));
        assert!(code.contains("equivalence"));
        assert!(code.contains("bound"));
        assert!(code.contains("monotonicity"));
        assert!(code.contains("idempotency"));
        assert!(code.contains("linearity"));
        assert!(code.contains("symmetry"));
        assert!(code.contains("associativity"));
        assert!(code.contains("conservation"));
        // 9 test functions
        assert_eq!(code.matches("#[test]").count(), 9);
    }

    #[test]
    fn generate_mixed_obligations_and_falsification() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Mixed"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is finite"
    if_fails: "overflow"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        // Both sections present
        assert!(code.contains(
            "Property tests derived from proof obligations"
        ));
        assert!(code.contains("Falsification test stubs"));
        assert_eq!(code.matches("#[test]").count(), 2);
    }
}
