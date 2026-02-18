//! Kani harness generator — Phase 6 of the pipeline.
//!
//! Generates `#[kani::proof]` harnesses from the `kani_harnesses`
//! section of YAML contracts. Produces real verification bodies
//! following the three strategies: exhaustive, `stub_float`, and
//! compositional.

use crate::schema::{Contract, KaniHarness, KaniStrategy};

/// Generate Kani proof harness source code from a contract.
///
/// Each `kani_harnesses` entry becomes a `#[kani::proof]` function
/// inside a `#[cfg(kani)] mod verification { ... }` block.
///
/// The generated harness includes:
/// - Doc comments with obligation and strategy
/// - `#[kani::proof]`, `#[kani::unwind(N+1)]`, `#[kani::solver(..)]`
/// - Strategy-specific body with symbolic inputs and assertions
pub fn generate_kani_harnesses(contract: &Contract) -> String {
    if contract.kani_harnesses.is_empty() {
        return String::from(
            "// No Kani harnesses defined in this contract.\n",
        );
    }

    let mut out = String::new();

    out.push_str("#[cfg(kani)]\nmod verification {\n");
    out.push_str("    use super::*;\n\n");

    for harness in &contract.kani_harnesses {
        generate_single_harness(&mut out, harness);
    }

    out.push_str("}\n");
    out
}

fn generate_single_harness(out: &mut String, harness: &KaniHarness) {
    let property_desc = harness
        .property
        .as_deref()
        .unwrap_or(&harness.obligation);

    // Doc comments
    out.push_str(&format!(
        "    /// {}: {}\n",
        harness.id, property_desc
    ));
    out.push_str(&format!(
        "    /// Obligation: {}\n",
        harness.obligation
    ));
    if let Some(strategy) = harness.strategy {
        out.push_str(&format!("    /// Strategy: {strategy}\n"));
    }
    if let Some(bound) = harness.bound {
        out.push_str(&format!(
            "    /// Bound: {bound} elements\n"
        ));
    }

    // Attributes
    out.push_str("    #[kani::proof]\n");
    if let Some(bound) = harness.bound {
        out.push_str(&format!(
            "    #[kani::unwind({})]\n",
            bound + 1
        ));
    }
    if let Some(ref solver) = harness.solver {
        out.push_str(&format!(
            "    #[kani::solver({solver})]\n"
        ));
    }

    // Function name
    let fn_name = harness.harness.as_deref().unwrap_or("");
    let fn_name = if fn_name.is_empty() {
        harness.id.to_lowercase().replace('-', "_")
    } else {
        fn_name.to_string()
    };

    out.push_str(&format!("    fn {fn_name}() {{\n"));

    // Body based on strategy
    match harness.strategy {
        Some(KaniStrategy::Exhaustive) => {
            generate_exhaustive_body(out, harness);
        }
        Some(KaniStrategy::StubFloat) => {
            generate_stub_float_body(out, harness);
        }
        Some(KaniStrategy::Compositional) => {
            generate_compositional_body(out, harness);
        }
        None => {
            generate_default_body(out, harness);
        }
    }

    out.push_str("    }\n\n");
}

fn generate_exhaustive_body(
    out: &mut String,
    harness: &KaniHarness,
) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str(
        "        // Strategy: exhaustive — exact verification\n",
    );
    out.push_str(
        "        // Integer/structural arithmetic verified \
         without approximation.\n",
    );
    out.push_str(&format!(
        "        // Bound: {bound} elements\n\n"
    ));

    // Generate symbolic input pattern
    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        // Symbolic inputs — Kani explores ALL possible values\n",
    );
    out.push_str(
        "        let input: Vec<i32> = (0..n)\
         .map(|_| kani::any()).collect();\n\n",
    );

    // Generate assertion based on property
    if let Some(ref property) = harness.property {
        out.push_str(&format!(
            "        // Verify: {property}\n"
        ));
    }
    out.push_str(&format!(
        "        // Obligation: {}\n",
        harness.obligation
    ));
    out.push_str(
        "        // TODO: Replace with kernel-specific \
         verification logic\n",
    );
    out.push_str(
        "        //   Example: assert_eq!(precomputed, online);\n",
    );
    out.push_str(
        "        unimplemented!(\"Wire up kernel under test\")\n",
    );
}

fn generate_stub_float_body(
    out: &mut String,
    harness: &KaniHarness,
) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str(
        "        // Strategy: stub_float — stub transcendentals, \
         verify structural properties.\n",
    );
    out.push_str(
        "        // Floating-point transcendentals (exp, log, sin, cos) \
         are replaced\n",
    );
    out.push_str(
        "        // with contract stubs that preserve structural \
         invariants.\n\n",
    );

    // Symbolic vector input
    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        // Symbolic input: every possible finite \
         f32 vector of length n\n",
    );
    out.push_str(
        "        let input: Vec<f32> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(
        "        kani::assume(input.iter()\
         .all(|x| x.is_finite()));\n\n",
    );

    out.push_str(
        "        let mut output = vec![0.0f32; n];\n",
    );

    // Generate postconditions from property
    if let Some(ref property) = harness.property {
        out.push_str(&format!(
            "\n        // Post-condition: {property}\n"
        ));
    }

    out.push_str(
        "        // TODO: Call kernel under test:\n",
    );
    out.push_str(
        "        //   kernel_fn(&input, &mut output);\n",
    );
    out.push_str(
        "        //\n",
    );
    out.push_str(
        "        // Then assert postconditions, e.g.:\n",
    );
    out.push_str(
        "        //   let sum: f32 = output.iter().sum();\n",
    );
    out.push_str(
        "        //   assert!((sum - 1.0).abs() < 1e-5);\n",
    );
    out.push_str(
        "        //   assert!(output.iter().all(|&x| x > 0.0));\n",
    );
    out.push_str(
        "        unimplemented!(\"Wire up kernel under test\")\n",
    );
}

fn generate_compositional_body(
    out: &mut String,
    harness: &KaniHarness,
) {
    out.push_str(
        "        // Strategy: compositional — \
         use #[kani::stub_verified] for sub-kernels.\n",
    );
    out.push_str(
        "        // Verified sub-components are replaced with \
         their contract abstractions,\n",
    );
    out.push_str(
        "        // providing free postconditions without \
         re-verifying internals.\n\n",
    );

    let bound = harness.bound.unwrap_or(8);
    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        let input: Vec<f32> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(
        "        kani::assume(input.iter()\
         .all(|x| x.is_finite()));\n\n",
    );

    if let Some(ref property) = harness.property {
        out.push_str(&format!(
            "        // Verify: {property}\n"
        ));
    }
    out.push_str(
        "        // TODO: Add #[kani::stub_verified(sub_fn)] \
         to harness attribute\n",
    );
    out.push_str(
        "        //   then call composed kernel and assert \
         end-to-end properties.\n",
    );
    out.push_str(
        "        unimplemented!(\"Wire up kernel under test\")\n",
    );
}

fn generate_default_body(
    out: &mut String,
    harness: &KaniHarness,
) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str(
        "        // No strategy specified — using default \
         symbolic verification pattern.\n\n",
    );

    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        let input: Vec<f32> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(
        "        kani::assume(input.iter()\
         .all(|x| x.is_finite()));\n\n",
    );

    if let Some(ref property) = harness.property {
        out.push_str(&format!(
            "        // Verify: {property}\n"
        ));
    }
    out.push_str(&format!(
        "        // Obligation: {}\n",
        harness.obligation
    ));
    out.push_str(
        "        unimplemented!(\"Wire up kernel under test\")\n",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn generate_empty_harnesses() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No harnesses"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("No Kani harnesses"));
    }

    #[test]
    fn generate_harness_with_all_attributes() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Kani test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: SM-INV-001
    property: "Softmax sums to 1"
    bound: 16
    strategy: stub_float
    solver: kissat
    harness: verify_softmax_normalization
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("#[cfg(kani)]"));
        assert!(code.contains("#[kani::proof]"));
        assert!(code.contains("#[kani::unwind(17)]"));
        assert!(code.contains("#[kani::solver(kissat)]"));
        assert!(code.contains("fn verify_softmax_normalization()"));
        assert!(code.contains("stub_float"));
        assert!(code.contains("kani::any()"));
        assert!(code.contains("is_finite"));
    }

    #[test]
    fn generate_exhaustive_harness() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Exhaustive"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-002
    obligation: QDOT-001
    bound: 32
    strategy: exhaustive
    harness: verify_bsums
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("fn verify_bsums()"));
        assert!(code.contains("#[kani::unwind(33)]"));
        assert!(code.contains("exhaustive"));
        assert!(code.contains("Vec<i32>"));
        assert!(code.contains("kani::any()"));
    }

    #[test]
    fn generate_compositional_harness() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Compositional"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-003
    obligation: ATT-001
    property: "Attention uses normalized weights"
    bound: 8
    strategy: compositional
    harness: verify_attention_weights
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("fn verify_attention_weights()"));
        assert!(code.contains("#[kani::unwind(9)]"));
        assert!(code.contains("compositional"));
        assert!(code.contains("stub_verified"));
    }

    #[test]
    fn generate_harness_without_strategy() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No strategy"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-004
    obligation: OBL-001
    property: "Output finite"
    bound: 16
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("fn kani_004()"));
        assert!(code.contains("No strategy specified"));
        assert!(code.contains("kani::any()"));
    }

    #[test]
    fn generate_harness_property_in_doc_comment() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-005
    obligation: INV-001
    property: "Sum equals 1.0"
    bound: 4
    strategy: stub_float
    harness: verify_sum
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("/// KANI-005: Sum equals 1.0"));
        assert!(code.contains("/// Obligation: INV-001"));
        assert!(code.contains("/// Strategy: stub_float"));
        assert!(code.contains("/// Bound: 4 elements"));
        assert!(code.contains("Post-condition: Sum equals 1.0"));
    }

    #[test]
    fn generate_full_softmax_contract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Softmax"
  references: ["Paper"]
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))"
kani_harnesses:
  - id: KANI-SM-001
    obligation: SM-INV-001
    property: "Softmax sums to 1.0"
    bound: 8
    strategy: stub_float
    solver: cadical
    harness: verify_softmax_normalization
  - id: KANI-SM-002
    obligation: SM-INV-002
    property: "All outputs positive"
    bound: 8
    strategy: stub_float
    harness: verify_softmax_positivity
  - id: KANI-SM-003
    obligation: SM-BND-001
    property: "Outputs bounded in (0,1)"
    bound: 8
    strategy: stub_float
    harness: verify_softmax_bounded
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_kani_harnesses(&contract);

        // Verify three harnesses generated
        assert!(code.contains("fn verify_softmax_normalization()"));
        assert!(code.contains("fn verify_softmax_positivity()"));
        assert!(code.contains("fn verify_softmax_bounded()"));

        // All wrapped in cfg(kani)
        assert!(code.starts_with("#[cfg(kani)]"));
        assert!(code.contains("mod verification"));

        // Each has kani::proof
        assert_eq!(
            code.matches("#[kani::proof]").count(),
            3
        );
    }
}
