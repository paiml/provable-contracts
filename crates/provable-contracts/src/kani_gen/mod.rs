//! Kani harness generator — Phase 6 of the pipeline.
//!
//! Generates `#[kani::proof]` harnesses from the `kani_harnesses`
//! section of YAML contracts.

use crate::schema::{Contract, KaniStrategy};

/// Generate Kani proof harness source code from a contract.
///
/// Each `kani_harnesses` entry becomes a `#[kani::proof]` function
/// inside a `#[cfg(kani)] mod verification { ... }` block.
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
        // Doc comment
        out.push_str(&format!(
            "    /// {}: {}\n",
            harness.id,
            harness.property.as_deref().unwrap_or(&harness.obligation)
        ));
        out.push_str(&format!(
            "    /// Obligation: {}\n",
            harness.obligation
        ));
        if let Some(strategy) = harness.strategy {
            out.push_str(&format!(
                "    /// Strategy: {strategy}\n"
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
        let fn_name = harness
            .harness
            .as_deref()
            .unwrap_or("");
        let fn_name = if fn_name.is_empty() {
            harness.id.to_lowercase().replace('-', "_")
        } else {
            fn_name.to_string()
        };

        out.push_str(&format!(
            "    fn {fn_name}() {{\n"
        ));

        // Body based on strategy
        match harness.strategy {
            Some(KaniStrategy::Exhaustive) => {
                out.push_str(
                    "        // Strategy: exhaustive — \
                     exact verification (integer arithmetic)\n",
                );
                out.push_str(
                    "        todo!(\"Implement exhaustive \
                     Kani harness\")\n",
                );
            }
            Some(KaniStrategy::StubFloat) => {
                out.push_str(
                    "        // Strategy: stub_float — \
                     stub transcendentals, verify structural properties\n",
                );
                out.push_str(
                    "        todo!(\"Implement stub_float \
                     Kani harness\")\n",
                );
            }
            Some(KaniStrategy::Compositional) => {
                out.push_str(
                    "        // Strategy: compositional — \
                     use #[kani::stub_verified] for sub-kernels\n",
                );
                out.push_str(
                    "        todo!(\"Implement compositional \
                     Kani harness\")\n",
                );
            }
            None => {
                out.push_str(
                    "        todo!(\"Implement Kani harness\")\n",
                );
            }
        }

        out.push_str("    }\n\n");
    }

    out.push_str("}\n");
    out
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
    }
}
