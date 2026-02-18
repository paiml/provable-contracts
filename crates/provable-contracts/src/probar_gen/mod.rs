//! Probar property-test generator — Phase 5 of the pipeline.
//!
//! Generates probar property-based test source code from
//! the `falsification_tests` section of YAML contracts.

use crate::schema::Contract;

/// Generate probar property-based tests from a contract.
///
/// Each `falsification_tests` entry becomes a probar property
/// test that exercises the kernel with random inputs and checks
/// the falsifiable prediction.
pub fn generate_probar_tests(contract: &Contract) -> String {
    if contract.falsification_tests.is_empty() {
        return String::from(
            "// No probar tests defined in this contract.\n",
        );
    }

    let mut out = String::new();

    out.push_str("#[cfg(test)]\nmod probar_tests {\n");
    out.push_str("    use super::*;\n\n");

    for test in &contract.falsification_tests {
        // Doc comment
        out.push_str(&format!(
            "    /// {}: {}\\n",
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

        let fn_name = test
            .id
            .to_lowercase()
            .replace('-', "_");

        out.push_str(&format!(
            "    #[test]\n    fn prop_{fn_name}() {{\n"
        ));
        out.push_str(&format!(
            "        todo!(\"Implement probar property test \
                     for {}\")\n",
            test.id
        ));
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
    fn generate_probar_test_stubs() {
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
        assert!(code.contains("todo!"));
    }
}
