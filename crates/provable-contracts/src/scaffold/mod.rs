//! Scaffold generator — Phase 3 of the pipeline.
//!
//! Generates Rust trait definitions and failing test stubs
//! from parsed YAML contracts.

use crate::schema::Contract;

/// Generate a Rust trait definition from a contract.
///
/// Each equation becomes a method. Each proof obligation
/// becomes a doc-comment with INVARIANT/REQUIRES prefix.
pub fn generate_trait(contract: &Contract) -> String {
    let mut out = String::new();
    let desc = &contract.metadata.description;

    // Header
    out.push_str(&format!(
        "/// Contract: {} v{}\n",
        desc, contract.metadata.version
    ));
    for r in &contract.metadata.references {
        out.push_str(&format!("/// Paper: {r}\n"));
    }
    out.push_str("pub trait KernelContract {\n");

    // One method per equation
    for (name, eq) in &contract.equations {
        out.push_str(&format!("    /// {}\n", eq.formula));
        if let Some(ref domain) = eq.domain {
            out.push_str(&format!("    /// Domain: {domain}\n"));
        }
        if let Some(ref codomain) = eq.codomain {
            out.push_str(&format!("    /// Codomain: {codomain}\n"));
        }
        for inv in &eq.invariants {
            out.push_str(&format!("    /// INVARIANT: {inv}\n"));
        }
        // Add proof obligations for this equation
        for ob in &contract.proof_obligations {
            out.push_str(&format!(
                "    /// {} ({}): {}\n",
                ob.obligation_type.to_string().to_uppercase(),
                ob.property,
                ob.formal.as_deref().unwrap_or("")
            ));
        }
        out.push_str(&format!(
            "    fn {name}(&self, input: &[f32], output: &mut [f32]);\n"
        ));
    }

    out.push_str("}\n");
    out
}

/// Generate failing contract test stubs from a contract.
///
/// Each falsification test becomes a `#[test]` with `todo!()`.
pub fn generate_contract_tests(contract: &Contract) -> String {
    let mut out = String::new();

    out.push_str("#[cfg(test)]\nmod contract_tests {\n");
    out.push_str("    use super::*;\n\n");

    for test in &contract.falsification_tests {
        out.push_str(&format!("    /// {}: {}\n", test.id, test.rule));
        out.push_str(&format!("    /// Prediction: {}\n", test.prediction));
        out.push_str(&format!("    /// If fails: {}\n", test.if_fails));
        let fn_name = test.id.to_lowercase().replace('-', "_");
        out.push_str(&format!("    #[test]\n    fn {fn_name}() {{\n"));
        out.push_str(&format!(
            "        todo!(\"Implementation not yet written — \
                     {} MUST fail\")\n",
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

    fn sample_contract() -> Contract {
        parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test kernel"
  references:
    - "Paper (2024)"
equations:
  softmax:
    formula: "σ(x) = exp(x-max) / Σexp(x-max)"
    domain: "ℝ^n"
    codomain: "(0,1)^n"
    invariants:
      - "sum(output) = 1.0"
proof_obligations:
  - type: invariant
    property: "normalization"
    formal: "|sum(σ(x)) - 1.0| < ε"
falsification_tests:
  - id: FALSIFY-SM-001
    rule: "normalization"
    prediction: "sum(output) ≈ 1.0"
    if_fails: "missing max subtraction"
  - id: FALSIFY-SM-002
    rule: "positivity"
    prediction: "output > 0"
    if_fails: "exp underflow"
"#,
        )
        .unwrap()
    }

    #[test]
    fn generate_trait_includes_equations() {
        let contract = sample_contract();
        let code = generate_trait(&contract);
        assert!(code.contains("pub trait KernelContract"));
        assert!(code.contains("fn softmax"));
        assert!(code.contains("INVARIANT: sum(output) = 1.0"));
    }

    #[test]
    fn generate_tests_creates_stubs() {
        let contract = sample_contract();
        let code = generate_contract_tests(&contract);
        assert!(code.contains("fn falsify_sm_001()"));
        assert!(code.contains("fn falsify_sm_002()"));
        assert!(code.contains("todo!"));
    }

    #[test]
    fn generate_tests_includes_predictions() {
        let contract = sample_contract();
        let code = generate_contract_tests(&contract);
        assert!(code.contains("sum(output) ≈ 1.0"));
        assert!(code.contains("missing max subtraction"));
    }
}
