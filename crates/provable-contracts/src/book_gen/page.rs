//! Per-contract mdBook page generator.
//!
//! Generates a markdown page for a single contract with KaTeX-rendered
//! equations and a Mermaid dependency graph.

use std::fmt::Write;

use crate::graph::DependencyGraph;
use crate::latex::math_to_latex;
use crate::schema::Contract;

/// Generate a complete mdBook page for a single contract.
///
/// `stem` is the contract filename without `.yaml` (e.g. "softmax-kernel-v1").
/// `dep_graph` is the full dependency graph (used to render this contract's neighborhood).
pub fn generate_contract_page(
    contract: &Contract,
    stem: &str,
    dep_graph: &DependencyGraph,
) -> String {
    let mut out = String::new();

    write_title(&mut out, contract, stem);
    write_references(&mut out, contract);
    write_dependencies(&mut out, contract);
    write_dependency_graph(&mut out, stem, dep_graph);
    write_equations(&mut out, contract);
    write_proof_obligations(&mut out, contract);
    write_kernel_phases(&mut out, contract);
    write_simd_dispatch(&mut out, contract);
    write_falsification_tests(&mut out, contract);
    write_kani_harnesses(&mut out, contract);
    write_qa_gate(&mut out, contract);

    out
}

fn write_title(out: &mut String, contract: &Contract, stem: &str) {
    let _ = writeln!(out, "# {}", contract.metadata.name_or(stem));
    let _ = writeln!(out);
    let _ = writeln!(out, "**Version:** {}", contract.metadata.version);
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", contract.metadata.description);
    let _ = writeln!(out);
}

fn write_references(out: &mut String, contract: &Contract) {
    if contract.metadata.references.is_empty() {
        return;
    }
    let _ = writeln!(out, "## References");
    let _ = writeln!(out);
    for r in &contract.metadata.references {
        let _ = writeln!(out, "- {r}");
    }
    let _ = writeln!(out);
}

fn write_dependencies(out: &mut String, contract: &Contract) {
    if contract.metadata.depends_on.is_empty() {
        return;
    }
    let _ = writeln!(out, "## Dependencies");
    let _ = writeln!(out);
    for dep in &contract.metadata.depends_on {
        let _ = writeln!(out, "- [{dep}]({dep}.md)");
    }
    let _ = writeln!(out);
}

fn write_dependency_graph(out: &mut String, stem: &str, graph: &DependencyGraph) {
    // Collect edges in this contract's neighborhood (direct deps + direct dependents)
    let deps = graph.edges.get(stem).cloned().unwrap_or_default();
    let dependents: Vec<&String> = graph
        .edges
        .iter()
        .filter(|(_, v)| v.iter().any(|d| d == stem))
        .map(|(k, _)| k)
        .collect();

    if deps.is_empty() && dependents.is_empty() {
        return;
    }

    let _ = writeln!(out, "## Dependency Graph");
    let _ = writeln!(out);
    let _ = writeln!(out, "```mermaid");
    let _ = writeln!(out, "graph LR");
    for dep in &deps {
        let _ = writeln!(
            out,
            "    {}[\"{stem}\"] --> {}[\"{dep}\"]",
            mermaid_id(stem),
            mermaid_id(dep)
        );
    }
    for dependent in &dependents {
        let _ = writeln!(
            out,
            "    {}[\"{dependent}\"] --> {}[\"{stem}\"]",
            mermaid_id(dependent),
            mermaid_id(stem)
        );
    }
    let _ = writeln!(out, "```");
    let _ = writeln!(out);
}

fn write_equations(out: &mut String, contract: &Contract) {
    if contract.equations.is_empty() {
        return;
    }
    let _ = writeln!(out, "## Equations");
    let _ = writeln!(out);

    for (id, eq) in &contract.equations {
        let _ = writeln!(out, "### {id}");
        let _ = writeln!(out);
        let _ = writeln!(out, "$$");
        let _ = writeln!(out, "{}", math_to_latex(&eq.formula));
        let _ = writeln!(out, "$$");
        let _ = writeln!(out);

        if let Some(ref dom) = eq.domain {
            let _ = writeln!(out, "**Domain:** ${}$", math_to_latex(dom));
            let _ = writeln!(out);
        }

        if let Some(ref cod) = eq.codomain {
            let _ = writeln!(out, "**Codomain:** ${}$", math_to_latex(cod));
            let _ = writeln!(out);
        }

        if !eq.invariants.is_empty() {
            let _ = writeln!(out, "**Invariants:**");
            let _ = writeln!(out);
            for inv in &eq.invariants {
                let _ = writeln!(out, "- ${}$", math_to_latex(inv));
            }
            let _ = writeln!(out);
        }
    }
}

fn write_proof_obligations(out: &mut String, contract: &Contract) {
    if contract.proof_obligations.is_empty() {
        return;
    }
    let _ = writeln!(out, "## Proof Obligations");
    let _ = writeln!(out);
    let _ = writeln!(out, "| # | Type | Property | Formal |");
    let _ = writeln!(out, "|---|------|----------|--------|");

    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        let formal = ob
            .formal
            .as_deref()
            .map(|f| format!("${}$", escape_pipe(&math_to_latex(f))))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            i + 1,
            ob.obligation_type,
            escape_pipe(&ob.property),
            formal
        );
    }
    let _ = writeln!(out);
}

fn write_kernel_phases(out: &mut String, contract: &Contract) {
    let Some(ref ks) = contract.kernel_structure else {
        return;
    };
    let _ = writeln!(out, "## Kernel Phases");
    let _ = writeln!(out);
    for (i, phase) in ks.phases.iter().enumerate() {
        let inv = phase
            .invariant
            .as_deref()
            .map(|s| format!(" — *{s}*"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "{}. **{}**: {}{}",
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
    let _ = writeln!(out, "## SIMD Dispatch");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Kernel | ISA | Target |");
    let _ = writeln!(out, "|--------|-----|--------|");

    for (kernel, dispatch) in &contract.simd_dispatch {
        for (isa, target) in dispatch {
            let _ = writeln!(out, "| {kernel} | {isa} | `{target}` |");
        }
    }
    let _ = writeln!(out);
}

fn write_falsification_tests(out: &mut String, contract: &Contract) {
    if contract.falsification_tests.is_empty() {
        return;
    }
    let _ = writeln!(out, "## Falsification Tests");
    let _ = writeln!(out);
    let _ = writeln!(out, "| ID | Rule | Prediction | If Fails |");
    let _ = writeln!(out, "|----|------|------------|----------|");

    for ft in &contract.falsification_tests {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            ft.id,
            escape_pipe(&ft.rule),
            escape_pipe(&ft.prediction),
            escape_pipe(&ft.if_fails)
        );
    }
    let _ = writeln!(out);
}

fn write_kani_harnesses(out: &mut String, contract: &Contract) {
    if contract.kani_harnesses.is_empty() {
        return;
    }
    let _ = writeln!(out, "## Kani Harnesses");
    let _ = writeln!(out);
    let _ = writeln!(out, "| ID | Obligation | Bound | Strategy |");
    let _ = writeln!(out, "|----|------------|-------|----------|");

    for kh in &contract.kani_harnesses {
        let bound = kh.bound.map_or_else(|| "-".to_string(), |b| b.to_string());
        let strategy = kh
            .strategy
            .map_or_else(|| "-".to_string(), |s| s.to_string());
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            kh.id, kh.obligation, bound, strategy
        );
    }
    let _ = writeln!(out);
}

fn write_qa_gate(out: &mut String, contract: &Contract) {
    let Some(ref qa) = contract.qa_gate else {
        return;
    };
    let _ = writeln!(out, "## QA Gate");
    let _ = writeln!(out);
    let _ = writeln!(out, "**{}** ({})", qa.name, qa.id);
    let _ = writeln!(out);
    if let Some(ref desc) = qa.description {
        let _ = writeln!(out, "{desc}");
        let _ = writeln!(out);
    }
    if !qa.checks.is_empty() {
        let _ = writeln!(out, "**Checks:** {}", qa.checks.join(", "));
        let _ = writeln!(out);
    }
    if let Some(ref criteria) = qa.pass_criteria {
        let _ = writeln!(out, "**Pass criteria:** {criteria}");
        let _ = writeln!(out);
    }
}

/// Escape pipe characters for markdown tables.
fn escape_pipe(s: &str) -> String {
    s.replace('|', "\\|")
}

/// Convert a contract stem to a valid Mermaid node ID (no hyphens).
fn mermaid_id(stem: &str) -> String {
    stem.replace('-', "_")
}

/// Extension trait for Metadata to provide a display name.
trait MetadataName {
    fn name_or(&self, stem: &str) -> String;
}

impl MetadataName for crate::schema::Metadata {
    fn name_or(&self, stem: &str) -> String {
        // Use description's first clause as title if it's descriptive enough,
        // otherwise fall back to stem
        if self.description.len() > 3 {
            // Use the stem as the heading — it's the canonical identifier
            stem.to_string()
        } else {
            stem.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::dependency_graph;
    use crate::schema::parse_contract_str;

    fn minimal_contract() -> Contract {
        parse_contract_str(
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
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
        )
        .unwrap()
    }

    #[test]
    fn generates_title_and_equations() {
        let c = minimal_contract();
        let refs = vec![("test-kernel-v1".to_string(), &c)];
        let graph = dependency_graph(&refs);
        let page = generate_contract_page(&c, "test-kernel-v1", &graph);

        assert!(page.contains("# test-kernel-v1"));
        assert!(page.contains("## Equations"));
        assert!(page.contains("$$"));
        assert!(page.contains("## Proof Obligations"));
        assert!(page.contains("## Falsification Tests"));
        assert!(page.contains("## Kani Harnesses"));
    }

    #[test]
    fn latex_rendering_in_equations() {
        let c = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  eq1:
    formula: "σ(x)_i = exp(x_i) / Σ_j exp(x_j)"
    domain: "x ∈ ℝ^n"
falsification_tests: []
"#,
        )
        .unwrap();
        let refs = vec![("test".to_string(), &c)];
        let graph = dependency_graph(&refs);
        let page = generate_contract_page(&c, "test", &graph);

        assert!(page.contains("\\sigma"));
        assert!(page.contains("\\exp"));
        assert!(page.contains("\\in"));
        assert!(page.contains("\\mathbb{R}"));
    }

    #[test]
    fn dependency_graph_rendered() {
        let a = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "A"
  depends_on: ["b"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();
        let b = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "B"
equations:
  g:
    formula: "g(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();
        let refs = vec![("a".to_string(), &a), ("b".to_string(), &b)];
        let graph = dependency_graph(&refs);
        let page = generate_contract_page(&a, "a", &graph);

        assert!(page.contains("```mermaid"));
        assert!(page.contains("## Dependencies"));
        assert!(page.contains("[b](b.md)"));
    }

    #[test]
    fn optional_sections_omitted_when_empty() {
        let c = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Minimal"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();
        let refs = vec![("minimal".to_string(), &c)];
        let graph = dependency_graph(&refs);
        let page = generate_contract_page(&c, "minimal", &graph);

        assert!(!page.contains("## Kernel Phases"));
        assert!(!page.contains("## SIMD Dispatch"));
        assert!(!page.contains("## QA Gate"));
        assert!(!page.contains("## Dependency Graph"));
    }

    #[test]
    fn pipe_escaped_in_tables() {
        let c = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
    formal: "|f(x)| < 1"
falsification_tests: []
"#,
        )
        .unwrap();
        let refs = vec![("test".to_string(), &c)];
        let graph = dependency_graph(&refs);
        let page = generate_contract_page(&c, "test", &graph);

        // The pipe in formal should be escaped
        assert!(page.contains("\\|"));
    }
}
