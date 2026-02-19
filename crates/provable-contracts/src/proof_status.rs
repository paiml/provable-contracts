//! Proof status report — cross-contract proof level assessment.
//!
//! Computes a hierarchical proof level (L1–L5) for each contract and
//! aggregates them into kernel equivalence classes that mirror the
//! `KernelOp` classification from apr-model-qa-playbook.
//!
//! Output is consumed by `pv proof-status` (text/JSON) and by the
//! playbook's `ProofBonus` MQS integration.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::Contract;

// ── Proof level hierarchy ─────────────────────────────────────────

/// Hierarchical proof assurance level.
///
/// Each level subsumes the ones below it:
/// - **L1** — Contract YAML exists with equations
/// - **L2** — Property tested (falsification tests cover obligations)
/// - **L3** — Kani bounded-model-checked
/// - **L4** — Lean 4 theorem proved
/// - **L5** — L4 + all bindings verified as implemented
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProofLevel {
    L1,
    L2,
    L3,
    L4,
    L5,
}

impl fmt::Display for ProofLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
            Self::L4 => "L4",
            Self::L5 => "L5",
        };
        write!(f, "{s}")
    }
}

// ── Per-contract status ───────────────────────────────────────────

/// Proof status for a single contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractProofStatus {
    pub stem: String,
    pub proof_level: ProofLevel,
    pub obligations: u32,
    pub falsification_tests: u32,
    pub kani_harnesses: u32,
    pub lean_proved: u32,
    pub bindings_implemented: u32,
    pub bindings_total: u32,
}

// ── Kernel class summary ──────────────────────────────────────────

/// Summary of proof status for a kernel equivalence class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelClassSummary {
    pub label: String,
    pub description: String,
    pub contract_stems: Vec<String>,
    pub min_proof_level: ProofLevel,
    pub all_bound: bool,
}

// ── Full report ───────────────────────────────────────────────────

/// Top-level proof status report, serializable to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatusReport {
    pub schema_version: String,
    pub timestamp: String,
    pub contracts: Vec<ContractProofStatus>,
    pub kernel_classes: Vec<KernelClassSummary>,
    pub totals: ProofStatusTotals,
}

/// Aggregate totals across all contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatusTotals {
    pub contracts: u32,
    pub obligations: u32,
    pub falsification_tests: u32,
    pub kani_harnesses: u32,
    pub lean_proved: u32,
    pub bindings_implemented: u32,
    pub bindings_total: u32,
}

// ── Kernel class → contract stem mapping ──────────────────────────

/// Static mapping from kernel equivalence class to contract stems.
///
/// Mirrors the `KernelOp` classification from `apr-model-qa-playbook`:
/// - **A** — GQA + `RMSNorm` + `SiLU` + `SwiGLU` + `RoPE` (Llama/Mistral)
/// - **B** — MHA + `LayerNorm` + GELU + `AbsPos` (GPT-2/BERT)
/// - **C** — MHA + `LayerNorm` + GELU + `ALiBi` (BLOOM/MPT)
/// - **D** — `LayerNorm` + GELU + `SiLU` + GQA (Gemma)
/// - **E** — `RMSNorm` + `SwiGLU` + GQA (Qwen)
fn kernel_class_map() -> Vec<(&'static str, &'static str, &'static [&'static str])> {
    vec![
        (
            "A",
            "GQA+RMSNorm+SiLU+SwiGLU+RoPE",
            &[
                "rmsnorm-kernel-v1",
                "silu-kernel-v1",
                "swiglu-kernel-v1",
                "rope-kernel-v1",
                "gqa-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
        (
            "B",
            "MHA+LayerNorm+GELU+AbsPos",
            &[
                "layernorm-kernel-v1",
                "gelu-kernel-v1",
                "attention-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
                "absolute-position-v1",
            ],
        ),
        (
            "C",
            "MHA+LayerNorm+GELU+ALiBi",
            &[
                "layernorm-kernel-v1",
                "gelu-kernel-v1",
                "attention-kernel-v1",
                "softmax-kernel-v1",
                "alibi-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
        (
            "D",
            "LayerNorm+GELU+SiLU+GQA",
            &[
                "layernorm-kernel-v1",
                "gelu-kernel-v1",
                "silu-kernel-v1",
                "gqa-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
        (
            "E",
            "RMSNorm+SwiGLU+GQA",
            &[
                "rmsnorm-kernel-v1",
                "swiglu-kernel-v1",
                "gqa-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
    ]
}

// ── Core computation ──────────────────────────────────────────────

/// Compute the proof level for a single contract.
///
/// Derivation rules (highest matching level wins):
/// - **L5**: all Lean proved AND all bindings implemented
/// - **L4**: all Lean proved (`verification_summary.l4_lean_proved == total`)
/// - **L3**: has Kani harnesses AND falsification tests cover obligations
/// - **L2**: falsification tests count >= obligations count
/// - **L1**: contract exists with equations
#[allow(clippy::cast_possible_truncation)]
pub fn compute_proof_level(contract: &Contract, binding_status: Option<(u32, u32)>) -> ProofLevel {
    let total_obligations = contract.proof_obligations.len() as u32;
    let ft_count = contract.falsification_tests.len() as u32;
    let kani_count = contract.kani_harnesses.len() as u32;

    // Check L4/L5: Lean proved
    if let Some(ref vs) = contract.verification_summary {
        if vs.total_obligations > 0 && vs.l4_lean_proved == vs.total_obligations {
            // All Lean proved — check bindings for L5
            if let Some((implemented, total)) = binding_status {
                if total > 0 && implemented == total {
                    return ProofLevel::L5;
                }
            }
            return ProofLevel::L4;
        }
    }

    // Check L3: Kani + falsification
    if kani_count > 0 && ft_count >= total_obligations && total_obligations > 0 {
        return ProofLevel::L3;
    }

    // Check L2: falsification tests cover obligations
    if ft_count >= total_obligations && total_obligations > 0 {
        return ProofLevel::L2;
    }

    // L1: contract exists with equations
    ProofLevel::L1
}

/// Build a complete proof status report.
///
/// `contracts` is a list of `(stem, &Contract)` pairs.
/// `binding` is an optional binding registry for binding coverage.
/// `include_classes` controls whether kernel class summaries are generated.
#[allow(clippy::cast_possible_truncation)]
pub fn proof_status_report(
    contracts: &[(String, &Contract)],
    binding: Option<&BindingRegistry>,
    include_classes: bool,
) -> ProofStatusReport {
    let mut statuses = Vec::new();
    let mut totals = ProofStatusTotals {
        contracts: contracts.len() as u32,
        obligations: 0,
        falsification_tests: 0,
        kani_harnesses: 0,
        lean_proved: 0,
        bindings_implemented: 0,
        bindings_total: 0,
    };

    for (stem, contract) in contracts {
        let contract_file = format!("{stem}.yaml");

        let obligations = contract.proof_obligations.len() as u32;
        let ft_count = contract.falsification_tests.len() as u32;
        let kani_count = contract.kani_harnesses.len() as u32;
        let lean_proved = contract
            .verification_summary
            .as_ref()
            .map_or(0, |vs| vs.l4_lean_proved);

        // Count bindings for this contract
        let (b_impl, b_total) = if let Some(reg) = binding {
            count_bindings(&contract_file, contract, reg)
        } else {
            (0, contract.equations.len() as u32)
        };

        let binding_status = if binding.is_some() {
            Some((b_impl, b_total))
        } else {
            None
        };

        let proof_level = compute_proof_level(contract, binding_status);

        totals.obligations += obligations;
        totals.falsification_tests += ft_count;
        totals.kani_harnesses += kani_count;
        totals.lean_proved += lean_proved;
        totals.bindings_implemented += b_impl;
        totals.bindings_total += b_total;

        statuses.push(ContractProofStatus {
            stem: stem.clone(),
            proof_level,
            obligations,
            falsification_tests: ft_count,
            kani_harnesses: kani_count,
            lean_proved,
            bindings_implemented: b_impl,
            bindings_total: b_total,
        });
    }

    // Build kernel class summaries
    let kernel_classes = if include_classes {
        build_kernel_classes(&statuses)
    } else {
        Vec::new()
    };

    let timestamp = current_timestamp();

    ProofStatusReport {
        schema_version: "1.0.0".to_string(),
        timestamp,
        contracts: statuses,
        kernel_classes,
        totals,
    }
}

/// Format a proof status report as human-readable text.
pub fn format_text(report: &ProofStatusReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Proof Status ({} contracts)\n\n",
        report.totals.contracts
    ));

    out.push_str(&format!(
        "  {:<35} {:>5} {:>6} {:>5} {:>4} {:>4} {:>9}\n",
        "Contract", "Level", "Obligs", "Tests", "Kani", "Lean", "Bindings"
    ));
    out.push_str(&format!("  {}\n", "─".repeat(72)));

    for c in &report.contracts {
        out.push_str(&format!(
            "  {:<35} {:>5} {:>6} {:>5} {:>4} {:>4} {:>4}/{:<4}\n",
            truncate(&c.stem, 35),
            c.proof_level,
            c.obligations,
            c.falsification_tests,
            c.kani_harnesses,
            c.lean_proved,
            c.bindings_implemented,
            c.bindings_total,
        ));
    }

    if !report.kernel_classes.is_empty() {
        out.push_str("\nKernel Classes:\n");
        for kc in &report.kernel_classes {
            let bound_str = if kc.all_bound { "all bound" } else { "gaps" };
            out.push_str(&format!(
                "  {} ({}): min={}, {} contracts, {}\n",
                kc.label,
                kc.description,
                kc.min_proof_level,
                kc.contract_stems.len(),
                bound_str,
            ));
        }
    }

    out.push_str(&format!(
        "\nTotals: {} obligations, {} tests, {} kani, {} lean proved, {}/{} bound\n",
        report.totals.obligations,
        report.totals.falsification_tests,
        report.totals.kani_harnesses,
        report.totals.lean_proved,
        report.totals.bindings_implemented,
        report.totals.bindings_total,
    ));

    out
}

// ── Internal helpers ──────────────────────────────────────────────

#[allow(clippy::cast_possible_truncation)]
fn count_bindings(
    contract_file: &str,
    contract: &Contract,
    binding: &BindingRegistry,
) -> (u32, u32) {
    let total = contract.equations.len() as u32;
    let implemented = binding
        .bindings
        .iter()
        .filter(|b| b.contract == contract_file && b.status == ImplStatus::Implemented)
        .count() as u32;
    (implemented, total)
}

fn build_kernel_classes(statuses: &[ContractProofStatus]) -> Vec<KernelClassSummary> {
    let status_map: BTreeMap<&str, &ContractProofStatus> =
        statuses.iter().map(|s| (s.stem.as_str(), s)).collect();

    kernel_class_map()
        .into_iter()
        .map(|(label, desc, stems)| {
            let found_stems: Vec<String> = stems
                .iter()
                .filter(|s| status_map.contains_key(**s))
                .map(|s| (*s).to_string())
                .collect();

            let min_level = found_stems
                .iter()
                .filter_map(|s| status_map.get(s.as_str()))
                .map(|c| c.proof_level)
                .min()
                .unwrap_or(ProofLevel::L1);

            let all_bound = !found_stems.is_empty()
                && found_stems.iter().all(|s| {
                    status_map.get(s.as_str()).is_some_and(|c| {
                        c.bindings_total > 0 && c.bindings_implemented == c.bindings_total
                    })
                });

            KernelClassSummary {
                label: label.to_string(),
                description: desc.to_string(),
                contract_stems: found_stems,
                min_proof_level: min_level,
                all_bound,
            }
        })
        .collect()
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() > max { &s[..max] } else { s }
}

fn current_timestamp() -> String {
    // Use a simple ISO-8601 timestamp without external deps.
    // In production this would use chrono or time crate.
    // For now we use std::time for a Unix epoch string.
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", duration.as_secs())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn minimal_contract(n_ob: usize, n_ft: usize, n_kani: usize) -> Contract {
        let mut yaml = String::from(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
"#,
        );
        for i in 0..n_ob {
            yaml.push_str(&format!(
                "  - type: invariant\n    property: \"prop {i}\"\n"
            ));
        }
        yaml.push_str("falsification_tests:\n");
        for i in 0..n_ft {
            yaml.push_str(&format!(
                "  - id: FT-{i:03}\n    rule: \"r\"\n    prediction: \"p\"\n    if_fails: \"f\"\n"
            ));
        }
        yaml.push_str("kani_harnesses:\n");
        for i in 0..n_kani {
            yaml.push_str(&format!(
                "  - id: KH-{i:03}\n    obligation: OBL-{i:03}\n    bound: 16\n"
            ));
        }
        parse_contract_str(&yaml).unwrap()
    }

    fn contract_with_lean(total: u32, lean_proved: u32) -> Contract {
        let mut c = minimal_contract(total as usize, total as usize, total as usize);
        c.verification_summary = Some(crate::schema::VerificationSummary {
            total_obligations: total,
            l2_property_tested: total,
            l3_kani_proved: total,
            l4_lean_proved: lean_proved,
            l4_sorry_count: total - lean_proved,
            l4_not_applicable: 0,
        });
        c
    }

    #[test]
    fn proof_level_display() {
        assert_eq!(ProofLevel::L1.to_string(), "L1");
        assert_eq!(ProofLevel::L2.to_string(), "L2");
        assert_eq!(ProofLevel::L3.to_string(), "L3");
        assert_eq!(ProofLevel::L4.to_string(), "L4");
        assert_eq!(ProofLevel::L5.to_string(), "L5");
    }

    #[test]
    fn proof_level_ordering() {
        assert!(ProofLevel::L1 < ProofLevel::L2);
        assert!(ProofLevel::L2 < ProofLevel::L3);
        assert!(ProofLevel::L3 < ProofLevel::L4);
        assert!(ProofLevel::L4 < ProofLevel::L5);
    }

    #[test]
    fn level_l1_for_equations_only() {
        let c = minimal_contract(0, 0, 0);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
    }

    #[test]
    fn level_l2_for_falsification_covered() {
        let c = minimal_contract(3, 3, 0);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L2);
    }

    #[test]
    fn level_l2_not_enough_tests() {
        let c = minimal_contract(3, 2, 0);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
    }

    #[test]
    fn level_l3_kani_plus_falsification() {
        let c = minimal_contract(3, 3, 2);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L3);
    }

    #[test]
    fn level_l3_kani_without_enough_tests() {
        // Has kani but not enough falsification tests
        let c = minimal_contract(3, 2, 2);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
    }

    #[test]
    fn level_l4_all_lean_proved() {
        let c = contract_with_lean(3, 3);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L4);
    }

    #[test]
    fn level_l4_partial_lean_stays_l3() {
        let c = contract_with_lean(3, 2);
        assert_eq!(compute_proof_level(&c, None), ProofLevel::L3);
    }

    #[test]
    fn level_l5_lean_plus_all_bound() {
        let c = contract_with_lean(3, 3);
        assert_eq!(compute_proof_level(&c, Some((1, 1))), ProofLevel::L5);
    }

    #[test]
    fn level_l4_when_bindings_incomplete() {
        let c = contract_with_lean(3, 3);
        assert_eq!(compute_proof_level(&c, Some((0, 1))), ProofLevel::L4);
    }

    #[test]
    fn report_empty_contracts() {
        let report = proof_status_report(&[], None, false);
        assert_eq!(report.totals.contracts, 0);
        assert_eq!(report.totals.obligations, 0);
        assert!(report.contracts.is_empty());
    }

    #[test]
    fn report_single_contract() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("test-v1".to_string(), &c)], None, false);
        assert_eq!(report.contracts.len(), 1);
        assert_eq!(report.contracts[0].stem, "test-v1");
        assert_eq!(report.contracts[0].proof_level, ProofLevel::L3);
        assert_eq!(report.totals.obligations, 3);
        assert_eq!(report.totals.falsification_tests, 3);
        assert_eq!(report.totals.kani_harnesses, 2);
    }

    #[test]
    fn report_with_binding() {
        let c = minimal_contract(3, 3, 2);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test-v1.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
        )
        .unwrap();
        let report = proof_status_report(&[("test-v1".to_string(), &c)], Some(&binding), false);
        assert_eq!(report.contracts[0].bindings_implemented, 1);
        assert_eq!(report.contracts[0].bindings_total, 1);
    }

    #[test]
    fn report_with_kernel_classes() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("softmax-kernel-v1".to_string(), &c)], None, true);
        assert!(!report.kernel_classes.is_empty());
        // Softmax is in classes A, B, C, D, E
        let class_a = report
            .kernel_classes
            .iter()
            .find(|kc| kc.label == "A")
            .unwrap();
        assert!(
            class_a
                .contract_stems
                .contains(&"softmax-kernel-v1".to_string())
        );
    }

    #[test]
    fn format_text_produces_output() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("softmax-kernel-v1".to_string(), &c)], None, true);
        let text = format_text(&report);
        assert!(text.contains("Proof Status"));
        assert!(text.contains("softmax-kernel-v1"));
        assert!(text.contains("Kernel Classes:"));
        assert!(text.contains("Totals:"));
    }

    #[test]
    fn format_text_without_classes() {
        let c = minimal_contract(2, 2, 0);
        let report = proof_status_report(&[("test-v1".to_string(), &c)], None, false);
        let text = format_text(&report);
        assert!(text.contains("test-v1"));
        assert!(!text.contains("Kernel Classes:"));
    }

    #[test]
    fn json_roundtrip() {
        let c = minimal_contract(3, 3, 2);
        let report = proof_status_report(&[("test-v1".to_string(), &c)], None, true);
        let json = serde_json::to_string(&report).unwrap();
        let parsed: ProofStatusReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.contracts.len(), 1);
        assert_eq!(parsed.contracts[0].proof_level, ProofLevel::L3);
        assert_eq!(parsed.totals.obligations, 3);
    }

    #[test]
    fn kernel_class_min_level() {
        let c1 = minimal_contract(3, 3, 2); // L3
        let c2 = minimal_contract(3, 3, 0); // L2
        let report = proof_status_report(
            &[
                ("softmax-kernel-v1".to_string(), &c1),
                ("matmul-kernel-v1".to_string(), &c2),
            ],
            None,
            true,
        );
        let class_a = report
            .kernel_classes
            .iter()
            .find(|kc| kc.label == "A")
            .unwrap();
        // min of L3 and L2 is L2
        assert_eq!(class_a.min_proof_level, ProofLevel::L2);
    }

    #[test]
    fn count_bindings_helper() {
        let c = minimal_contract(1, 1, 1);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    status: implemented
  - contract: other.yaml
    equation: g
    status: implemented
"#,
        )
        .unwrap();
        let (implemented, total) = count_bindings("test.yaml", &c, &binding);
        assert_eq!(implemented, 1);
        assert_eq!(total, 1);
    }

    #[test]
    fn truncate_helper() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn schema_version_present() {
        let report = proof_status_report(&[], None, false);
        assert_eq!(report.schema_version, "1.0.0");
    }

    #[test]
    fn timestamp_is_populated() {
        let report = proof_status_report(&[], None, false);
        assert!(!report.timestamp.is_empty());
        assert!(report.timestamp.ends_with('Z'));
    }
}
