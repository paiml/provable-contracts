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

/// Returns `true` when all obligations are Lean-proved.
fn is_lean_proved(contract: &Contract) -> bool {
    contract
        .verification_summary
        .as_ref()
        .is_some_and(|vs| vs.total_obligations > 0 && vs.l4_lean_proved == vs.total_obligations)
}

/// Returns `true` when all bindings are implemented.
fn is_fully_bound(binding_status: Option<(u32, u32)>) -> bool {
    binding_status.is_some_and(|(implemented, total)| total > 0 && implemented == total)
}

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
    if is_lean_proved(contract) {
        return if is_fully_bound(binding_status) {
            ProofLevel::L5
        } else {
            ProofLevel::L4
        };
    }

    // Check L3: Kani + falsification
    let has_tests = ft_count >= total_obligations && total_obligations > 0;
    if kani_count > 0 && has_tests {
        return ProofLevel::L3;
    }

    // Check L2: falsification tests cover obligations
    if has_tests {
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
    include!("proof_status_tests.rs");
}
