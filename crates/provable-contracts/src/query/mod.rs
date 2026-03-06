//! Contract query engine with BM25 semantic search.
//!
//! Provides fast lookup across all contracts with semantic ranking,
//! regex/literal search, and structured filters.
//!
//! Spec: `docs/specifications/sub/query.md`

mod index;
mod types;

pub use index::ContractIndex;
pub use types::{
    EquationBinding, ProofStatusInfo, QueryOutput, QueryParams, QueryResult, ScoreInfo, SearchMode,
};

use crate::binding::BindingRegistry;
use crate::scoring;

/// Execute a query against a contract index.
pub fn execute(index: &ContractIndex, params: &QueryParams) -> QueryOutput {
    let scored_indices = match params.mode {
        SearchMode::Semantic => index.bm25_search(&params.query),
        SearchMode::Regex => {
            match index.regex_search(&params.query) {
                Ok(idxs) => idxs.into_iter().map(|i| (i, 1.0)).collect(),
                Err(_) => Vec::new(), // Invalid regex yields empty results
            }
        }
        SearchMode::Literal => index
            .literal_search(&params.query, params.case_sensitive)
            .into_iter()
            .map(|i| (i, 1.0))
            .collect(),
    };

    let binding = params.binding_path.as_ref().and_then(|p| {
        let content = std::fs::read_to_string(p).ok()?;
        serde_yaml::from_str::<BindingRegistry>(&content).ok()
    });

    let filtered = apply_filters(index, scored_indices, params, binding.as_ref());
    let total_matches = filtered.len();
    let limited: Vec<_> = filtered.into_iter().take(params.limit).collect();

    let results: Vec<QueryResult> = limited
        .into_iter()
        .enumerate()
        .map(|(rank, (idx, relevance))| {
            let entry = &index.entries[idx];
            let score = if params.show_score {
                build_score_info(entry)
            } else {
                None
            };
            let proof_status = if params.show_proof_status {
                build_proof_status_info(entry)
            } else {
                None
            };
            let bindings = if params.show_binding {
                build_binding_info(entry, binding.as_ref())
            } else {
                Vec::new()
            };

            QueryResult {
                rank: rank + 1,
                stem: entry.stem.clone(),
                path: entry.path.clone(),
                relevance,
                description: entry.description.clone(),
                equations: entry.equations.clone(),
                obligation_count: entry.obligation_count,
                references: if params.show_paper {
                    entry.references.clone()
                } else {
                    Vec::new()
                },
                depends_on: if params.show_graph {
                    entry.depends_on.clone()
                } else {
                    Vec::new()
                },
                score,
                proof_status,
                bindings,
            }
        })
        .collect();

    QueryOutput {
        query: params.query.clone(),
        total_matches,
        results,
    }
}

fn apply_filters(
    index: &ContractIndex,
    results: Vec<(usize, f64)>,
    params: &QueryParams,
    binding: Option<&BindingRegistry>,
) -> Vec<(usize, f64)> {
    results
        .into_iter()
        .filter(|(idx, _)| {
            let entry = &index.entries[*idx];
            filter_obligation(entry, params.obligation_filter.as_ref())
                && filter_depends_on(entry, params.depends_on.as_ref())
                && filter_depended_by(index, entry, params.depended_by.as_ref())
                && filter_unproven(entry, params.unproven_only)
                && filter_min_score(index, entry, params.min_score)
                && filter_binding_gaps(entry, params.binding_gaps_only, binding)
        })
        .collect()
}

fn filter_obligation(
    entry: &types::ContractEntry,
    obligation: Option<&String>,
) -> bool {
    match obligation {
        Some(ot) => entry.obligation_types.iter().any(|t| t == ot),
        None => true,
    }
}

fn filter_depends_on(
    entry: &types::ContractEntry,
    depends_on: Option<&String>,
) -> bool {
    match depends_on {
        Some(dep) => entry.depends_on.iter().any(|d| d == dep),
        None => true,
    }
}

fn filter_depended_by(
    index: &ContractIndex,
    entry: &types::ContractEntry,
    depended_by: Option<&String>,
) -> bool {
    match depended_by {
        Some(target) => {
            // entry must be depended-on by `target`
            index
                .get_by_stem(target)
                .is_some_and(|t| t.depends_on.contains(&entry.stem))
        }
        None => true,
    }
}

fn filter_min_score(
    index: &ContractIndex,
    entry: &types::ContractEntry,
    min_score: Option<f64>,
) -> bool {
    let Some(threshold) = min_score else {
        return true;
    };
    // Use score cache (O(1)) when available, fall back to re-computing
    if let Some(cached) = index.cached_score(&entry.stem) {
        return cached >= threshold;
    }
    build_score_info(entry).is_some_and(|s| s.composite >= threshold)
}

fn filter_binding_gaps(
    entry: &types::ContractEntry,
    gaps_only: bool,
    binding: Option<&BindingRegistry>,
) -> bool {
    if !gaps_only {
        return true;
    }
    let Some(binding) = binding else {
        return false; // No binding registry = can't check gaps
    };
    let contract_file = format!("{}.yaml", entry.stem);
    binding.bindings.iter().any(|b| {
        b.contract == contract_file
            && (b.status == crate::binding::ImplStatus::NotImplemented
                || b.status == crate::binding::ImplStatus::Partial)
    })
}

fn filter_unproven(entry: &types::ContractEntry, unproven_only: bool) -> bool {
    if !unproven_only {
        return true;
    }
    // Show contracts with more obligations than kani harnesses
    entry.obligation_count > entry.kani_count
}

#[allow(clippy::cast_possible_truncation)]
fn build_proof_status_info(entry: &types::ContractEntry) -> Option<ProofStatusInfo> {
    let path = std::path::Path::new(&entry.path);
    let contract = crate::schema::parse_contract(path).ok()?;
    let level = crate::proof_status::compute_proof_level(&contract, None);
    Some(ProofStatusInfo {
        level: level.to_string(),
        obligations: entry.obligation_count as u32,
        falsification_tests: entry.falsification_count as u32,
        kani_harnesses: entry.kani_count as u32,
        lean_proved: contract
            .verification_summary
            .as_ref()
            .map_or(0, |vs| vs.l4_lean_proved),
    })
}

fn build_binding_info(
    entry: &types::ContractEntry,
    binding: Option<&BindingRegistry>,
) -> Vec<EquationBinding> {
    let Some(binding) = binding else {
        return entry
            .equations
            .iter()
            .map(|eq| EquationBinding {
                equation: eq.clone(),
                status: "no binding registry".into(),
                module_path: None,
            })
            .collect();
    };
    let contract_file = format!("{}.yaml", entry.stem);
    entry
        .equations
        .iter()
        .map(|eq| {
            let found = binding
                .bindings
                .iter()
                .find(|b| b.contract == contract_file && b.equation == *eq);
            match found {
                Some(b) => EquationBinding {
                    equation: eq.clone(),
                    status: b.status.to_string(),
                    module_path: b.module_path.clone(),
                },
                None => EquationBinding {
                    equation: eq.clone(),
                    status: "unbound".into(),
                    module_path: None,
                },
            }
        })
        .collect()
}

fn build_score_info(entry: &types::ContractEntry) -> Option<ScoreInfo> {
    // Parse the contract to compute the score
    let path = std::path::Path::new(&entry.path);
    let contract = crate::schema::parse_contract(path).ok()?;
    let score = scoring::score_contract(&contract, None, &entry.stem);
    Some(ScoreInfo {
        composite: score.composite,
        grade: score.grade.to_string(),
        spec_depth: score.spec_depth,
        falsification: score.falsification_coverage,
        kani: score.kani_coverage,
        lean: score.lean_coverage,
        binding: score.binding_coverage,
    })
}

#[cfg(test)]
mod tests {
    include!("query_tests.rs");
}
