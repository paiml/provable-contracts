//! Contract query engine with BM25 semantic search.
//!
//! Provides fast lookup across all contracts with semantic ranking,
//! regex/literal search, and structured filters.
//!
//! Spec: `docs/specifications/sub/query.md`

mod index;
mod persist;
mod types;

pub use index::ContractIndex;
pub use types::{
    DiffInfo, EquationBinding, ProofStatusInfo, QueryOutput, QueryParams, QueryResult, ScoreInfo,
    SearchMode,
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
            build_result(index, params, binding.as_ref(), rank, idx, relevance)
        })
        .collect();

    QueryOutput {
        query: params.query.clone(),
        total_matches,
        results,
    }
}

fn build_result(
    index: &ContractIndex,
    params: &QueryParams,
    binding: Option<&BindingRegistry>,
    rank: usize,
    idx: usize,
    relevance: f64,
) -> QueryResult {
    let entry = &index.entries[idx];
    let (depends_on, depended_by) = graph_fields(index, entry, params.show_graph);
    QueryResult {
        rank: rank + 1,
        stem: entry.stem.clone(),
        path: entry.path.clone(),
        relevance,
        description: entry.description.clone(),
        equations: entry.equations.clone(),
        obligation_count: entry.obligation_count,
        references: opt_vec(&entry.references, params.show_paper),
        depends_on,
        depended_by,
        score: params.show_score.then(|| build_score_info(entry)).flatten(),
        proof_status: params.show_proof_status.then(|| build_proof_status_info(entry)).flatten(),
        bindings: opt_binding(entry, binding, params.show_binding),
        diff: params.show_diff.then(|| build_diff_info(entry)).flatten(),
    }
}

fn opt_vec(source: &[String], include: bool) -> Vec<String> {
    if include { source.to_vec() } else { Vec::new() }
}

fn graph_fields(
    index: &ContractIndex,
    entry: &types::ContractEntry,
    show: bool,
) -> (Vec<String>, Vec<String>) {
    if !show {
        return (Vec::new(), Vec::new());
    }
    let deps = entry.depends_on.clone();
    let rev = index.depended_by(&entry.stem).into_iter().map(String::from).collect();
    (deps, rev)
}

fn opt_binding(
    entry: &types::ContractEntry,
    binding: Option<&BindingRegistry>,
    show: bool,
) -> Vec<EquationBinding> {
    if show { build_binding_info(entry, binding) } else { Vec::new() }
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
                && filter_min_level(entry, params.min_level.as_deref())
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

fn filter_min_level(entry: &types::ContractEntry, min_level: Option<&str>) -> bool {
    let Some(min) = min_level else { return true };
    let threshold = parse_proof_level(min);
    let path = std::path::Path::new(&entry.path);
    let Ok(contract) = crate::schema::parse_contract(path) else {
        return false;
    };
    let level = crate::proof_status::compute_proof_level(&contract, None);
    level >= threshold
}

fn parse_proof_level(s: &str) -> crate::proof_status::ProofLevel {
    match s.to_uppercase().as_str() {
        "L5" => crate::proof_status::ProofLevel::L5,
        "L4" => crate::proof_status::ProofLevel::L4,
        "L3" => crate::proof_status::ProofLevel::L3,
        "L2" => crate::proof_status::ProofLevel::L2,
        _ => crate::proof_status::ProofLevel::L1,
    }
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

fn build_diff_info(entry: &types::ContractEntry) -> Option<DiffInfo> {
    let output = std::process::Command::new("git")
        .args(["log", "-1", "--format=%H %aI", "--"])
        .arg(&entry.path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let line = line.trim();
    let (hash, date) = line.split_once(' ')?;
    let date_part = date.split('T').next().unwrap_or(date);
    // Calculate days ago from ISO date
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let days_ago = parse_iso_days_ago(date_part, now);
    Some(DiffInfo {
        last_modified: date_part.to_string(),
        days_ago,
        commit_hash: hash.to_string(),
    })
}

fn parse_iso_days_ago(date: &str, now_epoch: u64) -> u64 {
    // Simple ISO date parsing: YYYY-MM-DD
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return 0;
    }
    let y: u64 = parts[0].parse().unwrap_or(0);
    let m: usize = parts[1].parse().unwrap_or(0);
    let d: u64 = parts[2].parse().unwrap_or(0);
    // Approximate epoch seconds for the date
    let days_from_epoch = y.saturating_sub(1970) * 365
        + y.saturating_sub(1969) / 4
        + month_days(m)
        + d.saturating_sub(1);
    let date_epoch = days_from_epoch * 86400;
    now_epoch.saturating_sub(date_epoch) / 86400
}

fn month_days(m: usize) -> u64 {
    const CUMULATIVE: [u64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    CUMULATIVE.get(m).copied().unwrap_or(0)
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

#[cfg(test)]
mod coverage_tests {
    include!("query_tests_coverage.rs");
}
