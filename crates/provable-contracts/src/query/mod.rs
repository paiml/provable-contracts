//! Contract query engine with BM25 semantic search.
//!
//! Provides fast lookup across all contracts with semantic ranking,
//! regex/literal search, and structured filters.
//!
//! Spec: `docs/specifications/sub/query.md`

mod index;
mod types;

pub use index::ContractIndex;
pub use types::{QueryOutput, QueryParams, QueryResult, ScoreInfo, SearchMode};

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

    let filtered = apply_filters(index, scored_indices, params);
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
) -> Vec<(usize, f64)> {
    results
        .into_iter()
        .filter(|(idx, _)| {
            let entry = &index.entries[*idx];
            filter_obligation(entry, params.obligation_filter.as_ref())
                && filter_depends_on(entry, params.depends_on.as_ref())
                && filter_depended_by(index, entry, params.depended_by.as_ref())
                && filter_unproven(entry, params.unproven_only)
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

fn filter_unproven(entry: &types::ContractEntry, unproven_only: bool) -> bool {
    if !unproven_only {
        return true;
    }
    // Show contracts with more obligations than kani harnesses
    entry.obligation_count > entry.kani_count
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
    use super::*;

    fn test_index() -> ContractIndex {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        ContractIndex::from_directory(&dir).unwrap()
    }

    #[test]
    fn semantic_query_returns_results() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax numerical stability".to_string(),
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // softmax-kernel-v1 should appear in results
        assert!(
            output.results.iter().any(|r| r.stem.contains("softmax")),
            "Results should include softmax contract"
        );
    }

    #[test]
    fn literal_query_finds_contracts() {
        let index = test_index();
        let params = QueryParams {
            query: "RMSNorm".to_string(),
            mode: SearchMode::Literal,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
    }

    #[test]
    fn regex_query_works() {
        let index = test_index();
        let params = QueryParams {
            query: r"(?i)softmax".to_string(),
            mode: SearchMode::Regex,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
    }

    #[test]
    fn obligation_filter_narrows_results() {
        let index = test_index();
        let all = QueryParams {
            query: "kernel".to_string(),
            ..Default::default()
        };
        let filtered = QueryParams {
            query: "kernel".to_string(),
            obligation_filter: Some("invariant".to_string()),
            ..Default::default()
        };
        let all_out = execute(&index, &all);
        let filtered_out = execute(&index, &filtered);
        assert!(filtered_out.total_matches <= all_out.total_matches);
    }

    #[test]
    fn limit_caps_results() {
        let index = test_index();
        let params = QueryParams {
            query: "kernel".to_string(),
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(output.results.len() <= 3);
    }

    #[test]
    fn show_score_enriches_results() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_score: true,
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        assert!(output.results[0].score.is_some());
    }

    #[test]
    fn display_output_is_valid() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let text = output.to_string();
        assert!(text.contains("[1]"));
        assert!(text.contains("softmax"));
    }

    #[test]
    fn empty_query_returns_empty() {
        let index = test_index();
        let params = QueryParams {
            query: String::new(),
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(output.results.is_empty());
    }
}
