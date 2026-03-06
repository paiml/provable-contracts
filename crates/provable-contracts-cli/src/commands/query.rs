//! `pv query` — Search contracts by intent, regex, or literal match.

use std::path::Path;

use provable_contracts::query::{self, ContractIndex, QueryParams, SearchMode};

pub fn run(
    contract_dir: &Path,
    query_str: &str,
    regex: bool,
    literal: bool,
    case_sensitive: bool,
    limit: usize,
    obligation: Option<&str>,
    depends_on: Option<&str>,
    depended_by: Option<&str>,
    unproven: bool,
    show_score: bool,
    show_graph: bool,
    show_paper: bool,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let index = ContractIndex::from_directory(contract_dir)?;

    let mode = if regex {
        SearchMode::Regex
    } else if literal {
        SearchMode::Literal
    } else {
        SearchMode::Semantic
    };

    let params = QueryParams {
        query: query_str.to_string(),
        mode,
        case_sensitive,
        limit,
        obligation_filter: obligation.map(String::from),
        min_score: None,
        depends_on: depends_on.map(String::from),
        depended_by: depended_by.map(String::from),
        unproven_only: unproven,
        show_score,
        show_graph,
        show_paper,
    };

    let output = query::execute(&index, &params);

    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&output)?),
        _ => print!("{output}"),
    }

    Ok(())
}
