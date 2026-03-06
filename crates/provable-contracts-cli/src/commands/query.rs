//! `pv query` — Search contracts by intent, regex, or literal match.

use std::path::Path;

use provable_contracts::query::{self, ContractIndex, QueryParams, SearchMode};

/// All parameters needed to execute a query from the CLI.
#[allow(clippy::struct_excessive_bools)]
pub struct QueryCliParams<'a> {
    pub contract_dir: &'a Path,
    pub query_str: &'a str,
    pub regex: bool,
    pub literal: bool,
    pub case_sensitive: bool,
    pub limit: usize,
    pub obligation: Option<&'a str>,
    pub min_score: Option<f64>,
    pub depends_on: Option<&'a str>,
    pub depended_by: Option<&'a str>,
    pub unproven: bool,
    pub show_score: bool,
    pub show_graph: bool,
    pub show_paper: bool,
    pub show_proof_status: bool,
    pub show_binding: bool,
    pub binding: Option<&'a Path>,
    pub format: &'a str,
}

pub fn run(p: &QueryCliParams<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let index = ContractIndex::from_directory(p.contract_dir)?;

    let mode = if p.regex {
        SearchMode::Regex
    } else if p.literal {
        SearchMode::Literal
    } else {
        SearchMode::Semantic
    };

    let params = QueryParams {
        query: p.query_str.to_string(),
        mode,
        case_sensitive: p.case_sensitive,
        limit: p.limit,
        obligation_filter: p.obligation.map(String::from),
        min_score: p.min_score,
        depends_on: p.depends_on.map(String::from),
        depended_by: p.depended_by.map(String::from),
        unproven_only: p.unproven,
        show_score: p.show_score,
        show_graph: p.show_graph,
        show_paper: p.show_paper,
        show_proof_status: p.show_proof_status,
        show_binding: p.show_binding,
        binding_path: p.binding.map(|b| b.display().to_string()),
    };

    let output = query::execute(&index, &params);

    match p.format {
        "json" => println!("{}", serde_json::to_string_pretty(&output)?),
        "markdown" => print!("{}", output.to_markdown()),
        _ => print!("{output}"),
    }

    Ok(())
}
