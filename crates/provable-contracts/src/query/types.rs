//! Types for the contract query engine.

use serde::Serialize;

/// How to interpret the query string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchMode {
    /// BM25 semantic ranking over corpus (default).
    Semantic,
    /// Regex match against all string fields.
    Regex,
    /// Exact substring match (case-insensitive by default).
    Literal,
}

/// A single entry in the contract index.
#[derive(Debug, Clone)]
pub struct ContractEntry {
    pub stem: String,
    pub path: String,
    pub description: String,
    pub equations: Vec<String>,
    pub obligation_types: Vec<String>,
    pub properties: Vec<String>,
    pub references: Vec<String>,
    pub depends_on: Vec<String>,
    pub is_registry: bool,
    pub obligation_count: usize,
    pub falsification_count: usize,
    pub kani_count: usize,
    /// Concatenated searchable text for BM25.
    pub corpus_text: String,
}

/// Parameters controlling a query.
#[derive(Debug, Clone)]
pub struct QueryParams {
    pub query: String,
    pub mode: SearchMode,
    pub case_sensitive: bool,
    pub limit: usize,
    pub obligation_filter: Option<String>,
    pub min_score: Option<f64>,
    pub depends_on: Option<String>,
    pub depended_by: Option<String>,
    pub unproven_only: bool,
    pub show_score: bool,
    pub show_graph: bool,
    pub show_paper: bool,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            mode: SearchMode::Semantic,
            case_sensitive: false,
            limit: 10,
            obligation_filter: None,
            min_score: None,
            depends_on: None,
            depended_by: None,
            unproven_only: false,
            show_score: false,
            show_graph: false,
            show_paper: false,
        }
    }
}

/// A single query result with relevance score and optional enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub rank: usize,
    pub stem: String,
    pub path: String,
    pub relevance: f64,
    pub description: String,
    pub equations: Vec<String>,
    pub obligation_count: usize,
    pub references: Vec<String>,
    pub depends_on: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<ScoreInfo>,
}

/// Inline score info for enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreInfo {
    pub composite: f64,
    pub grade: String,
    pub spec_depth: f64,
    pub falsification: f64,
    pub kani: f64,
    pub lean: f64,
    pub binding: f64,
}

/// Output of a query execution.
#[derive(Debug, Clone, Serialize)]
pub struct QueryOutput {
    pub query: String,
    pub total_matches: usize,
    pub results: Vec<QueryResult>,
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} (relevance: {:.2})", self.rank, self.stem, self.relevance)?;
        writeln!(f)?;
        writeln!(f, "    {}", self.description)?;
        if !self.equations.is_empty() {
            writeln!(f, "    Equations: {}", self.equations.join(", "))?;
        }
        writeln!(f, "    Obligations: {}", self.obligation_count)?;
        if !self.references.is_empty() {
            writeln!(f, "    Papers: {}", self.references.join("; "))?;
        }
        if let Some(s) = &self.score {
            writeln!(
                f,
                "    Score: {:.2} (Grade {})",
                s.composite, s.grade
            )?;
            writeln!(
                f,
                "    Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}",
                s.spec_depth, s.falsification, s.kani, s.lean, s.binding
            )?;
        }
        if !self.depends_on.is_empty() {
            writeln!(f, "    Depends on: {}", self.depends_on.join(", "))?;
        }
        Ok(())
    }
}

impl std::fmt::Display for QueryOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in &self.results {
            write!(f, "{r}")?;
            writeln!(f, "    ---")?;
        }
        writeln!(f, "\n{} matches for \"{}\"", self.total_matches, self.query)
    }
}
