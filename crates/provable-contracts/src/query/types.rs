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
#[allow(clippy::struct_excessive_bools)]
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
    pub show_proof_status: bool,
    pub show_binding: bool,
    pub binding_path: Option<String>,
    pub binding_gaps_only: bool,
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
            show_proof_status: false,
            show_binding: false,
            binding_path: None,
            binding_gaps_only: false,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_status: Option<ProofStatusInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<EquationBinding>,
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

/// Inline proof status info for enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct ProofStatusInfo {
    pub level: String,
    pub obligations: u32,
    pub falsification_tests: u32,
    pub kani_harnesses: u32,
    pub lean_proved: u32,
}

/// Binding status for a single equation.
#[derive(Debug, Clone, Serialize)]
pub struct EquationBinding {
    pub equation: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_path: Option<String>,
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
        if let Some(ps) = &self.proof_status {
            writeln!(
                f,
                "    Proof Level: {} (ob:{} ft:{} kani:{} lean:{})",
                ps.level, ps.obligations, ps.falsification_tests, ps.kani_harnesses, ps.lean_proved
            )?;
        }
        if !self.bindings.is_empty() {
            writeln!(f, "    Bindings:")?;
            for b in &self.bindings {
                let loc = b.module_path.as_deref().unwrap_or("unbound");
                writeln!(f, "      {}: {} ({})", b.equation, b.status, loc)?;
            }
        }
        if !self.depends_on.is_empty() {
            writeln!(f, "    Depends on: {}", self.depends_on.join(", "))?;
        }
        Ok(())
    }
}

impl QueryOutput {
    /// Render results as Markdown (for `--format markdown`).
    pub fn to_markdown(&self) -> String {
        let mut out = format!("## Query: \"{}\"\n\n", self.query);
        for r in &self.results {
            out.push_str(&format!("### {}. {}\n\n", r.rank, r.stem));
            out.push_str(&format!("- **Relevance:** {:.2}\n", r.relevance));
            if !r.equations.is_empty() {
                out.push_str(&format!("- **Equations:** {}\n", r.equations.join(", ")));
            }
            out.push_str(&format!("- **Obligations:** {}\n", r.obligation_count));
            if let Some(s) = &r.score {
                out.push_str(&format!("- **Score:** {:.2} (Grade {})\n", s.composite, s.grade));
                out.push_str(&format!(
                    "- Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}\n",
                    s.spec_depth, s.falsification, s.kani, s.lean, s.binding
                ));
            }
            if let Some(ps) = &r.proof_status {
                out.push_str(&format!(
                    "- **Proof Level:** {} (ob:{} ft:{} kani:{} lean:{})\n",
                    ps.level, ps.obligations, ps.falsification_tests,
                    ps.kani_harnesses, ps.lean_proved
                ));
            }
            if !r.bindings.is_empty() {
                out.push_str("- **Bindings:**\n");
                for b in &r.bindings {
                    let loc = b.module_path.as_deref().unwrap_or("unbound");
                    out.push_str(&format!("  - `{}`: {} (`{}`)\n", b.equation, b.status, loc));
                }
            }
            if !r.references.is_empty() {
                out.push_str(&format!("- **Papers:** {}\n", r.references.join("; ")));
            }
            if !r.depends_on.is_empty() {
                out.push_str(&format!("- **Depends on:** {}\n", r.depends_on.join(", ")));
            }
            out.push('\n');
        }
        out.push_str(&format!("*{} matches*\n", self.total_matches));
        out
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
