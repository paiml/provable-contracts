use std::path::PathBuf;

use clap::Subcommand;

/// Available subcommands for the `pv` CLI
#[derive(Subcommand)]
pub enum Commands {
    /// Explain a contract in detail — narrative walkthrough of equations,
    /// obligations, verification chain, and falsification strategy
    Explain {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Output format: text (default), markdown, or json
        #[arg(long, default_value = "text")]
        format: String,
        /// Path to binding registry YAML (adds binding context)
        #[arg(long)]
        binding: Option<PathBuf>,
    },
    /// Validate a YAML kernel contract
    Validate {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate Rust trait + test scaffolding from a contract
    Scaffold {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Extract kernel equations from `PyTorch` source into YAML contract
    #[command(name = "extract-pytorch")]
    ExtractPytorch {
        /// `PyTorch` source target (`file.py::function_name`)
        target: String,
        /// Output YAML file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate Rust `debug_assert`!() from YAML contract preconditions/postconditions
    Codegen {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output Rust file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate Kani proof harnesses from a contract
    Kani {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate probar property tests from a contract
    Probar {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Path to binding registry YAML (generates wired tests)
        #[arg(long)]
        binding: Option<PathBuf>,
    },
    /// Show contract status (equations, obligations, coverage)
    Status {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Run traceability audit on a contract
    Audit {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Path to binding registry YAML (adds binding audit)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Show Coq proof tier per obligation
        #[arg(long)]
        coq: bool,
        /// Show Flux shape coverage per obligation
        #[arg(long)]
        flux: bool,
    },
    /// Diff two contract versions and suggest semver bump
    Diff {
        /// Path to the old contract YAML file
        old: PathBuf,
        /// Path to the new contract YAML file
        new: PathBuf,
    },
    /// Show cross-contract obligation coverage report
    Coverage {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Path to binding registry YAML (adds binding coverage)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Include fuzz coverage data
        #[arg(long)]
        fuzz: bool,
        /// Reverse coverage: scan crate dir for unbound pub fns
        #[arg(long)]
        reverse: Option<PathBuf>,
    },
    /// Generate all artifacts (scaffold, kani, probar) to disk
    Generate {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Output directory for generated files
        #[arg(short, long, default_value = "generated")]
        output: PathBuf,
        /// Path to binding registry YAML (generates wired tests)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Generate CONTRACT-README.md (requires --binding)
        #[arg(long)]
        readme: bool,
        /// Generate .github/workflows/contracts.yml
        #[arg(long)]
        ci: bool,
    },
    /// Show contract dependency graph
    Graph {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output format: text (default), dot, json, or mermaid
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Display equations from a contract
    Equations {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Output format: text (default), latex, ptx, or asm
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Generate Lean 4 definitions and theorem stubs from a contract
    Lean {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Output directory for generated Lean files
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    /// Report Lean 4 proof status across contracts
    LeanStatus {
        /// Path to a contract YAML file or directory of contracts
        #[arg(default_value = "contracts")]
        path: PathBuf,
    },
    /// Report hierarchical proof levels (L1–L5) across contracts
    ProofStatus {
        /// Path to a contract YAML file or directory of contracts
        #[arg(default_value = "contracts")]
        path: PathBuf,
        /// Path to binding registry YAML (adds binding coverage)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Output format: text (default) or json
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Run all contract quality gates (validate + audit + score)
    Lint {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Minimum composite score threshold (default: 0.0 = no score gate)
        #[arg(long, default_value = "0.0")]
        min_score: f64,
        /// Path to binding registry YAML
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Output format: text (default), json, sarif, github
        #[arg(short, long)]
        format: Option<String>,
        /// Minimum severity to report: error, warning, info
        #[arg(long)]
        severity: Option<String>,
        /// Promote warnings to errors
        #[arg(long)]
        strict: bool,
        /// Suppress specific finding IDs (comma-separated)
        #[arg(long)]
        suppress: Option<String>,
        /// Suppress all findings for a rule (comma-separated)
        #[arg(long)]
        suppress_rule: Option<String>,
        /// Suppress all findings matching a file path (comma-separated)
        #[arg(long)]
        suppress_file: Option<String>,
        /// Override rule severity (e.g. PV-AUD-001=info)
        #[arg(long)]
        rule: Vec<String>,
        /// Path to .pv.toml config file
        #[arg(long)]
        config: Option<PathBuf>,
        /// Only lint contracts changed since base ref (e.g. main, HEAD~5)
        #[arg(long = "diff")]
        diff_ref: Option<String>,
        /// Record quality trend snapshot
        #[arg(long)]
        trend: bool,
        /// Show quality trend history
        #[arg(long)]
        show_trend: bool,
        /// Bypass lint cache
        #[arg(long)]
        no_cache: bool,
        /// Show cache hit/miss statistics
        #[arg(long)]
        cache_stats: bool,
        /// Show auto-fix suggestions (dry run)
        #[arg(long)]
        suggest: bool,
        /// Suppress findings in baseline SARIF file
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Apply deterministic auto-fixes
        #[arg(long)]
        fix: bool,
        /// Re-lint on file change (polling)
        #[arg(long)]
        watch: bool,
        /// Show aggregate contract coverage metric
        #[arg(long)]
        coverage: bool,
        /// Minimum coverage percentage (exit 1 if below)
        #[arg(long)]
        min_coverage: Option<f64>,
        /// Path to crate directory for reverse coverage gate
        #[arg(long)]
        crate_dir: Option<PathBuf>,
    },
    /// Score contracts or a codebase directory
    Score {
        /// Path to a contract YAML file or directory of contracts
        #[arg(default_value = "contracts")]
        path: PathBuf,
        /// Path to binding registry YAML
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Output format: text (default) or json
        #[arg(short, long, default_value = "text")]
        format: String,
        /// Minimum score threshold (exit 1 if below)
        #[arg(long)]
        min_score: Option<f64>,
        /// Show aggregate summary only (no per-contract detail)
        #[arg(long)]
        summary: bool,
        /// Show top N gaps by impact (default: 5)
        #[arg(long, default_value = "5")]
        top_gaps: usize,
        /// Custom weights as JSON
        #[arg(long)]
        weights: Option<String>,
        /// Exit with status 1 if any contract below --min-score
        #[arg(long)]
        exit_code: bool,
    },
    /// Search contracts by intent, regex, or literal match
    Query {
        /// Search query string
        query: String,
        /// Directory containing contract YAML files
        #[arg(long, default_value = "contracts")]
        contract_dir: PathBuf,
        /// Use regex matching instead of semantic search
        #[arg(long)]
        regex: bool,
        /// Use literal substring matching
        #[arg(long)]
        literal: bool,
        /// Force case-sensitive matching
        #[arg(long)]
        case_sensitive: bool,
        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Filter by obligation type (invariant, equivalence, bound, etc.)
        #[arg(long)]
        obligation: Option<String>,
        /// Filter to contracts depending on this stem
        #[arg(long)]
        depends_on: Option<String>,
        /// Filter to contracts depended on by this stem
        #[arg(long)]
        depended_by: Option<String>,
        /// Show only contracts with unproven obligations
        #[arg(long)]
        unproven: bool,
        /// Minimum score threshold (filter results below this)
        #[arg(long)]
        min_score: Option<f64>,
        /// Minimum proof level (L1-L5) to include
        #[arg(long)]
        min_level: Option<String>,
        /// Include contract scores in output
        #[arg(long)]
        score: bool,
        /// Include dependency graph info
        #[arg(long)]
        graph: bool,
        /// Include paper references
        #[arg(long)]
        paper: bool,
        /// Include proof level (L1-L5) in output
        #[arg(long)]
        proof_status: bool,
        /// Include binding status per equation
        #[arg(long)]
        binding_info: bool,
        /// Show only contracts with unimplemented bindings
        #[arg(long)]
        binding_gaps: bool,
        /// Show last git modification date
        #[arg(long)]
        diff: bool,
        /// Show dependency pagerank score
        #[arg(long)]
        pagerank: bool,
        /// Show cross-project call sites
        #[arg(long)]
        call_sites: bool,
        /// Show contract violations in consumer projects
        #[arg(long)]
        violations: bool,
        /// Show cross-project coverage matrix
        #[arg(long)]
        coverage_map: bool,
        /// Filter cross-project results to a named project
        #[arg(long)]
        project: Option<String>,
        /// Add an explicit project path to the cross-project scan
        #[arg(long)]
        include_project: Option<PathBuf>,
        /// Force full cross-project scan
        #[arg(long)]
        all_projects: bool,
        /// Filter by contract tier (1-7)
        #[arg(long)]
        tier: Option<u8>,
        /// Filter by kernel equivalence class (A-E)
        #[arg(long, value_name = "CLASS")]
        class: Option<char>,
        /// Force rebuild of the contract index (ignore cache)
        #[arg(long)]
        rebuild_index: bool,
        /// Path to binding registry YAML (for --binding-info)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Output format: text, json, or markdown
        #[arg(short, long, default_value = "text")]
        format: String,
        /// Exit with status 1 if no results match (for CI quality gates)
        #[arg(long)]
        exit_code: bool,
    },
    /// Generate type invariant trait + Kani preservation harnesses
    Invariants {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate Coq theorem stubs from a contract
    Coq {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate libfuzzer fuzz targets from a contract
    Fuzz {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate MIRAI abstract interpretation annotations from a contract
    Mirai {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate Flux refinement type annotations from a contract
    Flux {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Generate TLA+ system-level specification from contract dependency DAG
    Tla {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
    },
    /// Generate mdBook pages for contracts
    Book {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output directory for generated pages
        #[arg(short, long, default_value = "book/src/contracts")]
        output: PathBuf,
        /// Also update book/src/SUMMARY.md with contract links
        #[arg(long)]
        update_summary: bool,
        /// Path to SUMMARY.md (default: book/src/SUMMARY.md)
        #[arg(long)]
        summary_path: Option<PathBuf>,
    },
    /// Infer contracts and bindings for unbound functions in a crate
    Infer {
        /// Path to the crate directory to scan
        crate_dir: PathBuf,
        /// Path to binding registry YAML
        #[arg(long)]
        binding: PathBuf,
        /// Directory containing contract YAML files
        #[arg(long, default_value = "contracts")]
        contract_dir: PathBuf,
        /// Maximum number of suggestions to show
        #[arg(long, default_value = "20")]
        top: usize,
    },
    /// Remove enforcement level lock from a contract (requires --reason)
    Unlock {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Mandatory reason for unlocking (audit trail)
        #[arg(long)]
        reason: String,
    },
}
