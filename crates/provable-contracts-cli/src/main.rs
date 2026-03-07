use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

mod commands;

/// Top-level CLI argument parser for the `pv` command
#[derive(Parser)]
#[command(
    name = "pv",
    about = "provable-contracts — papers to provable Rust kernels",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Available subcommands for the `pv` CLI
#[derive(Subcommand)]
enum Commands {
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
        /// Output format: text, json, sarif, github
        #[arg(short, long, default_value = "text")]
        format: String,
        /// Minimum severity to report: error, warning, info
        #[arg(long)]
        severity: Option<String>,
        /// Promote warnings to errors
        #[arg(long)]
        strict: bool,
        /// Suppress specific finding IDs (comma-separated)
        #[arg(long)]
        suppress: Option<String>,
        /// Suppress all findings for a rule
        #[arg(long)]
        suppress_rule: Option<String>,
        /// Suppress all findings in a file
        #[arg(long)]
        suppress_file: Option<String>,
        /// Override rule severity (e.g. PV-AUD-001=info)
        #[arg(long)]
        rule: Vec<String>,
        /// Path to .pv.toml config file
        #[arg(long)]
        config: Option<PathBuf>,
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
        /// Custom weights as JSON, e.g. `{"spec_depth":0.1,"falsification":0.3,"kani":0.3,"lean":0.1,"binding":0.2}`
        #[arg(long)]
        weights: Option<String>,
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
        /// Filter cross-project results to a named project (aprender, trueno, etc.)
        #[arg(long)]
        project: Option<String>,
        /// Add an explicit project path to the cross-project scan
        #[arg(long)]
        include_project: Option<PathBuf>,
        /// Force full cross-project scan even without --call-sites/--violations/--coverage-map
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
}

/// Dispatch a parsed CLI subcommand to its handler
#[allow(clippy::too_many_lines)]
fn run_command(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Validate { contract } => commands::validate::run(&contract),
        Commands::Scaffold { contract } => commands::scaffold::run(&contract),
        Commands::Kani { contract } => commands::kani::run(&contract),
        Commands::Probar { contract, binding } => {
            commands::probar::run(&contract, binding.as_deref())
        }
        Commands::Status { contract } => commands::status::run(&contract),
        Commands::Audit { contract, binding } => {
            commands::audit::run(&contract, binding.as_deref())
        }
        Commands::Diff { old, new } => commands::diff::run(&old, &new),
        Commands::Coverage {
            contract_dir,
            binding,
        } => commands::coverage::run(&contract_dir, binding.as_deref()),
        Commands::Generate {
            contract,
            output,
            binding,
        } => commands::generate::run(&contract, &output, binding.as_deref()),
        Commands::Graph {
            contract_dir,
            format,
        } => match commands::graph::GraphFormat::from_str(&format) {
            Ok(fmt) => commands::graph::run(&contract_dir, fmt),
            Err(e) => Err(e.into()),
        },
        Commands::Equations { contract, format } => {
            match commands::equations::OutputFormat::from_str(&format) {
                Ok(fmt) => commands::equations::run(&contract, fmt),
                Err(e) => Err(e.into()),
            }
        }
        Commands::Lean {
            contract,
            output_dir,
        } => commands::lean::run(&contract, output_dir.as_deref()),
        Commands::LeanStatus { path } => commands::lean_status::run(&path),
        Commands::ProofStatus {
            path,
            binding,
            format,
        } => commands::proof_status::run(&path, binding.as_deref(), &format),
        Commands::Lint {
            contract_dir,
            min_score,
            binding,
            format,
            severity,
            strict,
            suppress,
            suppress_rule,
            suppress_file,
            rule,
            config,
        } => commands::lint::run(
            &contract_dir,
            binding.as_deref(),
            min_score,
            &format,
            severity.as_deref(),
            strict,
            suppress.as_deref(),
            suppress_rule.as_deref(),
            suppress_file.as_deref(),
            &rule,
            config.as_deref(),
        ),
        Commands::Score {
            path,
            binding,
            format,
            min_score,
            summary,
            top_gaps,
            weights,
        } => commands::score::run(
            &path,
            binding.as_deref(),
            &format,
            min_score,
            summary,
            top_gaps,
            weights.as_deref(),
        ),
        Commands::Query {
            query,
            contract_dir,
            regex,
            literal,
            case_sensitive,
            limit,
            obligation,
            min_score,
            min_level,
            depends_on,
            depended_by,
            unproven,
            score,
            graph,
            paper,
            proof_status,
            binding_info,
            binding_gaps,
            diff,
            pagerank,
            call_sites,
            violations,
            coverage_map,
            project,
            include_project,
            tier,
            class,
            all_projects,
            rebuild_index,
            binding,
            format,
            exit_code,
        } => commands::query::run(&commands::query::QueryCliParams {
            contract_dir: &contract_dir,
            query_str: &query,
            regex,
            literal,
            case_sensitive,
            limit,
            obligation: obligation.as_deref(),
            min_score,
            min_level,
            depends_on: depends_on.as_deref(),
            depended_by: depended_by.as_deref(),
            unproven,
            show_score: score,
            show_graph: graph,
            show_paper: paper,
            show_proof_status: proof_status,
            show_binding: binding_info,
            binding_gaps,
            show_diff: diff,
            show_pagerank: pagerank,
            show_call_sites: call_sites,
            show_violations: violations,
            show_coverage_map: coverage_map,
            project_filter: project.as_deref(),
            include_project: include_project.as_deref(),
            tier,
            class,
            all_projects,
            rebuild_index,
            binding: binding.as_deref(),
            format: &format,
            exit_code,
        }),
        Commands::Book {
            contract_dir,
            output,
            update_summary,
            summary_path,
        } => commands::book::run(
            &contract_dir,
            &output,
            update_summary,
            summary_path.as_deref(),
        ),
    }
}

/// Entry point: parse CLI arguments and run the selected subcommand
fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_command(cli.command) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

#[cfg(test)]
#[path = "../tests/includes/dispatch_tests.rs"]
mod dispatch_tests;

#[cfg(test)]
#[path = "../tests/includes/dispatch_query_tests.rs"]
mod dispatch_query_tests;
