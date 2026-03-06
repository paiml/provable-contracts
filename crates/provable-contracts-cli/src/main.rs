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
        /// Include contract scores in output
        #[arg(long)]
        score: bool,
        /// Include dependency graph info
        #[arg(long)]
        graph: bool,
        /// Include paper references
        #[arg(long)]
        paper: bool,
        /// Output format: text (default) or json
        #[arg(short, long, default_value = "text")]
        format: String,
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
        Commands::Score {
            path,
            binding,
            format,
            min_score,
        } => commands::score::run(&path, binding.as_deref(), &format, min_score),
        Commands::Query {
            query,
            contract_dir,
            regex,
            literal,
            case_sensitive,
            limit,
            obligation,
            depends_on,
            depended_by,
            unproven,
            score,
            graph,
            paper,
            format,
        } => commands::query::run(
            &contract_dir,
            &query,
            regex,
            literal,
            case_sensitive,
            limit,
            obligation.as_deref(),
            depends_on.as_deref(),
            depended_by.as_deref(),
            unproven,
            score,
            graph,
            paper,
            &format,
        ),
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
