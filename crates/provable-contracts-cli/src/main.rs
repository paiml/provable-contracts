#[cfg(test)]
use std::path::PathBuf;
use std::process;

use clap::Parser;

mod cli;
mod commands;

use cli::Commands;

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

/// Dispatch a parsed CLI subcommand to its handler
#[allow(clippy::too_many_lines)]
fn run_command(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Explain {
            contract,
            format,
            binding,
        } => commands::explain::run(&contract, binding.as_deref(), &format),
        Commands::Validate { contract } => commands::validate::run(&contract),
        Commands::Scaffold { contract } => commands::scaffold::run(&contract),
        Commands::ExtractPytorch { target, output } => {
            commands::extract::run(&target, output.as_deref())
        }
        Commands::Codegen {
            contract_dir,
            output,
        } => commands::codegen::run(&contract_dir, output.as_deref()),
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
            fuzz,
            reverse,
        } => commands::coverage::run(
            &contract_dir,
            binding.as_deref(),
            fuzz,
            reverse.as_deref(),
        ),
        Commands::Generate {
            contract,
            output,
            binding,
            readme,
            ci,
        } => commands::generate::run(&contract, &output, binding.as_deref(), readme, ci),
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
            diff_ref,
            trend,
            show_trend,
            no_cache,
            cache_stats,
        } => commands::lint::run(
            &contract_dir,
            binding.as_deref(),
            min_score,
            format.as_deref(),
            severity.as_deref(),
            strict,
            suppress.as_deref(),
            suppress_rule.as_deref(),
            suppress_file.as_deref(),
            &rule,
            config.as_deref(),
            diff_ref.as_deref(),
            trend,
            show_trend,
            no_cache,
            cache_stats,
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
        Commands::Invariants { contract } => commands::invariants::run(&contract),
        Commands::Coq { contract } => commands::coq::run(&contract),
        Commands::Fuzz { contract } => commands::fuzz::run(&contract),
        Commands::Mirai { contract } => commands::mirai::run(&contract),
        Commands::Flux { contract } => commands::flux::run(&contract),
        Commands::Tla { contract_dir } => commands::tla::run(&contract_dir),
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
