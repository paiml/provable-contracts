use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

mod commands;

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
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Validate { contract } => {
            commands::validate::run(&contract)
        }
        Commands::Scaffold { contract } => {
            commands::scaffold::run(&contract)
        }
        Commands::Kani { contract } => {
            commands::kani::run(&contract)
        }
        Commands::Probar { contract } => {
            commands::probar::run(&contract)
        }
        Commands::Status { contract } => {
            commands::status::run(&contract)
        }
        Commands::Audit { contract } => {
            commands::audit::run(&contract)
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
