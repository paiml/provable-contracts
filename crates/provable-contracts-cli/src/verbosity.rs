//! Global verbosity level for CLI output control.

use std::sync::OnceLock;

/// Verbosity level set by `--quiet` / `--verbose` global flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Quiet,
    Normal,
    Verbose,
}

static LEVEL: OnceLock<Verbosity> = OnceLock::new();

/// Set the global verbosity level (call once from main).
pub fn set(level: Verbosity) {
    let _ = LEVEL.set(level);
}

/// Get the current verbosity level.
pub fn get() -> Verbosity {
    LEVEL.get().copied().unwrap_or(Verbosity::Normal)
}

/// Returns true if output should be suppressed (quiet mode).
pub fn is_quiet() -> bool {
    get() == Verbosity::Quiet
}

/// Returns true if verbose output is enabled.
pub fn is_verbose() -> bool {
    get() == Verbosity::Verbose
}
