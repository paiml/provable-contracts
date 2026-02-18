use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("Failed to read contract file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("Schema violation: {0}")]
    Schema(String),

    #[error("Missing required field: {section}.{field}")]
    MissingField { section: String, field: String },

    #[error("Invalid reference: {from} references non-existent {to}")]
    InvalidReference { from: String, to: String },

    #[error("Duplicate ID: {id} in {section}")]
    DuplicateId { id: String, section: String },
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub severity: Severity,
    pub rule: String,
    pub message: String,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.severity {
            Severity::Error => "ERROR",
            Severity::Warning => "WARN",
            Severity::Info => "INFO",
        };
        write!(f, "[{prefix}] {}: {}", self.rule, self.message)
    }
}
