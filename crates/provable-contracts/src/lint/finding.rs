//! Lint findings — individual diagnostics emitted by `pv lint` gates.
//!
//! Each finding maps to a rule from the catalog and carries location
//! and suppression metadata. Findings are the intermediate representation
//! that feeds into all output formats (text, JSON, SARIF, GitHub).
//!
//! Spec: `docs/specifications/sub/lint.md` Section 4

use serde::Serialize;

use super::rules::RuleSeverity;

/// A single lint finding (diagnostic).
#[derive(Debug, Clone, Serialize)]
pub struct LintFinding {
    pub rule_id: String,
    pub severity: RuleSeverity,
    pub message: String,
    pub file: String,
    pub line: Option<u32>,
    pub contract_stem: Option<String>,
    pub suppressed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suppression_reason: Option<String>,
}

impl LintFinding {
    /// Create a new unsuppressed finding.
    pub fn new(
        rule_id: impl Into<String>,
        severity: RuleSeverity,
        message: impl Into<String>,
        file: impl Into<String>,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            severity,
            message: message.into(),
            file: file.into(),
            line: None,
            contract_stem: None,
            suppressed: false,
            suppression_reason: None,
        }
    }

    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_stem(mut self, stem: impl Into<String>) -> Self {
        self.contract_stem = Some(stem.into());
        self
    }

    pub fn suppress(mut self, reason: impl Into<String>) -> Self {
        self.suppressed = true;
        self.suppression_reason = Some(reason.into());
        self
    }

    /// Format as GitHub Actions workflow command.
    pub fn to_github_annotation(&self) -> String {
        let level = match self.severity {
            RuleSeverity::Error => "error",
            RuleSeverity::Warning => "warning",
            RuleSeverity::Info | RuleSeverity::Off => "notice",
        };
        let line_part = self
            .line
            .map(|l| format!(",line={l}"))
            .unwrap_or_default();
        format!(
            "::{level} file={}{line_part}::{}: {}",
            self.file, self.rule_id, self.message
        )
    }

    /// Fingerprint for baseline matching: (rule_id, file, message hash).
    pub fn fingerprint(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.rule_id.hash(&mut hasher);
        self.file.hash(&mut hasher);
        self.message.hash(&mut hasher);
        format!("{}:{}:{:016x}", self.rule_id, self.file, hasher.finish())
    }
}

impl std::fmt::Display for LintFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sev = match self.severity {
            RuleSeverity::Error => "ERROR",
            RuleSeverity::Warning => "WARN",
            RuleSeverity::Info => "INFO",
            RuleSeverity::Off => "OFF",
        };
        let suppressed = if self.suppressed { " [suppressed]" } else { "" };
        write!(
            f,
            "[{sev}] {}: {} ({}){suppressed}",
            self.rule_id, self.message, self.file
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LintFinding {
        LintFinding::new(
            "PV-VAL-001",
            RuleSeverity::Error,
            "Missing proof_obligations",
            "contracts/example-v1.yaml",
        )
    }

    #[test]
    fn display_format() {
        let f = sample();
        let s = f.to_string();
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("PV-VAL-001"));
        assert!(s.contains("Missing proof_obligations"));
        assert!(s.contains("example-v1.yaml"));
    }

    #[test]
    fn display_suppressed() {
        let f = sample().suppress("known gap");
        assert!(f.to_string().contains("[suppressed]"));
    }

    #[test]
    fn github_annotation_error() {
        let f = sample().with_line(42);
        let ann = f.to_github_annotation();
        assert!(ann.starts_with("::error "));
        assert!(ann.contains("file=contracts/example-v1.yaml"));
        assert!(ann.contains(",line=42"));
        assert!(ann.contains("PV-VAL-001"));
    }

    #[test]
    fn github_annotation_warning() {
        let f = LintFinding::new("PV-AUD-001", RuleSeverity::Warning, "test", "file.yaml");
        assert!(f.to_github_annotation().starts_with("::warning "));
    }

    #[test]
    fn github_annotation_no_line() {
        let f = sample();
        let ann = f.to_github_annotation();
        assert!(!ann.contains(",line="));
    }

    #[test]
    fn fingerprint_deterministic() {
        let f = sample();
        let fp1 = f.fingerprint();
        let fp2 = f.fingerprint();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_differs_on_rule() {
        let f1 = sample();
        let f2 = LintFinding::new(
            "PV-VAL-002",
            RuleSeverity::Error,
            "Missing proof_obligations",
            "contracts/example-v1.yaml",
        );
        assert_ne!(f1.fingerprint(), f2.fingerprint());
    }

    #[test]
    fn with_stem() {
        let f = sample().with_stem("example-v1");
        assert_eq!(f.contract_stem.as_deref(), Some("example-v1"));
    }

    #[test]
    fn serializes_to_json() {
        let f = sample();
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains("PV-VAL-001"));
        assert!(json.contains("error"));
    }

    #[test]
    fn suppressed_serializes() {
        let f = sample().suppress("reason");
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains("\"suppressed\":true"));
        assert!(json.contains("\"suppression_reason\":\"reason\""));
    }
}
