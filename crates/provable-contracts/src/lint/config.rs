//! `.pv.toml` configuration file for `pv lint`.
//!
//! Search order: `./.pv.toml`, repo root, `$HOME/.config/pv/config.toml`.
//! CLI flags override config file values.
//!
//! Spec: `docs/specifications/sub/lint.md` Section 11

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::rules::RuleSeverity;

/// Parsed `.pv.toml` configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PvConfig {
    #[serde(default)]
    pub lint: LintSection,
    #[serde(default)]
    pub output: OutputSection,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LintSection {
    pub min_score: Option<f64>,
    pub severity: Option<String>,
    #[serde(default)]
    pub strict: bool,
    pub contracts_dir: Option<String>,
    pub binding: Option<String>,
    #[serde(default)]
    pub rules: HashMap<String, String>,
    #[serde(default)]
    pub suppress: SuppressSection,
    #[serde(default)]
    pub diff: DiffSection,
    #[serde(default)]
    pub trend: TrendSection,
    #[serde(default)]
    pub cache: CacheSection,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SuppressSection {
    #[serde(default)]
    pub findings: Vec<String>,
    #[serde(default)]
    pub rules: Vec<String>,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffSection {
    pub base_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSection {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_retention_days")]
    pub retention_days: u32,
    #[serde(default = "default_drift_threshold")]
    pub drift_threshold: f64,
}

impl Default for TrendSection {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_days: 90,
            drift_threshold: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSection {
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub dir: Option<String>,
}

impl Default for CacheSection {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputSection {
    pub format: Option<String>,
    pub color: Option<String>,
}

fn default_true() -> bool {
    true
}
fn default_retention_days() -> u32 {
    90
}
fn default_drift_threshold() -> f64 {
    0.05
}

/// Resolved rule severity map: rule ID -> effective severity.
pub fn resolve_rule_severities(
    config: &PvConfig,
    cli_overrides: &[(String, String)],
    strict: bool,
) -> HashMap<String, RuleSeverity> {
    let mut map = HashMap::new();

    // Start with defaults from rule catalog
    for rule in super::rules::RULES {
        map.insert(rule.id.to_string(), rule.default_severity);
    }

    // Apply config file overrides
    for (id, sev_str) in &config.lint.rules {
        if let Some(sev) = RuleSeverity::from_str_opt(sev_str) {
            map.insert(id.clone(), sev);
        }
    }

    // Apply CLI overrides
    for (id, sev_str) in cli_overrides {
        if let Some(sev) = RuleSeverity::from_str_opt(sev_str) {
            map.insert(id.clone(), sev);
        }
    }

    // Strict mode: promote warnings to errors
    if strict || config.lint.strict {
        for sev in map.values_mut() {
            if *sev == RuleSeverity::Warning {
                *sev = RuleSeverity::Error;
            }
        }
    }

    map
}

/// Search for `.pv.toml` in standard locations.
pub fn find_config(start: &Path) -> Option<PathBuf> {
    // 1. Current directory
    let local = start.join(".pv.toml");
    if local.is_file() {
        return Some(local);
    }

    // 2. Git repo root
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(start)
        .output()
    {
        if output.status.success() {
            let root = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let repo_config = PathBuf::from(&root).join(".pv.toml");
            if repo_config.is_file() {
                return Some(repo_config);
            }
        }
    }

    // 3. User config
    if let Some(home) = std::env::var_os("HOME") {
        let user_config = PathBuf::from(home).join(".config/pv/config.toml");
        if user_config.is_file() {
            return Some(user_config);
        }
    }

    None
}

/// Load and parse config from a path.
pub fn load_config(path: &Path) -> Result<PvConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    Ok(toml_parse(&content))
}

/// Parse TOML string into `PvConfig`. Avoids adding `toml` as a dependency
/// by parsing the subset we need manually.
fn toml_parse(content: &str) -> PvConfig {
    // We parse a deliberately limited TOML subset.
    // Keys we recognise: [lint], [lint.rules], [lint.suppress], [output]
    let mut config = PvConfig::default();
    let mut current_section = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].to_string();
            continue;
        }

        if let Some((key, val)) = parse_kv(trimmed) {
            apply_kv(&mut config, &current_section, &key, &val);
        }
    }

    config
}

fn parse_kv(line: &str) -> Option<(String, String)> {
    let mut parts = line.splitn(2, '=');
    let key = parts.next()?.trim().to_string();
    let val = parts.next()?.trim().to_string();
    // Strip surrounding quotes
    let val = val.trim_matches('"').trim_matches('\'').to_string();
    Some((key, val))
}

fn apply_kv(config: &mut PvConfig, section: &str, key: &str, val: &str) {
    match section {
        "lint" => match key {
            "min_score" => config.lint.min_score = val.parse().ok(),
            "severity" => config.lint.severity = Some(val.to_string()),
            "strict" => config.lint.strict = val == "true",
            "contracts_dir" => config.lint.contracts_dir = Some(val.to_string()),
            "binding" => config.lint.binding = Some(val.to_string()),
            _ => {}
        },
        "lint.rules" => {
            config.lint.rules.insert(key.to_string(), val.to_string());
        }
        "lint.suppress" => match key {
            "findings" => config.lint.suppress.findings = parse_toml_array(val),
            "rules" => config.lint.suppress.rules = parse_toml_array(val),
            "files" => config.lint.suppress.files = parse_toml_array(val),
            _ => {}
        },
        "lint.diff" => {
            if key == "base_ref" {
                config.lint.diff.base_ref = Some(val.to_string());
            }
        }
        "lint.trend" => match key {
            "enabled" => config.lint.trend.enabled = val == "true",
            "retention_days" => config.lint.trend.retention_days = val.parse().unwrap_or(90),
            "drift_threshold" => config.lint.trend.drift_threshold = val.parse().unwrap_or(0.05),
            _ => {}
        },
        "lint.cache" => match key {
            "enabled" => config.lint.cache.enabled = val == "true",
            "dir" => config.lint.cache.dir = Some(val.to_string()),
            _ => {}
        },
        "output" => match key {
            "format" => config.output.format = Some(val.to_string()),
            "color" => config.output.color = Some(val.to_string()),
            _ => {}
        },
        _ => {}
    }
}

fn parse_toml_array(val: &str) -> Vec<String> {
    let trimmed = val.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(',')
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_config() {
        let config = toml_parse("");
        assert!(config.lint.min_score.is_none());
        assert!(!config.lint.strict);
    }

    #[test]
    fn parse_lint_section() {
        let toml = r#"
[lint]
min_score = 0.60
severity = "warning"
strict = true
contracts_dir = "contracts/"
binding = "contracts/aprender/binding.yaml"
"#;
        let config = toml_parse(toml);
        assert_eq!(config.lint.min_score, Some(0.60));
        assert_eq!(config.lint.severity.as_deref(), Some("warning"));
        assert!(config.lint.strict);
        assert_eq!(config.lint.contracts_dir.as_deref(), Some("contracts/"));
        assert!(config.lint.binding.as_deref().unwrap().contains("aprender"));
    }

    #[test]
    fn parse_rules_section() {
        let toml = r#"
[lint.rules]
PV-VAL-001 = "error"
PV-AUD-001 = "info"
PV-SCR-001 = "warning"
"#;
        let config = toml_parse(toml);
        assert_eq!(config.lint.rules.get("PV-VAL-001").unwrap(), "error");
        assert_eq!(config.lint.rules.get("PV-AUD-001").unwrap(), "info");
    }

    #[test]
    fn parse_suppress_section() {
        let toml = r#"
[lint.suppress]
findings = ["SM-INV-001", "KANI-SM-002"]
rules = ["PV-AUD-002"]
files = ["contracts/arch-constraints-v1.yaml"]
"#;
        let config = toml_parse(toml);
        assert_eq!(config.lint.suppress.findings.len(), 2);
        assert_eq!(config.lint.suppress.rules, vec!["PV-AUD-002"]);
        assert_eq!(config.lint.suppress.files.len(), 1);
    }

    #[test]
    fn parse_output_section() {
        let toml = r#"
[output]
format = "sarif"
color = "auto"
"#;
        let config = toml_parse(toml);
        assert_eq!(config.output.format.as_deref(), Some("sarif"));
        assert_eq!(config.output.color.as_deref(), Some("auto"));
    }

    #[test]
    fn parse_comments_and_blank_lines() {
        let toml = r#"
# This is a comment

[lint]
# Another comment
min_score = 0.75

"#;
        let config = toml_parse(toml);
        assert_eq!(config.lint.min_score, Some(0.75));
    }

    #[test]
    fn resolve_defaults() {
        let config = PvConfig::default();
        let map = resolve_rule_severities(&config, &[], false);
        assert_eq!(map.get("PV-VAL-001"), Some(&RuleSeverity::Error));
        assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Warning));
        assert_eq!(map.get("PV-AUD-002"), Some(&RuleSeverity::Info));
    }

    #[test]
    fn resolve_config_override() {
        let mut config = PvConfig::default();
        config
            .lint
            .rules
            .insert("PV-AUD-001".into(), "error".into());
        let map = resolve_rule_severities(&config, &[], false);
        assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Error));
    }

    #[test]
    fn resolve_cli_override_wins() {
        let mut config = PvConfig::default();
        config
            .lint
            .rules
            .insert("PV-AUD-001".into(), "error".into());
        let cli = vec![("PV-AUD-001".into(), "off".into())];
        let map = resolve_rule_severities(&config, &cli, false);
        assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Off));
    }

    #[test]
    fn resolve_strict_mode() {
        let config = PvConfig::default();
        let map = resolve_rule_severities(&config, &[], true);
        // Warnings should become errors in strict mode
        assert_eq!(map.get("PV-AUD-001"), Some(&RuleSeverity::Error));
        // Errors stay errors
        assert_eq!(map.get("PV-VAL-001"), Some(&RuleSeverity::Error));
        // Info stays info
        assert_eq!(map.get("PV-AUD-002"), Some(&RuleSeverity::Info));
    }

    #[test]
    fn find_config_nonexistent() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_config(tmp.path()).is_none());
    }

    #[test]
    fn find_config_local() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".pv.toml"), "[lint]\nmin_score = 0.5\n").unwrap();
        let found = find_config(tmp.path());
        assert!(found.is_some());
    }

    #[test]
    fn load_config_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".pv.toml");
        std::fs::write(
            &path,
            "[lint]\nmin_score = 0.75\nstrict = true\n\n[lint.rules]\nPV-VAL-001 = \"error\"\n",
        )
        .unwrap();
        let config = load_config(&path).unwrap();
        assert_eq!(config.lint.min_score, Some(0.75));
        assert!(config.lint.strict);
    }

    #[test]
    fn load_config_missing_file() {
        let result = load_config(Path::new("/nonexistent/.pv.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn trend_defaults() {
        let t = TrendSection::default();
        assert!(t.enabled);
        assert_eq!(t.retention_days, 90);
        assert!((t.drift_threshold - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_defaults() {
        let c = CacheSection::default();
        assert!(c.enabled);
        assert!(c.dir.is_none());
    }

    #[test]
    fn config_serializes() {
        let config = PvConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"lint\""));
        assert!(json.contains("\"output\""));
    }
}
