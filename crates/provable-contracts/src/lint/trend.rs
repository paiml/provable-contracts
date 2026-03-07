//! Quality trend tracking — record and display lint snapshots over time.
//!
//! Each `--trend` invocation records a timestamped snapshot to
//! `.pv/trend/`. Historical snapshots enable drift detection
//! and quality regression alerts.
//!
//! Spec: `docs/specifications/sub/lint.md` Section 9

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::LintReport;

/// A single point-in-time lint snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSnapshot {
    pub timestamp: String,
    pub commit: Option<String>,
    pub total_contracts: usize,
    pub errors: usize,
    pub warnings: usize,
    pub mean_score: f64,
    pub findings_count: usize,
    pub passed: bool,
}

/// Get the trend directory path.
pub fn trend_dir(base: &Path) -> PathBuf {
    base.join(".pv").join("trend")
}

/// Record a lint report as a trend snapshot.
pub fn record_snapshot(
    trend_root: &Path,
    report: &LintReport,
    contracts_count: usize,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(trend_root)
        .map_err(|e| format!("Failed to create trend dir: {e}"))?;

    let timestamp = now_iso8601();
    let commit = current_commit().ok();

    let (errors, warnings) = count_by_severity(&report.findings);
    let mean_score = extract_mean_score(report);

    let snapshot = TrendSnapshot {
        timestamp: timestamp.clone(),
        commit,
        total_contracts: contracts_count,
        errors,
        warnings,
        mean_score,
        findings_count: report.findings.len(),
        passed: report.passed,
    };

    let filename = format!("{}.json", timestamp.replace(':', "-"));
    let path = trend_root.join(&filename);
    let json = serde_json::to_string_pretty(&snapshot)
        .map_err(|e| format!("Failed to serialize snapshot: {e}"))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Failed to write snapshot: {e}"))?;

    Ok(path)
}

/// Load all trend snapshots, sorted by timestamp.
pub fn load_snapshots(trend_root: &Path) -> Vec<TrendSnapshot> {
    let Ok(entries) = std::fs::read_dir(trend_root) else {
        return vec![];
    };
    let mut snapshots: Vec<TrendSnapshot> = entries
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                == Some("json")
        })
        .filter_map(|e| {
            let content = std::fs::read_to_string(e.path()).ok()?;
            serde_json::from_str(&content).ok()
        })
        .collect();
    snapshots.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    snapshots
}

/// Check for quality drift: mean score drop >threshold from rolling avg.
pub fn detect_drift(snapshots: &[TrendSnapshot], threshold: f64) -> Option<f64> {
    if snapshots.len() < 2 {
        return None;
    }

    let window = snapshots.len().min(7);
    let recent = &snapshots[snapshots.len().saturating_sub(window)..snapshots.len() - 1];
    if recent.is_empty() {
        return None;
    }

    let avg: f64 = recent.iter().map(|s| s.mean_score).sum::<f64>() / recent.len() as f64;
    let current = snapshots.last()?.mean_score;
    let drop = avg - current;

    if drop > threshold { Some(drop) } else { None }
}

/// Format trend display.
pub fn format_trend(snapshots: &[TrendSnapshot], limit: usize) -> String {
    let mut lines = Vec::new();
    lines.push("Date                 Score  Errors  Warnings  Result".into());
    lines.push("-".repeat(60));

    for snap in snapshots.iter().rev().take(limit) {
        let result = if snap.passed { "PASS" } else { "FAIL" };
        lines.push(format!(
            "{:<20} {:.2}   {:<7} {:<9} {}",
            &snap.timestamp[..19.min(snap.timestamp.len())],
            snap.mean_score,
            snap.errors,
            snap.warnings,
            result,
        ));
    }

    if snapshots.len() >= 2 {
        let first = &snapshots[0];
        let last = snapshots.last().unwrap();
        let delta = last.mean_score - first.mean_score;
        let direction = if delta > 0.01 {
            "improving"
        } else if delta < -0.01 {
            "declining"
        } else {
            "stable"
        };
        lines.push(format!(
            "Trend: {:+.3} score over {} snapshots ({direction})",
            delta,
            snapshots.len()
        ));
    }

    lines.join("\n")
}

fn count_by_severity(findings: &[LintFinding]) -> (usize, usize) {
    let errors = findings
        .iter()
        .filter(|f| f.severity == RuleSeverity::Error && !f.suppressed)
        .count();
    let warnings = findings
        .iter()
        .filter(|f| f.severity == RuleSeverity::Warning && !f.suppressed)
        .count();
    (errors, warnings)
}

fn extract_mean_score(report: &LintReport) -> f64 {
    for gate in &report.gates {
        if let super::GateDetail::Score { mean_score, .. } = &gate.detail {
            return *mean_score;
        }
    }
    0.0
}

fn now_iso8601() -> String {
    // Simple UTC timestamp without chrono dependency
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Approximate: just use epoch seconds formatted
    format!("{secs}")
}

fn current_commit() -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err("not a git repo".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(score: f64, ts: &str) -> TrendSnapshot {
        TrendSnapshot {
            timestamp: ts.into(),
            commit: Some("abc123".into()),
            total_contracts: 107,
            errors: 0,
            warnings: 5,
            mean_score: score,
            findings_count: 5,
            passed: true,
        }
    }

    #[test]
    fn record_and_load_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let trend = tmp.path().join("trend");

        let report = LintReport {
            passed: true,
            gates: vec![],
            total_duration_ms: 10,
            findings: vec![],
        };
        let path = record_snapshot(&trend, &report, 107).unwrap();
        assert!(path.exists());

        let loaded = load_snapshots(&trend);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].total_contracts, 107);
    }

    #[test]
    fn detect_drift_no_data() {
        assert!(detect_drift(&[], 0.05).is_none());
    }

    #[test]
    fn detect_drift_one_snapshot() {
        let snaps = vec![sample_snapshot(0.80, "2026-03-01")];
        assert!(detect_drift(&snaps, 0.05).is_none());
    }

    #[test]
    fn detect_drift_stable() {
        let snaps = vec![
            sample_snapshot(0.80, "2026-03-01"),
            sample_snapshot(0.79, "2026-03-02"),
            sample_snapshot(0.80, "2026-03-03"),
        ];
        assert!(detect_drift(&snaps, 0.05).is_none());
    }

    #[test]
    fn detect_drift_declining() {
        let snaps = vec![
            sample_snapshot(0.80, "2026-03-01"),
            sample_snapshot(0.80, "2026-03-02"),
            sample_snapshot(0.70, "2026-03-03"),
        ];
        let drift = detect_drift(&snaps, 0.05);
        assert!(drift.is_some());
        assert!(drift.unwrap() > 0.05);
    }

    #[test]
    fn format_trend_display() {
        let snaps = vec![
            sample_snapshot(0.75, "2026-03-01T10:00:00"),
            sample_snapshot(0.78, "2026-03-02T10:00:00"),
            sample_snapshot(0.80, "2026-03-03T10:00:00"),
        ];
        let display = format_trend(&snaps, 10);
        assert!(display.contains("2026-03-03"));
        assert!(display.contains("improving"));
    }

    #[test]
    fn format_trend_empty() {
        let display = format_trend(&[], 10);
        assert!(display.contains("Date"));
    }

    #[test]
    fn load_snapshots_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let snaps = load_snapshots(tmp.path());
        assert!(snaps.is_empty());
    }

    #[test]
    fn trend_dir_path() {
        let path = trend_dir(Path::new("/repo"));
        assert_eq!(path, PathBuf::from("/repo/.pv/trend"));
    }
}
