//! Scoring types: contract scores, codebase scores, grades.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Score for a single contract across 5 dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractScore {
    pub stem: String,
    pub spec_depth: f64,
    pub falsification_coverage: f64,
    pub kani_coverage: f64,
    pub lean_coverage: f64,
    pub binding_coverage: f64,
    pub composite: f64,
    pub grade: Grade,
}

/// Score for a codebase that consumes contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseScore {
    pub path: String,
    pub contract_coverage: f64,
    pub binding_completeness: f64,
    pub mean_contract_score: f64,
    pub proof_depth_dist: f64,
    pub drift: f64,
    pub composite: f64,
    pub grade: Grade,
    pub top_gaps: Vec<ScoringGap>,
}

/// A gap identified by the scoring system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringGap {
    pub contract: String,
    pub dimension: String,
    pub current: f64,
    pub target: f64,
    pub impact: f64,
}

/// Letter grade A-F.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    A,
    B,
    C,
    D,
    F,
}

impl Grade {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.90 {
            Grade::A
        } else if score >= 0.75 {
            Grade::B
        } else if score >= 0.60 {
            Grade::C
        } else if score >= 0.40 {
            Grade::D
        } else {
            Grade::F
        }
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Grade::A => write!(f, "A"),
            Grade::B => write!(f, "B"),
            Grade::C => write!(f, "C"),
            Grade::D => write!(f, "D"),
            Grade::F => write!(f, "F"),
        }
    }
}

impl fmt::Display for ContractScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} — {:.2} (Grade {})",
            self.stem, self.composite, self.grade
        )?;
        writeln!(
            f,
            "  Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}",
            self.spec_depth,
            self.falsification_coverage,
            self.kani_coverage,
            self.lean_coverage,
            self.binding_coverage
        )
    }
}
