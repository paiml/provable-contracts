//! Contract diff — detect drift between contract versions.
//!
//! Compares two [`Contract`] values and produces a [`ContractDiff`]
//! listing added, removed, and changed sections. Suggests a semver
//! bump (major, minor, patch) based on the nature of the changes.

use std::collections::BTreeSet;

use crate::schema::Contract;

/// The result of diffing two contracts.
#[derive(Debug, Clone)]
pub struct ContractDiff {
    /// Version in the "old" contract.
    pub old_version: String,
    /// Version in the "new" contract.
    pub new_version: String,
    /// Per-section change summaries.
    pub sections: Vec<SectionDiff>,
    /// Suggested semver bump based on the changes.
    pub suggested_bump: SemverBump,
}

/// A change summary for one section of the contract.
#[derive(Debug, Clone)]
pub struct SectionDiff {
    /// Section name (e.g. "equations", "`proof_obligations`").
    pub section: String,
    /// Items added in the new contract.
    pub added: Vec<String>,
    /// Items removed from the old contract.
    pub removed: Vec<String>,
    /// Items present in both but changed.
    pub changed: Vec<String>,
}

impl SectionDiff {
    /// True if this section has no changes.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Suggested semantic version bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemverBump {
    /// No changes detected.
    None,
    /// Only additive or cosmetic changes.
    Patch,
    /// New equations, obligations, or tests added.
    Minor,
    /// Equations or obligations removed or semantically changed.
    Major,
}

impl std::fmt::Display for SemverBump {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Patch => write!(f, "patch"),
            Self::Minor => write!(f, "minor"),
            Self::Major => write!(f, "major"),
        }
    }
}

/// Diff two contracts and produce a structured change report.
pub fn diff_contracts(old: &Contract, new: &Contract) -> ContractDiff {
    let mut sections = Vec::new();
    let mut max_bump = SemverBump::None;

    // Equations
    let eq_diff = diff_keys(
        "equations",
        &old.equations.keys().cloned().collect(),
        &new.equations.keys().cloned().collect(),
    );
    let eq_changed = diff_equation_changes(old, new);
    let eq_section = SectionDiff {
        section: "equations".to_string(),
        added: eq_diff.0,
        removed: eq_diff.1,
        changed: eq_changed,
    };
    if !eq_section.removed.is_empty() || !eq_section.changed.is_empty() {
        max_bump = SemverBump::Major;
    } else if !eq_section.added.is_empty() {
        max_bump = bump_max(max_bump, SemverBump::Minor);
    }
    sections.push(eq_section);

    // Proof obligations
    let old_obs: BTreeSet<String> = old
        .proof_obligations
        .iter()
        .map(|o| format!("{}:{}", o.obligation_type, o.property))
        .collect();
    let new_obs: BTreeSet<String> = new
        .proof_obligations
        .iter()
        .map(|o| format!("{}:{}", o.obligation_type, o.property))
        .collect();
    let ob_diff = diff_sets("proof_obligations", &old_obs, &new_obs);
    if !ob_diff.removed.is_empty() {
        max_bump = SemverBump::Major;
    } else if !ob_diff.added.is_empty() {
        max_bump = bump_max(max_bump, SemverBump::Minor);
    }
    sections.push(ob_diff);

    // Falsification tests
    let old_ft: BTreeSet<String> = old
        .falsification_tests
        .iter()
        .map(|t| t.id.clone())
        .collect();
    let new_ft: BTreeSet<String> = new
        .falsification_tests
        .iter()
        .map(|t| t.id.clone())
        .collect();
    let ft_diff = diff_sets("falsification_tests", &old_ft, &new_ft);
    if !ft_diff.removed.is_empty() || !ft_diff.added.is_empty() {
        max_bump = bump_max(max_bump, SemverBump::Minor);
    }
    sections.push(ft_diff);

    // Kani harnesses
    let old_kh: BTreeSet<String> = old
        .kani_harnesses
        .iter()
        .map(|h| h.id.clone())
        .collect();
    let new_kh: BTreeSet<String> = new
        .kani_harnesses
        .iter()
        .map(|h| h.id.clone())
        .collect();
    let kh_diff = diff_sets("kani_harnesses", &old_kh, &new_kh);
    if !kh_diff.removed.is_empty() || !kh_diff.added.is_empty() {
        max_bump = bump_max(max_bump, SemverBump::Minor);
    }
    sections.push(kh_diff);

    // Enforcement rules
    let old_enf: BTreeSet<String> = old.enforcement.keys().cloned().collect();
    let new_enf: BTreeSet<String> = new.enforcement.keys().cloned().collect();
    let enf_diff = diff_sets("enforcement", &old_enf, &new_enf);
    if !enf_diff.removed.is_empty() {
        max_bump = bump_max(max_bump, SemverBump::Minor);
    } else if !enf_diff.added.is_empty() {
        max_bump = bump_max(max_bump, SemverBump::Patch);
    }
    sections.push(enf_diff);

    // Metadata version change
    if old.metadata.version != new.metadata.version {
        max_bump = bump_max(max_bump, SemverBump::Patch);
    }

    ContractDiff {
        old_version: old.metadata.version.clone(),
        new_version: new.metadata.version.clone(),
        sections,
        suggested_bump: max_bump,
    }
}

fn diff_keys(
    _section: &str,
    old_keys: &BTreeSet<String>,
    new_keys: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    let added: Vec<String> = new_keys.difference(old_keys).cloned().collect();
    let removed: Vec<String> = old_keys.difference(new_keys).cloned().collect();
    (added, removed)
}

fn diff_sets(section: &str, old: &BTreeSet<String>, new: &BTreeSet<String>) -> SectionDiff {
    let added: Vec<String> = new.difference(old).cloned().collect();
    let removed: Vec<String> = old.difference(new).cloned().collect();
    SectionDiff {
        section: section.to_string(),
        added,
        removed,
        changed: Vec::new(),
    }
}

fn diff_equation_changes(old: &Contract, new: &Contract) -> Vec<String> {
    let mut changed = Vec::new();
    for (name, old_eq) in &old.equations {
        if let Some(new_eq) = new.equations.get(name) {
            if old_eq.formula != new_eq.formula {
                changed.push(format!("{name}: formula changed"));
            }
            if old_eq.domain != new_eq.domain {
                changed.push(format!("{name}: domain changed"));
            }
            if old_eq.codomain != new_eq.codomain {
                changed.push(format!("{name}: codomain changed"));
            }
            if old_eq.invariants != new_eq.invariants {
                changed.push(format!("{name}: invariants changed"));
            }
        }
    }
    changed
}

fn bump_max(current: SemverBump, candidate: SemverBump) -> SemverBump {
    let rank = |b: SemverBump| match b {
        SemverBump::None => 0,
        SemverBump::Patch => 1,
        SemverBump::Minor => 2,
        SemverBump::Major => 3,
    };
    if rank(candidate) > rank(current) {
        candidate
    } else {
        current
    }
}

/// True if the diff contains no changes at all.
pub fn is_identical(diff: &ContractDiff) -> bool {
    diff.sections.iter().all(SectionDiff::is_empty) && diff.suggested_bump == SemverBump::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn minimal_contract(version: &str) -> Contract {
        parse_contract_str(&format!(
            r#"
metadata:
  version: "{version}"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#
        ))
        .unwrap()
    }

    #[test]
    fn identical_contracts() {
        let c = minimal_contract("1.0.0");
        let diff = diff_contracts(&c, &c);
        assert!(is_identical(&diff));
        assert_eq!(diff.suggested_bump, SemverBump::None);
    }

    #[test]
    fn added_equation() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
  g:
    formula: "g(x) = x^2"
falsification_tests: []
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert_eq!(eq_section.added, vec!["g"]);
        assert!(eq_section.removed.is_empty());
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn removed_equation() {
        let old = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
  g:
    formula: "g(x) = x^2"
falsification_tests: []
"#,
        )
        .unwrap();
        let new = minimal_contract("1.0.0");
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert_eq!(eq_section.removed, vec!["g"]);
        assert_eq!(diff.suggested_bump, SemverBump::Major);
    }

    #[test]
    fn changed_formula() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x + 1"
falsification_tests: []
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert!(!eq_section.changed.is_empty());
        assert_eq!(diff.suggested_bump, SemverBump::Major);
    }

    #[test]
    fn added_obligation() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let ob_section = &diff.sections[1];
        assert_eq!(ob_section.added.len(), 1);
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn removed_obligation_is_major() {
        let old = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#,
        )
        .unwrap();
        let new = minimal_contract("1.0.0");
        let diff = diff_contracts(&old, &new);
        let ob_section = &diff.sections[1];
        assert_eq!(ob_section.removed.len(), 1);
        assert_eq!(diff.suggested_bump, SemverBump::Major);
    }

    #[test]
    fn version_change_is_patch() {
        let old = minimal_contract("1.0.0");
        let new = minimal_contract("1.0.1");
        let diff = diff_contracts(&old, &new);
        assert_eq!(diff.old_version, "1.0.0");
        assert_eq!(diff.new_version, "1.0.1");
        assert_eq!(diff.suggested_bump, SemverBump::Patch);
    }

    #[test]
    fn added_falsification_test() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let ft_section = &diff.sections[2];
        assert_eq!(ft_section.added, vec!["FALSIFY-001"]);
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn semver_bump_display() {
        assert_eq!(SemverBump::None.to_string(), "none");
        assert_eq!(SemverBump::Patch.to_string(), "patch");
        assert_eq!(SemverBump::Minor.to_string(), "minor");
        assert_eq!(SemverBump::Major.to_string(), "major");
    }

    #[test]
    fn section_diff_is_empty() {
        let s = SectionDiff {
            section: "test".to_string(),
            added: vec![],
            removed: vec![],
            changed: vec![],
        };
        assert!(s.is_empty());

        let s2 = SectionDiff {
            section: "test".to_string(),
            added: vec!["x".to_string()],
            removed: vec![],
            changed: vec![],
        };
        assert!(!s2.is_empty());
    }

    #[test]
    fn added_kani_harness() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let kh_section = &diff.sections[3];
        assert_eq!(kh_section.added, vec!["KANI-001"]);
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn enforcement_added() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
enforcement:
  rule1:
    description: "must hold"
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let enf_section = &diff.sections[4];
        assert_eq!(enf_section.added, vec!["rule1"]);
        assert_eq!(diff.suggested_bump, SemverBump::Patch);
    }

    #[test]
    fn domain_change_detected() {
        let old = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
falsification_tests: []
"#,
        )
        .unwrap();
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R^n"
falsification_tests: []
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert!(eq_section.changed.iter().any(|c| c.contains("domain")));
    }
}
