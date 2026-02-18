//! End-to-end codegen — generates all artifacts to disk.
//!
//! Combines scaffold, kani, and probar generators into a single
//! operation that writes files to a target directory.

use std::path::{Path, PathBuf};

use crate::binding::BindingRegistry;
use crate::kani_gen::generate_kani_harnesses;
use crate::probar_gen::{generate_probar_tests, generate_wired_probar_tests};
use crate::scaffold::generate_trait;
use crate::schema::Contract;

/// Manifest of generated files.
#[derive(Debug, Clone)]
pub struct GeneratedFiles {
    /// Files that were generated.
    pub files: Vec<GeneratedFile>,
}

/// A single generated file.
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    /// Path relative to the output directory.
    pub relative_path: PathBuf,
    /// Absolute path where the file was written.
    pub absolute_path: PathBuf,
    /// What kind of artifact this is.
    pub kind: ArtifactKind,
    /// Number of bytes written.
    pub bytes: usize,
}

/// The kind of generated artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Scaffold,
    KaniHarness,
    ProbarTest,
    WiredProbarTest,
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scaffold => write!(f, "scaffold"),
            Self::KaniHarness => write!(f, "kani"),
            Self::ProbarTest => write!(f, "probar"),
            Self::WiredProbarTest => write!(f, "wired-probar"),
        }
    }
}

/// Generate all artifacts from a contract and write to `output_dir`.
///
/// Creates the output directory if it doesn't exist. Generates:
/// - `scaffold.rs` — trait + test stubs
/// - `kani.rs` — `#[kani::proof]` harnesses
/// - `probar.rs` — property-based tests
/// - `wired_probar.rs` — wired tests (only if binding provided)
///
/// # Errors
///
/// Returns `io::Error` if directory creation or file writes fail.
pub fn generate_all(
    contract: &Contract,
    stem: &str,
    output_dir: &Path,
    binding: Option<&BindingRegistry>,
) -> Result<GeneratedFiles, std::io::Error> {
    std::fs::create_dir_all(output_dir)?;

    let mut files = Vec::new();

    // Scaffold
    let scaffold_content = generate_trait(contract);
    let scaffold_path = output_dir.join(format!("{stem}_scaffold.rs"));
    std::fs::write(&scaffold_path, &scaffold_content)?;
    files.push(GeneratedFile {
        relative_path: PathBuf::from(format!("{stem}_scaffold.rs")),
        absolute_path: scaffold_path,
        kind: ArtifactKind::Scaffold,
        bytes: scaffold_content.len(),
    });

    // Kani harnesses
    let kani_content = generate_kani_harnesses(contract);
    let kani_path = output_dir.join(format!("{stem}_kani.rs"));
    std::fs::write(&kani_path, &kani_content)?;
    files.push(GeneratedFile {
        relative_path: PathBuf::from(format!("{stem}_kani.rs")),
        absolute_path: kani_path,
        kind: ArtifactKind::KaniHarness,
        bytes: kani_content.len(),
    });

    // Probar tests
    let probar_content = generate_probar_tests(contract);
    let probar_path = output_dir.join(format!("{stem}_probar.rs"));
    std::fs::write(&probar_path, &probar_content)?;
    files.push(GeneratedFile {
        relative_path: PathBuf::from(format!("{stem}_probar.rs")),
        absolute_path: probar_path,
        kind: ArtifactKind::ProbarTest,
        bytes: probar_content.len(),
    });

    // Wired probar tests (only if binding provided)
    if let Some(reg) = binding {
        let contract_file = format!("{stem}.yaml");
        let wired_content = generate_wired_probar_tests(contract, &contract_file, reg);
        let wired_path = output_dir.join(format!("{stem}_wired_probar.rs"));
        std::fs::write(&wired_path, &wired_content)?;
        files.push(GeneratedFile {
            relative_path: PathBuf::from(format!("{stem}_wired_probar.rs")),
            absolute_path: wired_path,
            kind: ArtifactKind::WiredProbarTest,
            bytes: wired_content.len(),
        });
    }

    Ok(GeneratedFiles { files })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn minimal_contract() -> Contract {
        parse_contract_str(
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
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
        )
        .unwrap()
    }

    #[test]
    fn generate_all_without_binding() {
        let c = minimal_contract();
        let dir = tempfile::tempdir().unwrap();
        let result = generate_all(&c, "test-kernel-v1", dir.path(), None).unwrap();
        assert_eq!(result.files.len(), 3);
        assert!(result.files.iter().any(|f| f.kind == ArtifactKind::Scaffold));
        assert!(result
            .files
            .iter()
            .any(|f| f.kind == ArtifactKind::KaniHarness));
        assert!(result
            .files
            .iter()
            .any(|f| f.kind == ArtifactKind::ProbarTest));
        for f in &result.files {
            assert!(f.absolute_path.exists());
            assert!(f.bytes > 0);
        }
    }

    #[test]
    fn generate_all_with_binding() {
        let c = minimal_contract();
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test-kernel-v1.yaml
    equation: f
    module_path: "test::f"
    function: f
    signature: "fn f(x: &[f32]) -> Vec<f32>"
    status: implemented
"#,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let result = generate_all(&c, "test-kernel-v1", dir.path(), Some(&binding)).unwrap();
        assert_eq!(result.files.len(), 4);
        assert!(result
            .files
            .iter()
            .any(|f| f.kind == ArtifactKind::WiredProbarTest));
    }

    #[test]
    fn generates_into_subdir() {
        let c = minimal_contract();
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("deep").join("nested");
        let result = generate_all(&c, "test-kernel-v1", &sub, None).unwrap();
        assert_eq!(result.files.len(), 3);
        assert!(sub.exists());
    }

    #[test]
    fn artifact_kind_display() {
        assert_eq!(ArtifactKind::Scaffold.to_string(), "scaffold");
        assert_eq!(ArtifactKind::KaniHarness.to_string(), "kani");
        assert_eq!(ArtifactKind::ProbarTest.to_string(), "probar");
        assert_eq!(ArtifactKind::WiredProbarTest.to_string(), "wired-probar");
    }

    #[test]
    fn file_names_use_stem() {
        let c = minimal_contract();
        let dir = tempfile::tempdir().unwrap();
        let result = generate_all(&c, "softmax-kernel-v1", dir.path(), None).unwrap();
        let names: Vec<String> = result
            .files
            .iter()
            .map(|f| f.relative_path.to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"softmax-kernel-v1_scaffold.rs".to_string()));
        assert!(names.contains(&"softmax-kernel-v1_kani.rs".to_string()));
        assert!(names.contains(&"softmax-kernel-v1_probar.rs".to_string()));
    }
}
