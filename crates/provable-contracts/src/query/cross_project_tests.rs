// Tests for cross-project contract usage index.

use super::*;

#[test]
fn extract_contract_stem_basic() {
    let content = r#"#[contract("softmax-kernel-v1", equation = "softmax")]"#;
    assert_eq!(
        extract_contract_stem(content),
        Some("softmax-kernel-v1".to_string())
    );
}

#[test]
fn extract_contract_stem_with_yaml() {
    let content = r#"contract("rmsnorm-kernel-v1.yaml", equation = "rmsnorm")"#;
    assert_eq!(
        extract_contract_stem(content),
        Some("rmsnorm-kernel-v1".to_string())
    );
}

#[test]
fn extract_equation_basic() {
    let content = r#"#[contract("softmax-kernel-v1", equation = "softmax")]"#;
    assert_eq!(extract_equation(content), Some("softmax".to_string()));
}

#[test]
fn extract_equation_missing() {
    let content = r#"#[contract("softmax-kernel-v1")]"#;
    assert_eq!(extract_equation(content), None);
}

#[test]
fn extract_patterns_kaizen() {
    let patterns = extract_patterns("// KAIZEN-050: fused softmax backward");
    assert_eq!(patterns, vec!["KAIZEN-050"]);
}

#[test]
fn extract_patterns_contract_id() {
    let patterns = extract_patterns("// C-XENT-002: refs softmax-kernel-v1");
    assert_eq!(patterns, vec!["C-XENT-002"]);
}

#[test]
fn extract_patterns_multiple() {
    let patterns = extract_patterns("KAIZEN-050 and KAIZEN-051 and C-SM-001");
    assert_eq!(patterns.len(), 3);
    assert!(patterns.contains(&"KAIZEN-050".to_string()));
    assert!(patterns.contains(&"KAIZEN-051".to_string()));
    assert!(patterns.contains(&"C-SM-001".to_string()));
}

#[test]
fn extract_patterns_none() {
    let patterns = extract_patterns("just regular code");
    assert!(patterns.is_empty());
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

#[test]
fn discover_real_sibling_projects() {
    let root = repo_root();
    let parent = root.parent().unwrap();
    let projects = discover_projects(parent, &root);
    let names: Vec<&str> = projects.iter().map(|p| p.name.as_str()).collect();
    assert!(
        names.contains(&"aprender"),
        "Should discover aprender, found: {names:?}"
    );
}

#[test]
fn build_cross_project_index() {
    let index = CrossProjectIndex::build(&repo_root());
    assert!(index.project_count() > 0, "Should discover projects");
    assert!(
        index.total_call_sites() > 0,
        "Should find contract call sites"
    );
}

#[test]
fn call_sites_for_known_contract() {
    let index = CrossProjectIndex::build(&repo_root());
    let sites = index.call_sites_for("metrics-regression-v1");
    assert!(
        !sites.is_empty(),
        "Should find call sites for metrics-regression-v1"
    );
    assert_eq!(sites[0].project, "aprender");
}

#[test]
fn binding_refs_for_aprender() {
    let index = CrossProjectIndex::build(&repo_root());
    let refs = index.binding_refs_for("softmax-kernel-v1");
    assert!(!refs.is_empty(), "Should find binding ref for softmax");
}

#[test]
fn kaizen_refs_in_trueno() {
    let index = CrossProjectIndex::build(&repo_root());
    let refs = index.kaizen_refs_for("KAIZEN-015");
    assert!(!refs.is_empty(), "Should find KAIZEN-015 in trueno");
}

#[test]
fn parse_contract_annotation_line() {
    let line = "/home/user/aprender/src/metrics/mod.rs:39:#[provable_contracts_macros::contract(\"metrics-regression-v1\", equation = \"r_squared\")]";
    let project_path = Path::new("/home/user/aprender");
    let site = parse_contract_annotation(line, "aprender", project_path).unwrap();
    assert_eq!(site.contract_stem, "metrics-regression-v1");
    assert_eq!(site.equation, Some("r_squared".to_string()));
    assert_eq!(site.line, 39);
    assert_eq!(site.project, "aprender");
}

#[test]
fn find_binding_path_real() {
    let root = repo_root();
    let aprender_dir = root.parent().unwrap().join("aprender");
    if aprender_dir.exists() {
        let bp = find_binding_path(&aprender_dir, "aprender");
        assert!(bp.is_some(), "Should find binding.yaml for aprender");
    }
}

#[test]
fn call_sites_for_unknown_contract() {
    let index = CrossProjectIndex::build(&repo_root());
    let sites = index.call_sites_for("nonexistent-contract-v1");
    assert!(sites.is_empty());
}

#[test]
fn cross_project_index_accessors() {
    let index = CrossProjectIndex::build(&repo_root());
    assert!(index.project_count() >= 4, "Should find aprender, trueno, entrenar, bashrs");
    assert!(index.total_call_sites() > 5, "aprender has many contract annotations");
}
