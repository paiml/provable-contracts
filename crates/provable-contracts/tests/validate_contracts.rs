use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{parse_contract, validate_contract};

fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .canonicalize()
        .expect("contracts directory must exist")
}

fn all_contract_paths() -> Vec<std::path::PathBuf> {
    let dir = contracts_dir();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                && !path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with('.')
            {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths
}

fn validate_contract_file(path: &Path) {
    assert!(path.exists(), "Contract file not found: {}", path.display());

    let contract =
        parse_contract(path).unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

    let violations = validate_contract(&contract);
    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();

    assert!(
        errors.is_empty(),
        "Contract {} has validation errors: {:?}",
        path.display(),
        errors
    );
}

#[test]
fn validate_softmax_contract() {
    validate_contract_file(&contracts_dir().join("softmax-kernel-v1.yaml"));
}

#[test]
fn validate_rmsnorm_contract() {
    validate_contract_file(&contracts_dir().join("rmsnorm-kernel-v1.yaml"));
}

#[test]
fn validate_rope_contract() {
    validate_contract_file(&contracts_dir().join("rope-kernel-v1.yaml"));
}

#[test]
fn validate_activation_contract() {
    validate_contract_file(&contracts_dir().join("activation-kernel-v1.yaml"));
}

#[test]
fn validate_attention_contract() {
    validate_contract_file(&contracts_dir().join("attention-kernel-v1.yaml"));
}

#[test]
fn validate_matmul_contract() {
    validate_contract_file(&contracts_dir().join("matmul-kernel-v1.yaml"));
}

#[test]
fn validate_flash_attention_contract() {
    validate_contract_file(&contracts_dir().join("flash-attention-v1.yaml"));
}

#[test]
fn validate_all_contracts() {
    let paths = all_contract_paths();
    assert!(
        paths.len() >= 48,
        "Expected at least 48 contracts, found {}",
        paths.len()
    );
    for path in &paths {
        validate_contract_file(path);
    }
}

#[test]
fn qwen35_dag_integrity() {
    let paths = all_contract_paths();
    let contracts: Vec<_> = paths
        .iter()
        .map(|p| {
            let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
            let contract = parse_contract(p).unwrap();
            (stem, contract)
        })
        .collect();

    let refs: Vec<_> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    // No cycles in the full DAG
    assert!(graph.cycles.is_empty(), "DAG has cycles: {:?}", graph.cycles);

    // qwen35-e2e-verification is the capstone (no dependents)
    let e2e = "qwen35-e2e-verification-v1";
    assert!(graph.nodes.contains(e2e), "Missing e2e verification contract");

    // Verify e2e depends on exactly 8 sub-contracts
    let e2e_deps = graph.edges.get(e2e).unwrap();
    assert_eq!(
        e2e_deps.len(),
        8,
        "e2e should depend on 8 contracts, found {}",
        e2e_deps.len()
    );

    // All 7 Qwen 3.5 contracts exist
    let qwen_contracts = [
        "sliding-window-attention-v1",
        "rope-extrapolation-v1",
        "embedding-algebra-v1",
        "inference-pipeline-v1",
        "qwen35-hybrid-forward-v1",
        "attention-scaling-v1",
        "qwen35-e2e-verification-v1",
    ];
    for name in &qwen_contracts {
        assert!(graph.nodes.contains(*name), "Missing Qwen 3.5 contract: {name}");
    }

    // Topological order: foundations before composites
    let topo = &graph.topo_order;
    let softmax_pos = topo.iter().position(|n| n == "softmax-kernel-v1").unwrap();
    let attention_pos = topo.iter().position(|n| n == "attention-kernel-v1").unwrap();
    let e2e_pos = topo.iter().position(|n| n == e2e).unwrap();
    assert!(
        softmax_pos < attention_pos,
        "softmax should come before attention in topo order"
    );
    assert!(
        attention_pos < e2e_pos,
        "attention should come before e2e in topo order"
    );

    // e2e should be last or near-last (no dependents)
    assert_eq!(
        topo.len(),
        graph.nodes.len(),
        "Topo order should contain all nodes"
    );
}
