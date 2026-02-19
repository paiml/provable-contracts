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

#[test]
fn contract_data_integrity() {
    let paths = all_contract_paths();
    let mut total_eq = 0usize;
    let mut total_ob = 0usize;
    let mut total_ft = 0usize;
    let mut total_kani = 0usize;
    let mut errors = Vec::new();

    for path in &paths {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let contract = parse_contract(path)
            .unwrap_or_else(|e| panic!("Failed to parse {stem}: {e}"));

        let eq_count = contract.equations.len();
        let ob_count = contract.proof_obligations.len();
        let ft_count = contract.falsification_tests.len();
        let kani_count = contract.kani_harnesses.len();

        total_eq += eq_count;
        total_ob += ob_count;
        total_ft += ft_count;
        total_kani += kani_count;

        // Every contract must have at least 1 equation
        if eq_count == 0 {
            errors.push(format!("{stem}: no equations"));
        }

        // Every obligation must have at least one falsification test
        if ft_count < ob_count {
            errors.push(format!(
                "{stem}: obligations ({ob_count}) > falsification tests ({ft_count})"
            ));
        }

        // Every contract must have at least 1 Kani harness
        if kani_count == 0 {
            errors.push(format!("{stem}: no Kani harnesses"));
        }

        // Falsification test IDs must be sequential
        let prefix_pattern: Vec<&str> = contract
            .falsification_tests
            .first()
            .map(|ft| ft.id.rsplitn(2, '-').collect::<Vec<_>>())
            .unwrap_or_default();
        if prefix_pattern.len() == 2 {
            let prefix = prefix_pattern[1]; // "FALSIFY-XX"
            for (i, ft) in contract.falsification_tests.iter().enumerate() {
                let expected = format!("{prefix}-{:03}", i + 1);
                if ft.id != expected {
                    errors.push(format!(
                        "{stem}: test ID gap: expected {expected}, found {}",
                        ft.id
                    ));
                    break; // One ID error per contract is enough
                }
            }
        }

        // pass_criteria number must match actual test count
        if let Some(qa) = &contract.qa_gate {
            if let Some(criteria) = &qa.pass_criteria {
                // Extract "All N" pattern
                if let Some(n_str) = criteria
                    .strip_prefix("All ")
                    .and_then(|s| s.split_whitespace().next())
                {
                    if let Ok(n) = n_str.parse::<usize>() {
                        if n != ft_count {
                            errors.push(format!(
                                "{stem}: pass_criteria says {n} tests, \
                                 actual {ft_count}"
                            ));
                        }
                    }
                }
            }
        }
    }

    // Verify totals
    assert_eq!(total_eq, 234, "Total equations changed");
    assert_eq!(total_ob, 379, "Total obligations changed");
    assert_eq!(total_ft, 399, "Total falsification tests changed");
    assert_eq!(total_kani, 126, "Total Kani harnesses changed");

    assert!(
        errors.is_empty(),
        "Data integrity violations:\n{}",
        errors.join("\n")
    );
}
