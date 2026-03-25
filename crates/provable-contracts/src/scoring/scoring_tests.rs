// Coverage-targeted tests for contract scoring.

use super::*;
use crate::schema::parse_contract_str;

fn minimal_kernel_yaml() -> &'static str {
    r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
  depends_on: ["other-v1"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
    codomain: "R"
    invariants: ["output finite"]
proof_obligations:
  - type: invariant
    property: "finite"
    tolerance: 1e-6
falsification_tests:
  - id: FALSIFY-001
    rule: "finite"
    prediction: "finite"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: stub_float
    solver: cadical
"#
}

#[test]
fn score_complete_contract() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(score.spec_depth >= 0.70, "spec_depth={}", score.spec_depth);
    assert!((score.falsification_coverage - 1.0).abs() < f64::EPSILON);
    assert!(score.kani_coverage > 0.0);
    assert!(score.lean_coverage.abs() < f64::EPSILON);
    assert!(score.binding_coverage.abs() < f64::EPSILON);
    assert!(score.composite > 0.0);
}

#[test]
fn grade_thresholds() {
    assert_eq!(Grade::from_score(0.95), Grade::A);
    assert_eq!(Grade::from_score(0.80), Grade::B);
    assert_eq!(Grade::from_score(0.65), Grade::C);
    assert_eq!(Grade::from_score(0.45), Grade::D);
    assert_eq!(Grade::from_score(0.30), Grade::F);
}

#[test]
fn custom_weights_change_composite() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let default = score_contract(&contract, None, "test-v1");
    let kani_heavy = score_contract_weighted(
        &contract,
        None,
        "test-v1",
        &ScoringWeights {
            spec_depth: 0.05,
            falsification: 0.05,
            kani: 0.70,
            lean: 0.10,
            binding: 0.10,
        },
    );
    // Different weights should produce different composites
    assert!(
        (default.composite - kani_heavy.composite).abs() > 0.01,
        "default={} kani_heavy={}",
        default.composite,
        kani_heavy.composite
    );
    // Individual dimensions should be the same
    assert!((default.spec_depth - kani_heavy.spec_depth).abs() < f64::EPSILON);
    assert!((default.kani_coverage - kani_heavy.kani_coverage).abs() < f64::EPSILON);
}

#[test]
fn weights_normalization() {
    let w = ScoringWeights {
        spec_depth: 2.0,
        falsification: 2.0,
        kani: 2.0,
        lean: 2.0,
        binding: 2.0,
    };
    let n = w.normalized();
    let total = n.spec_depth + n.falsification + n.kani + n.lean + n.binding;
    assert!((total - 1.0).abs() < 1e-9, "total={total}");
    assert!((n.spec_depth - 0.2).abs() < 1e-9);
}

#[test]
fn empty_contract_scores_low() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty"
  registry: true
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "empty-v1");
    assert!(score.composite < 0.40, "composite={}", score.composite);
    assert_eq!(score.grade, Grade::F);
}

#[test]
fn score_real_softmax_contract() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/softmax-kernel-v1.yaml");
    let contract = crate::schema::parse_contract(&path).unwrap();
    let score = score_contract(&contract, None, "softmax-kernel-v1");
    assert!(
        score.composite > 0.40,
        "softmax should score well: {}",
        score.composite
    );
    assert!(score.spec_depth > 0.50);
    assert!(score.falsification_coverage > 0.0);
    assert!(score.kani_coverage > 0.0);
}

#[test]
fn grade_display_all_variants() {
    assert_eq!(format!("{}", Grade::A), "A");
    assert_eq!(format!("{}", Grade::B), "B");
    assert_eq!(format!("{}", Grade::C), "C");
    assert_eq!(format!("{}", Grade::D), "D");
    assert_eq!(format!("{}", Grade::F), "F");
}

#[test]
fn contract_score_display() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/softmax-kernel-v1.yaml");
    let contract = crate::schema::parse_contract(&path).unwrap();
    let score = score_contract(&contract, None, "softmax-kernel-v1");
    let text = format!("{score}");
    assert!(
        text.contains("softmax-kernel-v1"),
        "Display should include stem"
    );
    assert!(text.contains("Grade"), "Display should include Grade");
    assert!(
        text.contains("Spec:"),
        "Display should include dimension breakdown"
    );
}

#[test]
fn kani_compositional_strategy_weight() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test compositional"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "finite"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: compositional
    solver: cadical
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(
        (score.kani_coverage - 0.7).abs() < 1e-9,
        "got {}",
        score.kani_coverage
    );
}

#[test]
fn kani_no_strategy_weight() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test no strategy"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "finite"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    solver: cadical
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(
        (score.kani_coverage - 0.5).abs() < 1e-9,
        "got {}",
        score.kani_coverage
    );
}

#[test]
fn binding_partial_status_coverage() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        bindings: vec![crate::binding::KernelBinding {
            contract: "test-v1".into(),
            equation: "f".into(),
            module_path: Some("test::f".into()),
            function: Some("f".into()),
            signature: None,
            status: ImplStatus::Partial,
            notes: None,
        }],
    };
    let score = score_contract(&contract, Some(&binding), "test-v1");
    assert!(
        (score.binding_coverage - 0.5).abs() < 1e-9,
        "Partial = 0.5, got {}",
        score.binding_coverage
    );
}

#[test]
fn binding_no_relevant_entries() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        bindings: vec![crate::binding::KernelBinding {
            contract: "other-v1".into(),
            equation: "g".into(),
            module_path: None,
            function: None,
            signature: None,
            status: ImplStatus::Implemented,
            notes: None,
        }],
    };
    let score = score_contract(&contract, Some(&binding), "test-v1");
    assert!(score.binding_coverage.abs() < f64::EPSILON);
}

#[test]
fn kani_coverage_with_no_obligations() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: exhaustive
    solver: cadical
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!((score.kani_coverage - 1.0).abs() < f64::EPSILON);
}

#[test]
fn probes_populated_for_complete_contract() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(!score.probes.is_empty(), "probes should be populated");

    // Should have spec_depth probes
    let spec_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "spec_depth")
        .collect();
    assert!(
        spec_probes.len() >= 5,
        "spec_depth should have at least 5 probes, got {}",
        spec_probes.len()
    );

    // has_equations should pass
    let eq_probe = spec_probes
        .iter()
        .find(|p| p.probe == "has_equations")
        .expect("should have has_equations probe");
    assert!(eq_probe.outcome, "has_equations should pass");
    assert!(eq_probe.detail.contains("1 equation"), "detail: {}", eq_probe.detail);
}

#[test]
fn probes_kani_shows_harness_id() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let kani_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "kani")
        .collect();
    assert!(!kani_probes.is_empty(), "should have kani probes");

    // The obligation "finite" should match harness KANI-001
    let finite_probe = kani_probes
        .iter()
        .find(|p| p.probe == "finite")
        .expect("should have probe for 'finite' obligation");
    assert!(finite_probe.outcome, "finite should have a harness");
    assert!(
        finite_probe.detail.contains("KANI-001"),
        "detail should mention KANI-001, got: {}",
        finite_probe.detail
    );
}

#[test]
fn probes_falsification_shows_test_id() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let fals_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "falsification")
        .collect();
    assert!(!fals_probes.is_empty(), "should have falsification probes");

    // The obligation "finite" should match test FALSIFY-001 (rule="finite")
    let finite_probe = fals_probes
        .iter()
        .find(|p| p.probe == "finite")
        .expect("should have probe for 'finite' obligation");
    assert!(finite_probe.outcome, "finite should have a test");
    assert!(
        finite_probe.detail.contains("FALSIFY-001"),
        "detail should mention FALSIFY-001, got: {}",
        finite_probe.detail
    );
}

#[test]
fn probes_lean_no_obligations() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let lean_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "lean")
        .collect();
    // The minimal YAML has one obligation but no lean metadata
    assert!(!lean_probes.is_empty(), "should have lean probes");
    let probe = &lean_probes[0];
    assert!(!probe.outcome, "should fail (no lean_theorem)");
    assert!(
        probe.detail.contains("no lean_theorem"),
        "detail: {}",
        probe.detail
    );
}

#[test]
fn probes_binding_no_registry() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let bind_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "binding")
        .collect();
    assert!(!bind_probes.is_empty(), "should have binding probes");
    let probe = &bind_probes[0];
    assert!(!probe.outcome, "should fail (no binding registry)");
    assert!(
        probe.detail.contains("no binding registry"),
        "detail: {}",
        probe.detail
    );
}

#[test]
fn probes_binding_with_partial_status() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
  g:
    formula: "g(x) = x^2"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        bindings: vec![
            crate::binding::KernelBinding {
                contract: "test-v1".into(),
                equation: "f".into(),
                module_path: Some("test::f".into()),
                function: Some("f".into()),
                signature: None,
                status: ImplStatus::Implemented,
                notes: None,
            },
            crate::binding::KernelBinding {
                contract: "test-v1".into(),
                equation: "g".into(),
                module_path: Some("test::g".into()),
                function: Some("g".into()),
                signature: None,
                status: ImplStatus::Partial,
                notes: None,
            },
        ],
    };
    let score = score_contract(&contract, Some(&binding), "test-v1");

    let bind_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "binding")
        .collect();
    assert_eq!(bind_probes.len(), 2, "should have 2 binding probes");

    let f_probe = bind_probes
        .iter()
        .find(|p| p.probe == "f")
        .expect("should have probe for equation 'f'");
    assert!(f_probe.outcome, "f should be implemented");
    assert!(f_probe.detail.contains("implemented"), "detail: {}", f_probe.detail);

    let g_probe = bind_probes
        .iter()
        .find(|p| p.probe == "g")
        .expect("should have probe for equation 'g'");
    assert!(!g_probe.outcome, "g is partial, not fully implemented");
    assert!(g_probe.detail.contains("partial"), "detail: {}", g_probe.detail);
}

#[test]
fn probes_serialization_skip_empty() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    // When probes are present, they should serialize
    let json = serde_json::to_string(&score).unwrap();
    assert!(json.contains("probes"), "JSON should include probes when non-empty");

    // Manually construct a ContractScore with empty probes to verify skip_serializing_if
    let empty_score = ContractScore {
        stem: "empty".into(),
        spec_depth: 0.0,
        falsification_coverage: 0.0,
        kani_coverage: 0.0,
        lean_coverage: 0.0,
        binding_coverage: 0.0,
        composite: 0.0,
        grade: Grade::F,
        probes: vec![],
    };
    let json = serde_json::to_string(&empty_score).unwrap();
    assert!(!json.contains("probes"), "JSON should omit probes when empty");
}

#[test]
fn probes_spec_depth_per_equation() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
    invariants: ["finite"]
  g:
    formula: "g(x) = x^2"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let spec_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "spec_depth")
        .collect();

    // f should have domain=true and invariants=true
    let f_domain = spec_probes
        .iter()
        .find(|p| p.probe == "f: domain")
        .expect("should have f: domain probe");
    assert!(f_domain.outcome, "f has domain");

    let f_inv = spec_probes
        .iter()
        .find(|p| p.probe == "f: invariants")
        .expect("should have f: invariants probe");
    assert!(f_inv.outcome, "f has invariants");

    // g should have domain=false and invariants=false
    let g_domain = spec_probes
        .iter()
        .find(|p| p.probe == "g: domain")
        .expect("should have g: domain probe");
    assert!(!g_domain.outcome, "g has no domain");

    let g_inv = spec_probes
        .iter()
        .find(|p| p.probe == "g: invariants")
        .expect("should have g: invariants probe");
    assert!(!g_inv.outcome, "g has no invariants");
}
