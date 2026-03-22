# Sub-spec: Library API Reference

**Parent:** [pv-spec.md](../pv-spec.md) Section 6

---

## 1. Crate Structure

```
provable-contracts (library) — 30 public modules
+-- schema/           YAML parsing + validation
+-- scaffold/         Trait + test generation
+-- kani_gen/         Kani harness codegen
+-- probar_gen/       Property test codegen
+-- lean_gen/         Lean 4 codegen
+-- audit/            Traceability chain auditing
+-- binding           Contract -> impl mapping
+-- coverage          Cross-contract coverage
+-- diff              Version diffing
+-- generate          End-to-end codegen to disk
+-- graph             Dependency DAG + cycle detection
+-- proof_status      L1-L5 proof levels
+-- book_gen/         mdBook generation
+-- latex             Math -> LaTeX conversion
+-- kernels/          Reference kernel implementations
+-- scoring/          Contract + codebase scoring
+-- query/            O(1) contract search index
+-- lint/             5-gate quality pipeline
+-- build_helper      Build-time helpers
+-- error             Error types
+-- explain           Contract narrative walkthrough
+-- extract           PyTorch -> YAML extraction
+-- codegen           debug_assert!() generation
+-- coq_gen           Coq theorem stub generation
+-- invariant_gen     Type invariant + preservation harness generation
+-- fuzz_gen          libfuzzer fuzz target generation
+-- mirai_gen         MIRAI annotation generation
+-- flux_gen          Flux refinement type generation
+-- tla_gen           TLA+ system spec generation
+-- readme_gen        README generation for contracts
```

---

## 2. Schema Module

### Types

```rust
pub struct Contract {
    pub metadata: Metadata,
    pub equations: BTreeMap<String, Equation>,
    pub proof_obligations: Vec<ProofObligation>,
    pub kernel_structure: Option<KernelStructure>,
    pub simd_dispatch: BTreeMap<String, BTreeMap<String, String>>,
    pub enforcement: BTreeMap<String, EnforcementRule>,
    pub falsification_tests: Vec<FalsificationTest>,
    pub kani_harnesses: Vec<KaniHarness>,
    pub qa_gate: Option<QaGate>,
    pub verification_summary: Option<VerificationSummary>,
    pub type_invariants: Vec<TypeInvariant>,    // Meyer's class invariants
    pub coq_spec: Option<CoqSpec>,              // Coq theorem proving
    pub lean_theorems: Vec<LeanTheorem>,        // Phase 7 Lean 4
}

pub struct Metadata {
    pub version: String,
    pub created: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub references: Vec<String>,
    pub depends_on: Vec<String>,
}

pub struct Equation {
    pub formula: String,
    pub domain: Option<String>,
    pub codomain: Option<String>,
    pub invariants: Vec<String>,
}

pub struct ProofObligation {
    pub obligation_type: ObligationType,
    pub property: String,
    pub formal: Option<String>,
    pub tolerance: Option<f64>,
    pub applies_to: Option<AppliesTo>,
    pub lean: Option<LeanProof>,
}

pub enum ObligationType {
    // Property types (19)
    Invariant, Equivalence, Bound, Monotonicity,
    Idempotency, Linearity, Symmetry, Associativity,
    Conservation, Ordering, Completeness, Soundness,
    Involution, Determinism, Roundtrip, StateMachine,
    Classification, Independence, Termination,
    // Eiffel DbC types (7)
    Precondition, Postcondition, Frame,
    LoopInvariant, LoopVariant, OldState, Subcontract,
}

pub enum KaniStrategy {
    Exhaustive, StubFloat, Compositional, BoundedInt,
}
```

### API

```rust
pub fn parse_contract(path: &Path) -> Result<Contract, ContractError>
pub fn parse_contract_str(yaml: &str) -> Result<Contract, ContractError>
pub fn validate_contract(contract: &Contract) -> Vec<Violation>
```

---

## 3. Codegen Modules

### scaffold

```rust
pub fn generate_trait(contract: &Contract) -> String
pub fn generate_contract_tests(contract: &Contract) -> String
```

### kani_gen

```rust
pub fn generate_kani_harnesses(contract: &Contract) -> String
```

Respects `strategy` field: exhaustive, stub_float, compositional,
bounded_int. Generates `#[cfg(kani)]` gated modules.

### probar_gen

```rust
pub fn generate_probar_tests(contract: &Contract) -> String
pub fn generate_wired_probar_tests(
    contract: &Contract,
    contract_file: &str,
    binding: &BindingRegistry,
) -> String
```

Wired variant calls real functions from binding registry.

### lean_gen

```rust
pub fn generate_lean_files(contract: &Contract) -> Vec<LeanFile>
pub fn lean_status(contract: &Contract) -> LeanStatusReport
pub fn format_status_report(report: &LeanStatusReport) -> String
```

### coq_gen

```rust
pub fn generate_coq_spec(contract: &Contract, stem: &str) -> String
```

Generates `.v` Coq theorem stubs from contract proof obligations.

### invariant_gen

```rust
pub fn generate_invariant_trait(contract: &Contract) -> String
pub fn generate_invariant_harnesses(contract: &Contract) -> String
pub fn generate_invariants(contract: &Contract) -> String
```

Generates `Invariant` trait impls + Kani preservation harnesses from `type_invariants`.

### fuzz_gen

```rust
pub fn generate_fuzz_target(contract: &Contract, stem: &str) -> String
pub fn generate_fuzz_cargo_toml(crate_name: &str) -> String
```

Generates libfuzzer fuzz targets gated on contract preconditions.

### mirai_gen

```rust
pub fn generate_mirai_annotations(contract: &Contract, stem: &str) -> String
```

Generates MIRAI `precondition!` / `postcondition!` / `verify!` annotations.

### flux_gen

```rust
pub fn generate_flux_annotations(contract: &Contract, stem: &str) -> String
```

Generates Flux `#[flux::refined_by]` and `#[flux::sig]` annotations.

### tla_gen

```rust
pub fn generate_tla_module(
    module_name: &str,
    contracts: &[(String, &Contract)],
    graph: &DependencyGraph,
) -> String
```

Generates TLA+ `MODULE` from contract dependency DAG.

### explain

```rust
pub fn explain_contract(contract: &Contract, stem: &str, binding: Option<&BindingRegistry>) -> String
pub fn explain_contract_markdown(contract: &Contract, stem: &str, binding: Option<&BindingRegistry>) -> String
```

Narrative walkthrough of a contract's equations, obligations, and verification chain.

### extract

```rust
pub fn extract_from_pytorch(target: &str) -> Result<ExtractedContract>
```

Parses PyTorch source (`file.py::function_name`) into a contract structure.

---

## 4. Analysis Modules

### audit

```rust
pub fn audit_contract(contract: &Contract) -> AuditReport
pub fn audit_binding(
    contracts: &[(&str, &Contract)],
    binding: &BindingRegistry,
) -> BindingAuditReport
```

### coverage

```rust
pub fn coverage_report(
    contracts: &[(String, &Contract)],
    binding: Option<&BindingRegistry>,
) -> CoverageReport
pub fn overall_percentage(report: &CoverageReport) -> f64
```

### diff

```rust
pub fn diff_contracts(old: &Contract, new: &Contract) -> ContractDiff
// ContractDiff has field: pub suggested_bump: SemverBump
pub fn is_identical(diff: &ContractDiff) -> bool
```

### graph

```rust
pub fn dependency_graph(contracts: &[(String, &Contract)]) -> DependencyGraph
pub fn graph_nodes(graph: &DependencyGraph) -> Vec<GraphNode>

pub struct DependencyGraph {
    pub edges: BTreeMap<String, Vec<String>>,
    pub nodes: BTreeSet<String>,
    pub topo_order: Vec<String>,
    pub cycles: Vec<Vec<String>>,
}
```

### proof_status

```rust
pub fn proof_status_report(
    contracts: &[(String, &Contract)],
    binding: Option<&BindingRegistry>,
    include_classes: bool,
) -> ProofStatusReport

pub enum ProofLevel { L1, L2, L3, L4, L5 }
```

---

## 5. Binding Module

```rust
pub struct BindingRegistry {
    pub version: String,
    pub target_crate: String,
    pub bindings: Vec<KernelBinding>,
}

pub struct KernelBinding {
    pub contract: String,
    pub equation: String,
    pub module_path: Option<String>,
    pub function: Option<String>,
    pub signature: Option<String>,
    pub status: ImplStatus,
    pub notes: Option<String>,
}

pub enum ImplStatus { Implemented, Partial, NotImplemented }
```

---

## 6. Scoring Module [IMPLEMENTED]

```rust
pub struct ContractScore {
    pub stem: String,
    pub spec_depth: f64,           // 0.0-1.0
    pub falsification_coverage: f64,
    pub kani_coverage: f64,
    pub lean_coverage: f64,
    pub binding_coverage: f64,
    pub composite: f64,            // weighted average
    pub grade: Grade,              // A-F
}

pub struct CodebaseScore {
    pub path: PathBuf,
    pub contract_coverage: f64,
    pub binding_completeness: f64,
    pub mean_contract_score: f64,
    pub proof_depth_dist: f64,
    pub drift: f64,
    pub composite: f64,
    pub grade: Grade,
    pub top_gaps: Vec<ScoringGap>,
}

pub enum Grade { A, B, C, D, F }

pub fn score_contract(contract: &Contract, binding: Option<&BindingRegistry>) -> ContractScore
pub fn score_codebase(
    path: &Path,
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
) -> CodebaseScore
```

### Scoring Formulas

**Specification Depth:**
```
sd = (has_equations * 0.3 + has_domains * 0.15 + has_invariants * 0.15
    + has_kernel_structure * 0.15 + has_tolerances * 0.1
    + has_references * 0.1 + has_depends_on * 0.05)
```

**Falsification Coverage:**
```
fc = obligations_with_tests / total_obligations
```

**Kani Coverage:**
```
kc = sum(strategy_weight[h.strategy] for h in harnesses) / total_obligations
where strategy_weight = {exhaustive: 1.0, stub_float: 0.8,
                         compositional: 0.7, bounded_int: 0.9}
```

**Lean Coverage:**
```
lc = proved_count / total_lean_applicable
```

**Binding Coverage:**
```
bc = implemented / (implemented + partial + not_implemented)
where partial counts as 0.5
```

**Composite:**
```
composite = sd*0.20 + fc*0.25 + kc*0.25 + lc*0.10 + bc*0.20
```

---

## 7. Query Module [IMPLEMENTED]

```rust
pub struct ContractIndex {
    pub entries: Vec<ContractEntry>,
    pub name_index: HashMap<String, Vec<usize>>,
    pub equation_index: HashMap<String, Vec<usize>>,
    pub obligation_index: HashMap<ObligationType, Vec<usize>>,
    pub score_cache: HashMap<String, ContractScore>,
    pub dep_graph: DependencyGraph,
}

pub struct ContractEntry {
    pub stem: String,
    pub path: PathBuf,
    pub contract: Contract,
    pub score: ContractScore,
    pub proof_level: ProofLevel,
}

pub struct QueryResult {
    pub entry: ContractEntry,
    pub relevance: f64,            // 0.0-1.0 BM25 score
    pub match_context: String,     // matched text snippet
}

pub fn build_index(contracts_dir: &Path) -> Result<ContractIndex, ContractError>
pub fn search(
    index: &ContractIndex,
    query: &QueryParams,
) -> Vec<QueryResult>

pub struct QueryParams {
    pub terms: Option<String>,
    pub mode: SearchMode,          // Semantic, Regex, Literal
    pub obligation_type: Option<ObligationType>,
    pub min_score: Option<f64>,
    pub min_level: Option<ProofLevel>,
    pub depends_on: Option<String>,
    pub unproven: bool,
    pub binding_gaps: bool,
    pub include_project: Option<PathBuf>,  // Explicit project path
    pub all_projects: bool,                // Force cross-project scan
    pub limit: usize,
}

/// Cross-project contract usage index (auto-discovered).
pub struct CrossProjectIndex {
    pub projects: Vec<ProjectEntry>,
    pub call_sites: HashMap<String, Vec<CallSite>>,
    pub binding_refs: HashMap<String, Vec<BindingRef>>,
    pub kaizen_refs: HashMap<String, Vec<KaizenRef>>,
}

pub struct ProjectEntry {
    pub name: String,             // "aprender", "trueno", etc.
    pub path: PathBuf,            // ../aprender
    pub has_cargo_dep: bool,      // depends on provable-contracts crate
    pub binding_path: Option<PathBuf>,  // contracts/<name>/binding.yaml
}

pub struct CallSite {
    pub project: String,
    pub file: PathBuf,
    pub line: usize,
    pub annotation: String,       // #[contract("softmax-kernel-v1", eq="softmax")]
}

pub struct BindingRef {
    pub project: String,
    pub equation: String,
    pub function: String,
    pub status: ImplStatus,
}

pub struct KaizenRef {
    pub project: String,
    pub ticket: String,           // "KAIZEN-050"
    pub file: PathBuf,
    pub line: usize,
}

pub fn discover_projects(root: &Path) -> Vec<ProjectEntry>
pub fn build_cross_index(
    projects: &[ProjectEntry],
    contracts: &[(String, &Contract)],
) -> Result<CrossProjectIndex, ContractError>
```

### Index Build Strategy

1. Walk `contracts_dir`, parse all YAML files
2. Score each contract
3. Build inverted indices on names, equations, descriptions
4. Build dependency graph from `depends_on` fields
5. **Auto-discover sibling projects** via `../` parent directory
6. **Scan consumer projects** for `#[contract]`, binding.yaml, KAIZEN refs
7. Cache to `.pv/contracts.idx` + `.pv/cross-project.idx`
8. Auto-rebuild when any indexed directory mtime > cache mtime

### Search Pipeline

```
1. Parse query terms
2. Mode dispatch:
   - Semantic: BM25 over description + equation + obligation text
   - Regex: pattern match over all string fields
   - Literal: exact substring match
3. Apply filters (obligation type, score, level, deps)
4. Rank by relevance score
5. Truncate to limit
6. Enrich with requested flags (score, proof-status, binding)
7. Cross-project enrichment (call-sites, violations, coverage-map)
```

---

## 8. Error Types

```rust
pub enum ContractError {
    Io(std::io::Error),
    Yaml(serde_yaml::Error),
    Schema(String),
    MissingField { section: String, field: String },
    InvalidReference { from: String, to: String },
    DuplicateId { id: String, section: String },
}

pub struct Violation {
    pub severity: Severity,
    pub rule: String,
    pub message: String,
    pub location: Option<String>,
}

pub enum Severity { Error, Warning, Info }
```
