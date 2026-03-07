# Introduction

**Papers to Math to Contracts in Code**

A Rust library and CLI for converting peer-reviewed research papers into
mathematically provable kernel implementations via YAML contract
intermediaries with Kani bounded model checking verification.

Available as:
- **Library** (`provable-contracts`): Contract parsing, validation, scaffold
  generation, Kani harness codegen, probar test generation
- **CLI** (`provable-contracts-cli`): 18 commands including `pv validate`,
  `pv lint`, `pv score`, `pv query`, `pv lean-status`, `pv proof-status`

Primary consumer: [aprender](https://github.com/paiml/aprender) ML library
and the broader PAIML Sovereign AI stack.

**Tracking:** All work tracked via `pmat work` (PMAT-001 through PMAT-017).

---

## Project Structure (Library + CLI)

### Crate Layout

```
provable-contracts/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── provable-contracts/         # library crate
│   │   └── src/
│   │       ├── lib.rs              # public API
│   │       ├── schema/             # YAML contract parser + validator
│   │       ├── scaffold/           # Rust trait codegen
│   │       ├── kani_gen/           # Kani harness codegen
│   │       ├── probar/             # probar test codegen
│   │       ├── audit.rs            # traceability audit
│   │       ├── scoring.rs          # 5-dimension quality scoring
│   │       ├── lint.rs             # quality gate (validate+audit+score)
│   │       ├── query.rs            # BM25 semantic search
│   │       └── lean.rs             # Lean 4 codegen
│   ├── provable-contracts-cli/     # binary crate (`pv`)
│   │   └── src/
│   │       ├── main.rs
│   │       └── commands/           # 18 CLI commands
│   └── provable-contracts-macros/  # proc-macro crate
├── contracts/                      # 167 YAML kernel contracts
│   ├── softmax-kernel-v1.yaml
│   ├── aprender/                   # aprender-specific contracts
│   ├── entrenar/                   # training contracts
│   └── forjar/                     # forjar contracts
├── lean/                           # Lean 4 proofs (14/14 proved)
├── docs/specifications/            # canonical spec + sub-specs
├── book/                           # mdBook documentation
└── .pmat/project.toml              # pmat compliance config
```

### Library API (provable-contracts crate)

```rust
// Schema — Parse and validate YAML contracts
pub fn parse_contract(path: &Path) -> Result<Contract, SchemaError>;
pub fn validate_contract(contract: &Contract) -> Vec<Violation>;

// Codegen — Generate Rust code from contracts
pub fn generate_trait(contract: &Contract) -> TokenStream;
pub fn generate_contract_tests(contract: &Contract) -> TokenStream;
pub fn generate_kani_harnesses(contract: &Contract) -> TokenStream;
pub fn generate_probar_tests(contract: &Contract) -> TokenStream;

// Audit — Trace paper→code chain
pub fn audit_contract(contract: &Contract) -> AuditReport;

// Scoring — Five-dimension quality metric
pub fn score_contract(contract: &Contract, binding: Option<&BindingRegistry>, stem: &str) -> ContractScore;

// Lint — Quality gate (validate + audit + score)
pub fn run_lint(config: &LintConfig) -> LintReport;
```

### CLI Commands (pv binary)

```
pv validate <contract.yaml>     Validate YAML against contract schema
pv lint <contracts-dir/>        Quality gate: validate + audit + score
pv audit <contract.yaml>        Traceability audit (paper → proof chain)
pv score <path>                 Score contracts or codebase (A-F grades)
pv query "softmax"              Semantic search across contracts
pv scaffold <contract.yaml>     Generate Rust trait + test stubs
pv kani <contract.yaml>         Generate #[kani::proof] harnesses
pv probar <contract.yaml>       Generate probar property tests
pv lean <contract.yaml>         Generate Lean 4 theorem stubs
pv lean-status <contracts-dir/> Report Lean 4 proof status (14/14 proved)
pv proof-status <contracts-dir/> Hierarchical proof level (L1-L5) report
pv status <contract.yaml>       Show contract summary
pv diff <old.yaml> <new.yaml>   Diff versions, suggest semver bump
pv coverage <contracts-dir/>    Cross-contract obligation coverage
pv generate <contract.yaml>     Write all codegen artifacts to disk
pv graph <contracts-dir/>       Dependency DAG (text/dot/json/mermaid)
pv equations <contract.yaml>    Render equations (text/latex/ptx/asm)
pv book <contracts-dir/>        Generate mdBook pages
```

---

## Work Tracking (pmat work)

All implementation work is tracked via `pmat work` tickets:

### Critical Priority

| ID | Title | Tags |
|----|-------|------|
| PMAT-001 | Initialize Rust crate with lib + CLI binary targets | infrastructure, rust |
| PMAT-002 | Implement YAML contract schema parser and validator | library, parser |

### High Priority

| ID | Title | Tags |
|----|-------|------|
| PMAT-003 | Implement Rust trait scaffold generator | library, codegen |
| PMAT-004 | Implement Kani harness generator | library, kani, verification |
| PMAT-005 | Build CLI: validate/scaffold/verify commands | cli, ux |
| PMAT-006 | Write softmax-kernel-v1.yaml contract (Tier 1) | contract, kernel, tier1 |
| PMAT-007 | Write rmsnorm-kernel-v1.yaml contract (Tier 1) | contract, kernel, tier1 |
| PMAT-013 | Achieve full pmat comply compliance | quality, compliance |
| PMAT-015 | Add Kani verification backend integration tests | testing, kani |

### Medium Priority

| ID | Title | Tags |
|----|-------|------|
| PMAT-008 | Write rope-kernel-v1.yaml contract (Tier 1) | contract, kernel, tier1 |
| PMAT-009 | Write activation-kernel-v1.yaml (SwiGLU, Tier 1) | contract, kernel, tier1 |
| PMAT-010 | Write attention-kernel-v1.yaml (Tier 2) | contract, kernel, tier2 |
| PMAT-011 | Write matmul-kernel-v1.yaml (Tier 2) | contract, kernel, tier2 |
| PMAT-014 | Migrate existing aprender contracts | migration, contracts |
| PMAT-016 | Implement probar test generator | library, testing, probar |

### Low Priority

| ID | Title | Tags |
|----|-------|------|
| PMAT-012 | Write flash-attention-v1.yaml (Tier 2) | contract, kernel, tier2 |
| PMAT-017 | Publish to crates.io | release, crates-io |

### Dependency Graph

```
PMAT-001 (crate init)
  ├── PMAT-002 (schema parser)
  │     ├── PMAT-003 (scaffold generator)
  │     ├── PMAT-004 (Kani generator)
  │     └── PMAT-016 (probar generator)
  ├── PMAT-005 (CLI) ← depends on PMAT-002, 003, 004
  └── PMAT-013 (pmat compliance)

PMAT-006 (softmax)  ─┐
PMAT-007 (rmsnorm)   │
PMAT-008 (rope)      ├── can be written in parallel (YAML only)
PMAT-009 (activation)│
PMAT-011 (matmul)   ─┘
                      │
PMAT-010 (attention) ← depends on PMAT-006 + PMAT-011
PMAT-012 (flash-attn)← depends on PMAT-010

PMAT-014 (migrate aprender contracts) — independent
PMAT-015 (Kani integration tests) ← depends on PMAT-004
PMAT-017 (crates.io) ← depends on everything else
```

### Commands

```bash
# List all tickets
pmat work list

# Start working on a ticket
pmat work start PMAT-001

# Mark complete
pmat work complete PMAT-001

# Check status
pmat work status
```
