# Introduction

**Papers to Math to Contracts in Code**

A Rust library and CLI for converting peer-reviewed research papers into
mathematically provable kernel implementations via YAML contract
intermediaries with Kani bounded model checking verification.

Available as:
- **Library** (`provable-contracts`): Contract parsing, validation, scaffold
  generation, Kani harness codegen, probar test generation
- **CLI** (`provable-contracts-cli`): `pv validate`, `pv scaffold`,
  `pv verify`, `pv status`, `pv audit`

Primary consumer: [aprender](https://github.com/paiml/aprender) ML library
and the broader PAIML Sovereign AI stack.

**Tracking:** All work tracked via `pmat work` (PMAT-001 through PMAT-017).

---

## Project Structure (Library + CLI)

### Crate Layout

```
provable-contracts/
├── Cargo.toml              # workspace root
├── crates/
│   ├── provable-contracts/     # library crate (PMAT-001)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # public API
│   │       ├── schema/         # YAML contract parser (PMAT-002)
│   │       │   ├── mod.rs
│   │       │   ├── parser.rs
│   │       │   └── validator.rs
│   │       ├── scaffold/       # Rust trait codegen (PMAT-003)
│   │       │   ├── mod.rs
│   │       │   ├── trait_gen.rs
│   │       │   └── test_gen.rs
│   │       ├── kani/           # Kani harness codegen (PMAT-004)
│   │       │   ├── mod.rs
│   │       │   ├── exhaustive.rs
│   │       │   ├── stub_float.rs
│   │       │   └── compositional.rs
│   │       └── probar/         # probar test codegen (PMAT-016)
│   │           ├── mod.rs
│   │           └── generators.rs
│   └── provable-contracts-cli/ # binary crate (PMAT-005)
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           └── commands/
│               ├── validate.rs # pv validate
│               ├── scaffold.rs # pv scaffold
│               ├── verify.rs   # pv verify (runs cargo kani)
│               ├── status.rs   # pv status
│               └── audit.rs    # pv audit (trace chain)
├── contracts/                  # YAML contract registry
│   ├── softmax-kernel-v1.yaml      (PMAT-006)
│   ├── rmsnorm-kernel-v1.yaml      (PMAT-007)
│   ├── rope-kernel-v1.yaml         (PMAT-008)
│   ├── activation-kernel-v1.yaml   (PMAT-009)
│   ├── attention-kernel-v1.yaml    (PMAT-010)
│   ├── matmul-kernel-v1.yaml       (PMAT-011)
│   └── flash-attention-v1.yaml     (PMAT-012)
├── docs/
│   ├── specifications/
│   │   └── provable-contracts.md   # this document
│   └── roadmaps/
│       └── roadmap.yaml            # pmat work tickets
└── .pmat/
    └── project.toml                # pmat compliance config
```

### Library API (provable-contracts crate)

```rust
// provable_contracts::schema — Parse and validate YAML contracts
pub fn parse_contract(path: &Path) -> Result<Contract, SchemaError>;
pub fn validate_contract(contract: &Contract) -> Vec<Violation>;

// provable_contracts::scaffold — Generate Rust code from contracts
pub fn generate_trait(contract: &Contract) -> TokenStream;
pub fn generate_contract_tests(contract: &Contract) -> TokenStream;
pub fn generate_kani_harnesses(contract: &Contract) -> TokenStream;
pub fn generate_probar_tests(contract: &Contract) -> TokenStream;

// provable_contracts::audit — Trace paper→code chain
pub fn audit_contract(contract: &Contract, src: &Path) -> AuditReport;
```

### CLI Commands (pv binary)

```
pv validate contracts/softmax-kernel-v1.yaml
    Check YAML against contract schema. Report missing fields,
    invalid cross-references, unreachable falsification tests.

pv scaffold contracts/softmax-kernel-v1.yaml --output src/softmax/
    Generate trait.rs, contract_tests.rs, kani_proofs.rs, probar_tests.rs.
    All tests fail initially (Phase 3 scaffold).

pv verify contracts/softmax-kernel-v1.yaml
    Run cargo kani on all harnesses defined in the contract.
    Report per-harness pass/fail with counterexamples.

pv status contracts/
    Show verification level matrix:
    ┌─────────────┬────────┬────────┬────────┐
    │ Obligation  │ Type   │ probar │ Kani   │
    ├─────────────┼────────┼────────┼────────┤
    │ SM-INV-001  │ L3 ✓   │ L4 ✓   │ PROVEN │
    │ SM-EQV-001  │ L3 ✓   │ L4 ✗   │ TODO   │
    └─────────────┴────────┴────────┴────────┘

pv audit contracts/softmax-kernel-v1.yaml --src ../aprender/src/
    Trace full chain: paper → equation → contract → trait → test → proof.
    Report gaps (equation without test, test without Kani harness, etc).
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
