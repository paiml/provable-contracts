# Integration with PAIML Stack

## Consumer Projects

| Project | Consumes | Role | Integration Level |
|---------|----------|------|-------------------|
| **aprender** | All contracts | ML algorithm layer | Level 3 (compile-time) |
| **entrenar** | Training contracts | Training & optimization | Level 3 (compile-time) |
| **realizar** | Tier 2-3 contracts | GPU inference engine | Level 3 (compile-time) |
| **trueno** | Tier 1 contracts | SIMD kernel implementations | Level 3 (compile-time) |
| **certeza** | QA gates from all contracts | Quality enforcement | Level 1 (binding) |
| **probar** | Proof obligations from all contracts | Property-based testing | Level 2 (wired tests) |
| **Kani** | Proof obligations from all contracts | Bounded model checking | Level 4 (Kani proofs) |
| **pmat** | Contract metadata | Code quality annotations | Level 0 (YAML-only) |

## Compile-Time Enforcement (Level 3)

All four primary consumer crates enforce contract bindings at compile
time via `build.rs`. Each crate's build script reads its binding
registry from `../provable-contracts/contracts/<crate>/binding.yaml`
and emits `CONTRACT_*` environment variables consumed by the
`#[contract]` proc macro.

| Crate | Policy | Bindings | Coverage |
|-------|--------|----------|----------|
| **aprender** | AllImplemented | 301 | 100% |
| **entrenar** | WarnOnGaps | 96 | 84% |
| **realizar** | WarnOnGaps | 23 | 100% |
| **trueno** | AllImplemented | 22 | 100% |

**AllImplemented** — Build fails if any binding has status
`not_implemented` or `partial`. Used by crates with complete coverage.

**WarnOnGaps** — Emits `cargo:warning` for gaps but does not fail
the build. Used by crates with known gaps tracked via GitHub issues.

### Environment Variable Convention

```
CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>
```

Example: `CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented`

### Build Dependencies

Each downstream crate adds to `Cargo.toml`:

```toml
[build-dependencies]
serde = { version = "1", features = ["derive"] }
serde_yaml_ng = "0.10"
```

## Batuta Integration

Batuta orchestrates the pipeline, delegating to provable-contracts for
Phases 2-6:

```bash
# Phase 1: Extract equations from paper context
batuta oracle "softmax numerical stability" --arxiv --arxiv-live

# Phase 2: Validate contract (delegates to pv validate)
pv validate contracts/softmax-kernel-v1.yaml

# Phase 3: Generate scaffold (delegates to pv scaffold)
pv scaffold contracts/softmax-kernel-v1.yaml --output src/softmax/

# Phase 5: Run falsification suite (Level 3)
batuta falsify --contract contracts/softmax-kernel-v1.yaml

# Phase 6: Run Kani proof harnesses (Level 4)
pv verify contracts/softmax-kernel-v1.yaml
# or directly:
cargo kani --harness verify_softmax_normalization
cargo kani --harness verify_softmax_simd_parity --solver kissat

# Full status: which obligations are proven?
pv status contracts/
```

## Library Integration (Rust API)

Consumer crates (aprender, entrenar, realizar, trueno) add
provable-contracts as a dev-dependency for contract-driven testing:

```toml
[dev-dependencies]
provable-contracts = "0.1"
```

```rust
use provable_contracts::schema::parse_contract;
use provable_contracts::audit::audit_contract;

#[test]
fn test_contract_compliance() {
    let contract = parse_contract(
        Path::new("contracts/softmax-kernel-v1.yaml")
    ).unwrap();
    let report = audit_contract(&contract, Path::new("src/"));
    assert!(report.all_obligations_covered(),
        "Uncovered obligations: {:?}", report.gaps());
}
```

## EDD Recipe Integration

The `quality-edd` recipe from batuta's cookbook maps directly:

```
EDD Cycle              Provable Contracts Phase
─────────              ──────────────────
Equation        →      Phase 1 (Extract)
Failing Test    →      Phase 3 (Scaffold)
Implementation  →      Phase 4 (Implement)
Verification    →      Phase 5 (Falsify — probar, Level 3)
Falsification   →      Phase 5 (Falsify — introduce bugs, verify detection)
Proof           →      Phase 6 (Verify — Kani, Level 4)
```
