<p align="center">
  <img src=".github/pv-hero.svg" width="800" alt="provable-contracts">
</p>

<h1 align="center">provable-contracts</h1>

<p align="center">
  <strong>Papers to Math to Contracts in Code</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/provable-contracts">
    <img src="https://img.shields.io/crates/v/provable-contracts.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/provable-contracts">
    <img src="https://docs.rs/provable-contracts/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/provable-contracts/actions">
    <img src="https://github.com/paiml/provable-contracts/actions/workflows/ci.yml/badge.svg"
         alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
</p>

A Rust library and CLI for converting peer-reviewed research papers into
mathematically provable kernel implementations via YAML contract
intermediaries with Kani bounded model checking verification.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [CLI Reference](#cli-reference)
- [Contract Registry](#contract-registry)
- [The Six-Phase Pipeline](#the-six-phase-pipeline)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **YAML Contracts** -- Declare kernel semantics (equations, invariants,
  proof obligations) in a structured, version-controlled YAML schema.
- **Scaffold Generation** -- Automatically produce Rust trait stubs and
  failing test skeletons from any contract.
- **Kani Harness Codegen** -- Generate `#[kani::proof]` bounded model
  checking harnesses from proof obligation taxonomies.
- **Property Test Generation** -- Emit `proptest` / probar property-based
  tests from both obligations and falsification predicates.
- **Binding Registry** -- Map contract equations to real crate functions
  (`aprender`, etc.) for wired integration tests.
- **Traceability Audit** -- Verify the full chain from paper reference
  through equation, obligation, falsification test, and Kani harness.

## Installation

### Library

Add to your `Cargo.toml`:

```toml
[dependencies]
provable-contracts = "0.1"
```

### CLI

```bash
cargo install provable-contracts-cli
```

This installs the `pv` binary.

## Usage

### Library API

```rust
use provable_contracts::schema::{parse_contract, validate_contract};
use provable_contracts::scaffold::generate_trait;
use provable_contracts::kani_gen::generate_kani_harnesses;

let contract = parse_contract("contracts/softmax-kernel-v1.yaml")?;
let violations = validate_contract(&contract);
let trait_code = generate_trait(&contract);
let harnesses = generate_kani_harnesses(&contract);
```

### CLI Examples

```bash
# Validate a contract
pv validate contracts/softmax-kernel-v1.yaml

# Generate Rust trait + test scaffolding
pv scaffold contracts/softmax-kernel-v1.yaml

# Generate Kani proof harnesses
pv kani contracts/softmax-kernel-v1.yaml

# Generate probar property tests
pv probar contracts/softmax-kernel-v1.yaml

# Generate wired tests using binding registry
pv probar contracts/softmax-kernel-v1.yaml \
    --binding contracts/aprender/binding.yaml

# Show contract status
pv status contracts/softmax-kernel-v1.yaml

# Run traceability audit
pv audit contracts/softmax-kernel-v1.yaml

# Audit with binding coverage
pv audit contracts/softmax-kernel-v1.yaml \
    --binding contracts/aprender/binding.yaml

# Compare two contract versions
pv diff contracts/softmax-kernel-v1.yaml contracts/softmax-kernel-v2.yaml

# Cross-contract obligation coverage
pv coverage contracts/ --binding contracts/aprender/binding.yaml

# End-to-end codegen (scaffold + kani + probar)
pv generate contracts/softmax-kernel-v1.yaml -o generated/

# Dependency graph with topological order
pv graph contracts/
```

## CLI Reference

| Command    | Description                                          |
|------------|------------------------------------------------------|
| `validate` | Parse and validate a YAML kernel contract            |
| `scaffold` | Generate Rust trait definition + failing tests        |
| `kani`     | Generate `#[kani::proof]` bounded model harnesses    |
| `probar`   | Generate property-based tests from obligations       |
| `status`   | Display contract summary (equations, obligations)    |
| `audit`    | Run traceability audit with optional binding check   |
| `diff`     | Compare two contract versions, suggest semver bump   |
| `coverage` | Cross-contract obligation coverage report            |
| `generate` | End-to-end codegen (scaffold + kani + probar)        |
| `graph`    | Dependency DAG with cycle detection + topo order     |

## Contract Registry

48 kernel contracts ship in `contracts/`, organized by tier:

**Tier 1 -- Core Kernels** (7 contracts):
softmax, rmsnorm, rope, activation (GeLU/ReLU/SiLU), attention,
matmul, flash-attention.

**Tier 2 -- Compound Kernels** (6 contracts):
swiglu, gqa, layernorm, silu, cross-entropy, adamw.

**Tier 3 -- Extended Algorithms** (7 contracts):
ssm (Mamba), conv1d, batchnorm, kmeans, pagerank, lbfgs, cma-es.

**Model Architecture** (21 contracts):
model-config-algebra, qk-norm, tensor-shape-flow, roofline-model,
gated-delta-net, format-parity, shannon-entropy, f16-conversion,
kernel-launch-budget, tensor-inventory, performance-grading,
q4k-q6k-superblock, sampling-algorithms, validated-tensor,
hybrid-layer-dispatch, qwen35-shapes, kv-cache-sizing,
kv-cache-equivalence, backend-dispatch, lora-algebra,
quantization-ordering.

**Qwen 3.5 Verification** (7 contracts):
sliding-window-attention, rope-extrapolation (NTK/YaRN),
embedding-algebra, inference-pipeline, qwen35-hybrid-forward,
attention-scaling, qwen35-e2e-verification.

**Totals**: 166 equations, 262 proof obligations, 276 falsification
tests, 81 Kani harnesses, 174 binding entries. Every obligation has at
least one falsification test.

### Qwen 3.5 Verification DAG

The Qwen 3.5 end-to-end verification contract composes 8 sub-contracts
into a complete model proof. The dependency graph:

```
softmax ← attention ← sliding-window-attention
       ← cross-entropy        ↑
       ← sampling       qk-norm ← attention-scaling
       ← gqa                   ↑
                        rmsnorm ← qwen35-hybrid-forward ← e2e
silu ← swiglu ─────────────────↑
matmul ← gqa             conv1d ← gated-delta-net ──────↑
rope ← rope-extrapolation       hybrid-dispatch ────────↑
                          embedding-algebra ← inference-pipeline
model-config-algebra ← qwen35-shapes ──────────────────↑
                     ← kv-cache-sizing ─────────────────↑
```

## The Six-Phase Pipeline

The provable-contracts methodology follows six phases:

1. **Extract** -- Parse peer-reviewed papers into canonical math
   (equations, domains, invariants).
2. **Specify** -- Encode the math as a YAML kernel contract with proof
   obligations, falsification tests, and Kani harness definitions.
3. **Scaffold** -- Auto-generate Rust trait stubs and failing test
   skeletons from the contract.
4. **Implement** -- Write the scalar reference implementation, then the
   SIMD-accelerated version.
5. **Falsify** -- Run Popperian falsification via property-based testing
   (probar + certeza).
6. **Verify** -- Prove correctness bounds via Kani bounded model
   checking.

## Documentation

- **Specification**: [`docs/specifications/provable-contracts.md`](
  docs/specifications/provable-contracts.md)
- **mdBook**: Run `mdbook serve` from the repository root, or build
  with `mdbook build`.

## Contributing

1. Fork the repository
2. Create your changes on the `master` branch
3. Run quality gates: `make lint && make test`
4. Run coverage: `make coverage`
5. Submit a pull request

## License

MIT
