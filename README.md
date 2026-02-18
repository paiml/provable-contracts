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

## Contract Registry

Seven kernel contracts ship in `contracts/`:

| Contract | Paper | Key Property |
|----------|-------|-------------|
| `softmax-kernel-v1.yaml` | Bridle 1990 | Output sums to 1 |
| `rmsnorm-kernel-v1.yaml` | Zhang & Sennrich 2019 | Scale invariance |
| `rope-kernel-v1.yaml` | Su et al. 2021 | Rotary position encoding |
| `activation-kernel-v1.yaml` | Shazeer 2020 | SwiGLU gating |
| `attention-kernel-v1.yaml` | Vaswani et al. 2017 | Scaled dot-product |
| `matmul-kernel-v1.yaml` | Standard linear algebra | Associativity |
| `flash-attention-v1.yaml` | Dao et al. 2022 | Tiled online softmax |

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
