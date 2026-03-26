# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-26

### Added

- 192 YAML contracts (was 107), 16,977 bindings across 13 repos
- 29 CLI commands including `fuzz`, `mirai`, `flux`, `tla`, `infer`, `unlock`
- 7 lint gates: validate, audit, score, verify, enforce, enforcement-level, reverse-coverage
- 26 lint rules with `--explain`, grouped colored output, HTML reports
- `#[must_contract]` proc macro for unannotated pub fn detection
- `EnforcementLevel` (basic/standard/strict/proven) and `locked_level` schema fields
- PVScore 10-dimension geometric mean (`pv score --pvscore`)
- Probe-level score decomposition, per-obligation verification table (`--table`)
- Issue lifecycle tracking (new/pre-existing findings via fingerprint)
- Remediation effort estimation, structured fix suggestions
- Source snippets, counterexample evidence, per-contract timing
- Reverse coverage (`pv coverage --reverse`) and `pv infer`
- Section 21: contract gap analysis (9 ML domains + shape algebra)
- Section 22: diagnostic output (13 gaps falsified against 9 tools)
- Domain contracts: bashrs (parser/classifier/encoder), depyler (type/semantic/memory)
- Gap analysis contracts: speculative-decoding, fp8, dpo-loss, bpe, paged-attention

## [0.1.0] - 2025-01-01

### Added

- Initial release of provable-contracts
- YAML contract parsing and validation
- Scaffold generation for Kani harnesses
- CLI tool (`pv`) for contract management
- Proc macro `#[contract]` attribute for compile-time enforcement
- Support for numerical stability, convergence, and monotonicity contracts
- Property-based testing with proptest
- cargo-deny integration for dependency auditing
- Clippy pedantic linting configuration

[Unreleased]: https://github.com/paiml/provable-contracts/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/paiml/provable-contracts/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/paiml/provable-contracts/releases/tag/v0.1.0
