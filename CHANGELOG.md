# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Softmax theorems — positivity, partition of unity, monotonicity proofs
- PMAT proof status integration
- Falsification receipts for failed proof attempts

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

[Unreleased]: https://github.com/paiml/provable-contracts/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/provable-contracts/releases/tag/v0.1.0
