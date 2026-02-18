# provable-contracts

Papers to Math to Contracts in Code. Rust library and CLI for converting
research papers into provable kernel implementations via YAML contracts
with Kani bounded model checking.

## Code Search Policy

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

```bash
# Find functions by intent
pmat query "contract validation" --limit 10

# Find with fault patterns (--faults)
pmat query "unwrap" --faults --exclude-tests

# Regex search
pmat query --regex "fn\s+verify_\w+" --limit 10

# Literal search
pmat query --literal "kani::proof" --limit 10
```

See `pmat query --help` for full options.

## Project Structure

- `crates/provable-contracts/` — Library crate (schema, scaffold, kani, probar)
- `crates/provable-contracts-cli/` — CLI binary (`pv` command)
- `contracts/` — YAML kernel contracts
- `docs/specifications/provable-contracts.md` — Full specification

## Work Tracking

All work tracked via `pmat work`:

```bash
pmat work list          # See all tickets
pmat work start PMAT-N  # Start work
pmat work complete PMAT-N  # Mark done
```

## Quality Gates

```bash
pmat comply check       # Check compliance
pmat quality-gate       # Run quality gates
cargo kani              # Run Kani proof harnesses
```
