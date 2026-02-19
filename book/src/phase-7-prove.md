# Phase 7: Prove -- Unbounded Verification via Lean 4

Phase 7 completes the verification hierarchy by lifting selected proof
obligations into Lean 4's dependent type theory, producing proofs valid
**for all inputs unconditionally** -- achieving what Kani bounded model
checking cannot.

```
Phase 5: Falsify   →  Probabilistic confidence  (proptest/probar)
Phase 6: Verify    →  Bounded certainty         (Kani BMC)
Phase 7: Prove     →  Unbounded certainty       (Lean 4)
```

## The Bounded Verification Gap

Kani proves `softmax_non_negative` for all `f32` vectors up to length N=8. But:

- What about N=65536 (a real transformer sequence length)?
- What about the algebraic identity `Σᵢ softmax(xᵢ) = 1` independent of precision?
- What about compositional properties across the verification DAG?

These require **unbounded proofs over mathematical reals**, which is precisely
what Lean 4 + Mathlib provide.

## Verification Hierarchy

Each proof obligation gains a verification level representing the highest level
at which it has been discharged:

| Level | Method | Guarantees | Tool |
|-------|--------|-----------|------|
| L0 | Untested | None | -- |
| L1 | Unit tests | Spot checks | `cargo test` |
| L2 | Property tests | Probabilistic falsification | probar / proptest |
| L3 | Bounded model checking | Exhaustive within bounds | Kani |
| L4 | Theorem proving | Unbounded, unconditional | Lean 4 |

Obligations can be discharged at multiple levels simultaneously. Higher levels
subsume lower ones logically but serve different practical purposes (L2 catches
regressions fast; L4 provides mathematical certainty).

## Contract Schema: The `lean` Block

Each proof obligation can include a `lean:` block specifying its Lean 4 metadata:

```yaml
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|Σ σ(x)_i - 1.0| < ε"
    tolerance: 1.0e-6
    lean:
      theorem: Softmax.partition_of_unity
      module: ProvableContracts.Softmax
      status: sorry        # proved | sorry | wip | not-applicable
      depends_on:
        - Real.exp_pos
        - Finset.sum_div_distrib
      mathlib_imports:
        - Mathlib.Analysis.SpecialFunctions.ExpDeriv
        - Mathlib.Algebra.BigOperators.Group.Finset
      notes: "Proof over reals; f32 gap addressed by error-bound lemma"
```

The top-level `verification_summary` tracks aggregate L4 coverage:

```yaml
verification_summary:
  total_obligations: 6
  l2_property_tested: 6
  l3_kani_proved: 3
  l4_lean_proved: 0
  l4_sorry_count: 5
  l4_not_applicable: 1
```

## Lean 4 Project Structure

The `lean/` directory contains the theorem-proving layer:

```
lean/
├── lakefile.lean                      # Lake build with Mathlib dependency
├── lean-toolchain                     # Lean 4 version pin
├── ProvableContracts/
│   ├── Basic.lean                     # RVec, sum, max definitions
│   ├── Defs/
│   │   └── Softmax.lean              # softmax, log_softmax, stable_softmax
│   ├── Theorems/
│   │   └── Softmax/
│   │       ├── PartitionOfUnity.lean  # Σ softmax(x)_i = 1 (sorry)
│   │       ├── NonNegativity.lean     # softmax(x)_i > 0 (sorry)
│   │       └── Monotonicity.lean      # x_i > x_j → σ(x)_i > σ(x)_j (sorry)
│   ├── Tactics/                       # Custom tactics (future)
│   └── Meta/                          # Metaprograms (future)
└── test/
```

## CLI Commands

### `pv lean <contract.yaml> [--output-dir <dir>]`

Generate Lean 4 definitions and theorem stubs from a contract:

```bash
$ pv lean contracts/softmax-kernel-v1.yaml --output-dir lean/

  lean/ProvableContracts/Defs/Softmax.lean
  lean/ProvableContracts/Theorems/Softmax/partition_of_unity.lean
  lean/ProvableContracts/Theorems/Softmax/softmax_pos.lean
  lean/ProvableContracts/Theorems/Softmax/softmax_bounded.lean
  lean/ProvableContracts/Theorems/Softmax/monotone.lean
  lean/ProvableContracts/Theorems/Softmax/shift_invariance.lean

Generated 6 Lean files.
```

Without `--output-dir`, prints to stdout for inspection.

### `pv lean-status [<path>]`

Report Lean proof status across contracts:

```bash
$ pv lean-status contracts/

Contract                       Oblgs Proved Sorry WIP N/A
────────────────────────────────────────────────────────────
Softmax kernel — numerically       5      0     5   0   0
────────────────────────────────────────────────────────────
Total                              5      0     5   0   0
L4 Coverage: 0% (0/5)   Sorry Debt: 5
```

## Bridging the Real-Float Gap

Lean proofs operate over `ℝ` (mathematical reals). Rust code operates over
`f32`/`f64`. The strategy is **layered proofs**:

1. **Ideal layer** (Lean): Prove the property over `ℝ` unconditionally
2. **Error layer** (Lean): Prove an error bound for IEEE 754 rounding
3. **Concrete layer** (Kani): Verify the Rust implementation matches the
   error-bounded specification for concrete bit-widths

```lean
-- Ideal layer
theorem softmax_partition_of_unity (x : RVec n) :
    Finset.univ.sum (softmax x) = 1 := by
  sorry

-- Error layer (future)
theorem softmax_partition_f32_error (x : Vector Float32 n)
    (hx : ∀ i, |x i| ≤ 88) :
    |Finset.univ.sum (softmax_f32 x) - 1| ≤ n * Float32.epsilon := by
  sorry
```

## Obligation Triage: What Gets a Lean Proof

| Category | Priority | Rationale |
|----------|:--------:|-----------|
| Universal algebraic identities | **High** | `Σ softmax = 1`, RMSNorm idempotence |
| Compositional correctness | **High** | Verification DAG composition |
| Monotonicity / ordering | **High** | Natural in Mathlib's `Order` |
| Numerical error bounds | Medium | Requires careful epsilon arithmetic |
| Equivalence proofs | Medium | Flash attention = standard attention |
| Performance / hardware | Skip | Empirical by nature |
| Shape / dimension checks | Skip | Well-served by Rust types + Kani |

## Examples

```bash
# Generate Lean stubs for a contract
cargo run --example lean_codegen -- contracts/softmax-kernel-v1.yaml

# Report L4 coverage across all contracts
cargo run --example lean_status -- contracts/
```
