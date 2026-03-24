# Sub-spec: PVScore — Provable Verification Score

**Parent:** [pv-spec.md](../pv-spec.md) Section 18

---

## Design: A Score That's Hard to Fake

PVScore is a **0-100 composite** that's deliberately hard to achieve.
Grade A (90+) requires excellence across ALL dimensions — you can't
compensate weak formal proofs with good test coverage.

**Key design decision:** Geometric mean, not arithmetic. One zero
dimension tanks the entire score. This prevents gaming.

**Threshold:** Any project below 90/100 (Grade A) auto-fails CI.
This is a HARD requirement — not a suggestion, not a warning.

### Why Geometric Mean?

Arithmetic mean: `(100 + 100 + 0) / 3 = 67` — you can hide a zero.
Geometric mean: `(100 * 100 * 0)^(1/3) = 0` — one zero kills you.

This matches how real verification works: a proof with one
unchecked case is not a proof.

---

## 10 Dimensions (equally weighted via geometric mean)

### D1: Contract Specification Depth (0-100)

How completely is the math captured?

```
D1 = pv_score.spec_depth * 100
```

Components: equations (30%), domains (15%), invariants (15%),
kernel_structure (15%), tolerances (10%), references (10%),
depends_on (5%).

**Hard to fake:** Requires actual mathematical specification
from peer-reviewed papers.

### D2: Falsification Coverage (0-100)

Popperian: every obligation must have a test that COULD fail.

```
D2 = (obligations_with_falsification_test / total_obligations) * 100
```

**Hard to fake:** Tests must be property-based (proptest/probar),
not just unit tests. Must cover edge cases specified by contract.

### D3: Kani Bounded Model Checking (0-100)

Exhaustive verification within bounds.

```
D3 = sum(strategy_weight per obligation) / total_obligations * 100
```

Strategy weights: exhaustive=1.0, bounded_int=0.9, stub_float=0.8.

**Hard to fake:** Kani actually runs the prover. Can't stub it out.

### D4: Lean 4 Theorem Proving (0-100)

Full mathematical proof over reals.

```
D4 = proved_theorems / applicable_obligations * 100
```

Weight: now **equal** to other dimensions (was 10% in pv score).
This is the gold standard — formal proofs are the hardest thing
to produce and the strongest guarantee.

**Hard to fake:** Lean checks the proof. `sorry` = 0 points.

### D5: Binding Compliance (0-100)

Every equation maps to a real function with AllImplemented policy.

```
D5 = implemented_bindings / total_bindings * 100
```

Partial bindings count as 0. AllImplemented enforced by build.rs.

**Hard to fake:** Compiler rejects the build if bindings are wrong.

### D6: Reverse Coverage (0-100)

What fraction of the crate's public API is under contract?

```
D6 = bound_pub_fns / total_pub_fns * 100
```

Computed by `pv coverage --reverse`. This catches the
"whack-a-mole" gap — new functions must have contracts.

**Hard to fake:** Scans actual source code, not just YAML.

### D7: Mutation Testing Survival (0-100)

Can tests detect injected bugs?

```
D7 = mutants_killed / mutants_tested * 100
```

Sourced from `cargo-mutants` or certeza mutation results.

**Hardest to fake.** Mutation testing is the ultimate test
quality metric — high coverage with low mutation kill rate
means the tests don't actually check anything.

### D8: CI Pipeline Depth (0-100)

How many verification stages does the CI pipeline enforce?

| Stage | Points |
|---|---|
| `cargo test` passes | 10 |
| `cargo clippy -- -D warnings` | 10 |
| `pv lint` all gates pass | 15 |
| `pv score --min-score 0.75` | 10 |
| `cargo deny check` (supply chain) | 10 |
| `cargo-mutants` or certeza | 15 |
| AllImplemented build.rs | 10 |
| `#[contract]` macro coverage > 25% | 10 |
| `pv coverage --reverse > 50%` | 10 |

**Hard to fake:** CI logs are auditable. Must actually run.

### D9: Proof Freshness (0-100)

How recently were Kani proofs and Lean theorems verified?

```
D9 = 100 - (days_since_last_kani_run * 2)  // capped at 0
```

If Kani was run in CI within 7 days: 100.
If 30 days: 40. If 90+ days: 0.

**Hard to fake:** Requires CI timestamps. Stale proofs decay.

### D10: Defect Pattern Score (0-100)

From organizational-intelligence-plugin analysis.

```
D10 = 100 - (defect_density * 10)  // defects per 1000 LOC
```

Categories: SecurityVulnerabilities, ConfigurationErrors,
PerformanceRegressions, APIBreakingChanges.

**Hard to fake:** Based on git history pattern analysis.

---

## Composite Score

```
pvscore = (D1 * D2 * D3 * D4 * D5 * D6 * D7 * D8 * D9 * D10) ^ (1/10)
```

Geometric mean of 10 dimensions, each 0-100.

### Grade Thresholds

| Grade | Score | CI Action |
|---|---|---|
| A+ | 95-100 | Pass (exemplary) |
| A | 90-94 | Pass (minimum for merge) |
| B | 80-89 | **FAIL** (close but not sufficient) |
| C | 60-79 | **FAIL** (significant gaps) |
| D | 40-59 | **FAIL** (major deficiencies) |
| F | 0-39 | **FAIL** (critical) |

**Only Grade A (90+) passes CI.** This is non-negotiable.

### Why 90?

At 90/100 geometric mean across 10 dimensions, every single
dimension must be at least ~75 (since `75^10 ≈ 56` which is
below 90 — you need most dimensions above 90). You cannot
have even ONE weak dimension without falling below the threshold.

---

## Comparison with Existing Scores

| System | Scale | Dimensions | Combination | Threshold |
|---|---|---|---|---|
| pmat tdg | 0-10 | 6 | Weighted sum | A < 2.0 |
| pmat score | 0-100 | 8 | Geometric mean | A >= 90 |
| pv score | 0.0-1.0 | 5+5 | Weighted sum | A >= 0.90 |
| **PVScore** | **0-100** | **10** | **Geometric mean** | **A >= 90 (hard fail)** |

PVScore is strictly harder than any individual system because:
1. More dimensions (10 vs 5-8)
2. Geometric mean (one zero kills you)
3. Higher bar for A (90 across ALL 10 dimensions)
4. Includes dimensions the others lack (mutation, freshness, reverse coverage)

---

## CLI

```bash
# Compute PVScore for the current project
pv pvscore .

# Compute PVScore with CI gate
pv pvscore . --min-score 90 --exit-code

# Breakdown by dimension
pv pvscore . --verbose

# JSON output for dashboards
pv pvscore . --format json
```

---

## References

### Quality Scoring Models

1. Letouzey, J.-L. (2012). "The SQALE Method for Evaluating Technical Debt."
   *MTD Workshop, ICSE 2012*.

2. Zahan, N. et al. (2023). "OpenSSF Scorecard: On the Path Toward Ecosystem-wide
   Automated Security Metrics." *IEEE Security & Privacy*. arXiv:2208.03412.

3. Jin, S. et al. (2023). "Software Code Quality Measurement: Implications from
   Metric Distributions." arXiv:2307.12082.

4. Wong, S. et al. (2025). "A Note on Code Quality Score: LLMs for Maintainable
   Large Codebases." arXiv:2508.02732.

5. Perera, J. et al. (2023). "Quantifying Technical Debt: A Systematic Mapping Study
   and a Conceptual Model." arXiv:2303.06535.

6. Molnar, A. & Motogna, S. (2024). "Versioned Analysis of Software Quality
   Indicators and Technical Debt." arXiv:2407.15967.

### Contract and Verification Metrics

7. Meyer, B. (2025). "Software engineering as a domain to formalize."
   arXiv:2502.11434.

8. Lim, S. et al. (2025). "ContractEval: A Benchmark for Evaluating
   Contract-Satisfying Assertions." arXiv:2510.12047.

9. Goel, A. et al. (2026). "An End-to-End Agentic Pipeline for Smart Contract
   Translation and Quality Evaluation." arXiv:2602.13808.

10. Rust Std Lib Verification (2025). arXiv:2510.01072.

### Formal Verification Adoption

11. Tu, H. et al. (2025). "Agentic Program Verification." arXiv:2511.17330.

12. Pothireddypalli, S. et al. (2026). "Agentic AI-based Coverage Closure
    for Formal Verification." arXiv:2603.03147.

### Gradual Typing Coverage

13. Bader, J., Aldrich, J. & Tanter, E. (2018). "Gradual Program Verification."
    arXiv:1710.06422.

14. Cassano, F. et al. (2023). "Type Prediction With Program Decomposition
    and Fill-in-the-Type Training." arXiv:2305.17145.

### Mutation Testing

15. Petrovic, G. et al. (2022). "Practical Mutation Testing at Scale."
    *IEEE TSE 48(10)*.

16. Gopinath, R., Jensen, C. & Groce, A. (2014). "Mutations: How Close are
    they to Real Faults?" *ISSRE 2014*.

### CI/CD Quality Gates

17. Sun, S., Friberg, D. & Staron, M. (2025). "'Good' and 'Bad' Failures in
    Industrial CI/CD." arXiv:2504.11839.

18. Feist, J. et al. (2024). "Integrating Static Code Analysis Toolchains."
    arXiv:2403.05986.

### Build Reproducibility

19. Schmid, L. et al. (2025). "Maven-Lockfile: High Integrity Rebuild of
    Past Java Releases." arXiv:2510.00730.

### Organizational Intelligence

20. PAIML Organizational Intelligence Plugin (2025-2026).
    github.com/paiml/organizational-intelligence-plugin.
    GPU-accelerated defect pattern analysis across 28 paiml GitHub repos.
