# pv — Provable Contracts Specification v2.0.0

**Papers to Math to Contracts in Code.**

A Rust library and CLI for converting peer-reviewed research into
mathematically provable kernel implementations via YAML contracts with
Kani bounded model checking and Lean 4 theorem proving.

**Canonical spec.** This is the ONE spec. All other docs are deprecated.
Sub-specs live in `docs/specifications/sub/` and are linked from this TOC.

---

## Table of Contents

| # | Section | Sub-spec |
|---|---------|----------|
| 1 | [Vision and Architecture](#1-vision-and-architecture) | — |
| 2 | [The Verification Ladder](#2-the-verification-ladder) | — |
| 3 | [Contract Schema](#3-contract-schema) | [sub/schema.md](sub/schema.md), [sub/eiffel-dbc.md](sub/eiffel-dbc.md) |
| 4 | [The Seven-Phase Pipeline](#4-the-seven-phase-pipeline) | [sub/pipeline.md](sub/pipeline.md) |
| 5 | [CLI Reference (`pv`)](#5-cli-reference) | [sub/cli.md](sub/cli.md), [sub/lint.md](sub/lint.md) |
| 6 | [Library API](#6-library-api) | [sub/library.md](sub/library.md) |
| 7 | [Scoring System (`pv score`)](#7-scoring-system) | [sub/scoring.md](sub/scoring.md) |
| 8 | [Query Engine (`pv query`)](#8-query-engine) | [sub/query.md](sub/query.md) |
| 9 | [Proc Macro (`#[contract]`)](#9-proc-macro) | — |
| 10 | [Kernel Contract Registry](#10-kernel-contract-registry) | [sub/registry.md](sub/registry.md) |
| 11 | [Stack Integration](#11-stack-integration) | [sub/integration.md](sub/integration.md) |
| 12 | [References](#12-references) | — |
| 13 | [Escape-Proof Enforcement](#13-escape-proof-enforcement) | [sub/escape-proof-enforcement.md](sub/escape-proof-enforcement.md) |
| 14 | [Lean 4 + Kani Composition](#14-lean-kani-composition) | [sub/lean-kani-composition.md](sub/lean-kani-composition.md) |
| 15 | [Verification Extensions](#15-verification-extensions) | [sub/verification-extensions.md](sub/verification-extensions.md) |
| 16 | [Bidirectional Coverage](#16-bidirectional-coverage) | [sub/bidirectional-coverage.md](sub/bidirectional-coverage.md) |
| 17 | [Gradual Enforcement](#17-gradual-enforcement) | [sub/gradual-enforcement.md](sub/gradual-enforcement.md) |

---

## 1. Vision and Architecture

### The Problem

ML kernel implementations derive from research papers, but the derivation
chain is invisible:

```
Paper (LaTeX) -> Developer's head -> Code -> Tests -> Ship
```

The developer's head is an unauditable black box. When a SIMD kernel
produces wrong results six months later, nobody can trace back to which
equation was violated or which paper assumption broke.

### The Solution

Make the derivation chain explicit, auditable, and provable:

```
Paper (arXiv)
  -> Equations (canonical math)
    -> Contract (YAML, machine-parseable)
      -> Trait (Rust scaffold)
        -> Kernel (scalar, SIMD, PTX)
          -> Tests (probar property tests)
            -> Proof (Kani bounded model checking)
              -> Theorem (Lean 4 unbounded proof)
```

Every link is a concrete artifact in version control. The final links —
Kani and Lean 4 — elevate this from "really good testing" to "actual
proof."

### Theoretical Foundations

| Foundation | Source | Application |
|---|---|---|
| Falsificationism | Popper (1959) | Tests designed to refute, not confirm |
| Poka-Yoke | Shingo (1986) | `ValidatedTensor` makes bad states unrepresentable |
| Jidoka | Ohno (1988) | Stop the line on first defect |
| Design by Contract | Meyer (1988) | Preconditions, postconditions, invariants |
| Type-Driven Dev | Brady (2017) | Parse, don't validate (King 2019) |
| Equation-Driven Dev | batuta oracle | Equation -> failing test -> impl -> verify |
| Bounded Model Checking | Kani (AWS 2022) | Exhaustive verification within bounds |
| Theorem Proving | Lean 4 + Mathlib | Unbounded proofs over mathematical reals |

### Architecture

```
provable-contracts/
+-- crates/
|   +-- provable-contracts/         Library crate
|   |   +-- schema/                 YAML parsing + validation
|   |   +-- scaffold/               Trait + test generation
|   |   +-- kani_gen/               Kani harness codegen
|   |   +-- probar_gen/             Property test codegen
|   |   +-- lean_gen/               Lean 4 codegen
|   |   +-- audit/                  Traceability chain
|   |   +-- scoring/                Contract + codebase scoring
|   |   +-- query/                  BM25 contract search index
|   |   +-- binding.rs              Contract -> impl mapping
|   |   +-- diff.rs                 Version diffing
|   |   +-- coverage.rs             Obligation coverage
|   |   +-- graph.rs                Dependency DAG
|   |   +-- proof_status.rs         L1-L5 levels
|   |   +-- kernels/                Reference implementations
|   |   +-- error.rs                Error types
|   +-- provable-contracts-cli/     CLI binary (`pv`)
|   +-- provable-contracts-macros/  Proc macro (#[contract])
+-- contracts/                      YAML contract registry (165 contracts)
+-- docs/specifications/            This spec
```

### Scale

| Metric | Value |
|---|---|
| YAML contracts | 165 |
| Binding entries (4 crates) | 442 (aprender 301, entrenar 96, realizar 23, trueno 22) |
| Proof obligation types | 12 |
| CLI commands | 18 |
| Consuming projects | 4 Level 3 (aprender, entrenar, realizar, trueno) + 1 YAML-only (bashrs) |
| Stack LoC governed | ~900K Rust |

---

## 2. The Verification Ladder

Every proof obligation is verified at multiple levels. Higher levels
subsume lower ones. The goal is to push every obligation as high as
practically possible.

```
Level   Method                  Tool            Guarantee
-----   ------                  ----            ---------
  L5    Theorem proving         Lean 4          True for ALL inputs. Period.
  L4    Bounded model check     Kani            True for ALL inputs <= size N.
  L3    Property-based test     probar/proptest True for ~10,000 random inputs.
  L2    Falsification test      #[test]         True for specific edge cases.
  L1    Type system             rustc           True by construction.
  L0    Code review             Human eyes      "Looks right to me."
```

### The Provability Claim

When we say a kernel is "provable," we mean:

1. **L1:** The type system prevents invalid construction (Poka-Yoke).
2. **L3:** probar tested the property for 10,000+ random inputs.
3. **L4:** Kani exhaustively verified for ALL inputs within the kernel's
   natural bound (super-block size, SIMD width).

For fixed-size kernel operations — which ML inference IS — bounded
verification at the natural bound IS exhaustive. A Q4_K super-block is
always 256 elements. Verifying for all 256-element inputs IS verifying
for all inputs.

### The Provability Invariant (ENFORCED)

**If a contract has proof obligations, it MUST have Kani harnesses.**
No exceptions. No "we'll add them later." The test suite enforces this:

```
∀ contract C:
  |C.proof_obligations| > 0  →  |C.kani_harnesses| > 0
  |C.proof_obligations| > 0  →  |C.falsification_tests| >= |C.proof_obligations|
```

Contracts are classified into two categories:

| Category | Has equations | Has proof_obligations | Has kani_harnesses | Example |
|---|---|---|---|---|
| **Kernel contract** | REQUIRED | REQUIRED | REQUIRED | softmax-kernel-v1 |
| **Data registry** | REQUIRED | optional | optional | special-tokens-registry-v1 |

A data registry is a contract whose `equations` encode lookup tables,
enum definitions, or configuration bounds — not computable kernels.
Data registries are declared via a `registry: true` field in metadata.
All other contracts are kernel contracts and MUST have the full
provability chain: equations → obligations → falsification tests → Kani.

If we cannot enforce provability on our own contracts, the project has
no reason to exist.

### Where Each Tool Lives

| Obligation Type | L1 (Types) | L3 (probar) | L4 (Kani) | L5 (Lean) |
|---|---|---|---|---|
| Shape correctness | ValidatedTensor | N/A | N/A | N/A |
| Softmax sums to 1 | N/A | proptest random | kani::proof <=16 | Lean theorem |
| SIMD = scalar | N/A | proptest random | kani::proof <=256 | N/A |
| No overflow | N/A | proptest edges | Kani auto | N/A |
| Quantized bsums | N/A | proptest blocks | kani::proof exact | N/A |

### How L4 and L5 Compose

Lean and Kani are NOT alternatives — they verify different things about
the SAME obligation. See **[sub/lean-kani-composition.md](sub/lean-kani-composition.md)**
for the full design.

**Short version:** Lean proves the algorithm over ℝ. Kani proves the Rust
code over f32. The `stub_float` strategy bridges them: Kani replaces
transcendentals (exp, log) with arbitrary-but-constrained values (what
Lean proved valid), then verifies the surrounding code preserves the
invariant. This is compositional: Lean discharges the hard math, Kani
verifies the structural code.

---

## 3. Contract Schema

Every YAML contract follows a fixed schema. Full schema definition,
field reference, and examples in **[sub/schema.md](sub/schema.md)**.

### Top-Level Sections

| Section | Required | Purpose |
|---|---|---|
| `metadata` | yes | Version, author, references, `depends_on` |
| `equations` | yes | Named equations: formula, domain, codomain, invariants |
| `proof_obligations` | **yes** (kernel) / no (registry) | Typed obligations with tolerances |
| `kernel_structure` | no | Phase decomposition |
| `simd_dispatch` | no | Scalar/AVX2/NEON/PTX dispatch table |
| `enforcement` | no | Named rules with severity |
| `falsification_tests` | **yes** (kernel) / no (registry) | Popperian: prediction + test + if_fails |
| `kani_harnesses` | **yes** (kernel) / no (registry) | BMC harness definitions |
| `verification_summary` | no | Lean 4 proof status |
| `qa_gate` | no | certeza integration |

### Proof Obligation Types (26)

**Property types (19):** `invariant`, `equivalence`, `bound`,
`monotonicity`, `idempotency`, `linearity`, `symmetry`,
`associativity`, `conservation`, `ordering`, `completeness`,
`soundness`, `involution`, `determinism`, `roundtrip`, `state_machine`,
`classification`, `independence`, `termination`.

**Eiffel DbC types (7):** `precondition`, `postcondition`, `frame`,
`loop_invariant`, `loop_variant`, `old_state`, `subcontract`.
See **[sub/eiffel-dbc.md](sub/eiffel-dbc.md)** for full definitions.

---

## 4. The Seven-Phase Pipeline

```
Phase 1: EXTRACT    Paper -> canonical math + proof obligations
Phase 2: SPECIFY    Math -> YAML contract (machine-parseable)
Phase 3: SCAFFOLD   Contract -> Rust trait + failing tests
Phase 4: IMPLEMENT  Scalar reference, then SIMD, then PTX
Phase 5: FALSIFY    probar property tests + certeza gates
Phase 6: VERIFY     Kani bounded model checking
Phase 7: PROVE      Lean 4 unbounded theorem proving
```

Every phase produces a committed artifact. No phase is complete until
its artifact is in version control. Full phase details, examples, and
Kani verification strategies in **[sub/pipeline.md](sub/pipeline.md)**.

### Invariant: Every Phase Produces an Artifact

| Phase | Output Artifact |
|---|---|
| Extract | Canonical math + proof obligations |
| Specify | YAML contract |
| Scaffold | Rust trait + failing tests |
| Implement | Scalar + SIMD + PTX kernels |
| Falsify | probar tests + certeza report |
| Verify | Kani proof harnesses + verification report |
| Prove | Lean 4 theorems |

---

## 5. CLI Reference

The `pv` binary provides 18 commands. Full reference with examples, flags, and output formats in
**[sub/cli.md](sub/cli.md)**.

### Command Summary

| Command | Purpose |
|---|---|
| `pv validate <contract>` | Parse + validate YAML against schema |
| `pv scaffold <contract>` | Generate Rust trait + test stubs |
| `pv kani <contract>` | Generate `#[kani::proof]` harnesses |
| `pv probar <contract>` | Generate property tests |
| `pv status <contract>` | Show contract summary |
| `pv audit <contract>` | Traceability: paper -> code chain |
| `pv diff <old> <new>` | Compare versions, suggest semver bump |
| `pv coverage <dir>` | Cross-contract obligation coverage |
| `pv generate <contract> -o <dir>` | Write all artifacts to disk |
| `pv graph <dir>` | Dependency DAG (text/DOT/JSON/Mermaid) |
| `pv equations <contract>` | Render math (text/LaTeX/PTX/ASM) |
| `pv lean <contract>` | Generate Lean 4 files |
| `pv lean-status <dir>` | Lean proof status report |
| `pv proof-status <dir>` | L1-L5 level report |
| `pv book <dir>` | Generate mdBook pages |
| `pv lint <dir>` | Quality gate: validate + audit + score + SARIF |
| `pv score <target>` | Score contract or codebase (A-F grades) |
| `pv query <terms>` | Semantic search with tier/class/graph filters |

---

## 6. Library API

The `provable-contracts` crate exposes 17 public modules. Full API
reference in **[sub/library.md](sub/library.md)**.

### Core API

```rust
// Parse + validate
provable_contracts::schema::parse_contract(path) -> Result<Contract>
provable_contracts::schema::validate_contract(contract) -> Vec<Violation>

// Generate code
provable_contracts::scaffold::generate_trait(contract) -> String
provable_contracts::kani_gen::generate_kani_harnesses(contract) -> String
provable_contracts::probar_gen::generate_probar_tests(contract) -> String
provable_contracts::lean_gen::generate_lean_files(contract) -> Vec<LeanFile>

// Analyze
provable_contracts::audit::audit_contract(contract) -> AuditReport
provable_contracts::coverage::coverage_report(contracts, binding) -> CoverageReport
provable_contracts::diff::diff_contracts(old, new) -> ContractDiff
provable_contracts::graph::dependency_graph(contracts) -> DependencyGraph
provable_contracts::proof_status::proof_status_report(...) -> ProofStatusReport

// Scoring + Query (implemented)
provable_contracts::scoring::score_contract(contract, binding, stem) -> ContractScore
provable_contracts::scoring::score_contract_weighted(contract, binding, stem, weights) -> ContractScore
provable_contracts::scoring::score_codebase(contracts, binding) -> CodebaseScore
provable_contracts::scoring::score_codebase_with_pagerank(contracts, binding, pagerank) -> CodebaseScore
provable_contracts::scoring::score_codebase_full(contracts, binding, pagerank, drift) -> CodebaseScore
provable_contracts::scoring::drift::detect_stale_contracts(dir, binding_path, stems) -> HashSet<String>
provable_contracts::scoring::drift::compute_drift(stale, total) -> f64
provable_contracts::scoring::ScoringWeights { spec_depth, falsification, kani, lean, binding }
provable_contracts::query::ContractIndex::from_directory(dir) -> ContractIndex
provable_contracts::query::ContractIndex::cached_score(stem) -> Option<f64>
provable_contracts::query::ContractIndex::cached_pagerank(stem) -> Option<f64>
provable_contracts::query::ContractIndex::pagerank(iterations, damping) -> HashMap<String, f64>
provable_contracts::query::execute(index, params) -> QueryOutput
provable_contracts::query::QueryOutput::to_markdown() -> String
provable_contracts::query::CrossProjectIndex::build(repo_root) -> CrossProjectIndex
provable_contracts::query::CrossProjectIndex::build_with_extra(root, extra) -> CrossProjectIndex
provable_contracts::query::CrossProjectIndex::call_sites_for(stem) -> &[CallSite]
provable_contracts::query::CrossProjectIndex::binding_refs_for(stem) -> &[BindingRef]
provable_contracts::query::CrossProjectIndex::commit_refs_for(pattern) -> &[CommitRef]
provable_contracts::query::ContractIndex::from_directory_opts(dir, force) -> ContractIndex
```

---

## 7. Scoring System

`pv score` provides quantitative quality assessment for individual
contracts and entire codebases. Full scoring methodology, formulas,
and grade thresholds in **[sub/scoring.md](sub/scoring.md)**.

### Five Scoring Dimensions (Contract)

| # | Dimension | Weight | Measures |
|---|---|---|---|
| D1 | Specification Depth | 20% | Equations, domains, invariants, tolerances |
| D2 | Falsification Coverage | 25% | Obligations with tests / total obligations |
| D3 | Kani Proof Coverage | 25% | Obligations with harnesses (strategy-weighted) |
| D4 | Lean Proof Coverage | 10% | Obligations with proved Lean theorems |
| D5 | Binding Coverage | 20% | Equations with implemented bindings |

### Five Scoring Dimensions (Codebase)

| # | Dimension | Weight | Measures |
|---|---|---|---|
| CD1 | Contract Coverage | 30% | Kernel functions with contracts |
| CD2 | Binding Completeness | 20% | Implemented / total bindings |
| CD3 | Mean Contract Score | 20% | Avg composite of bound contracts |
| CD4 | Proof Depth Distribution | 15% | Weighted L1-L5 distribution |
| CD5 | Drift Detection | 15% | Contract freshness vs code |

### Grade Thresholds

| Grade | Range | Meaning |
|---|---|---|
| A | >= 0.90 | Exemplary |
| B | >= 0.75 | Strong |
| C | >= 0.60 | Adequate |
| D | >= 0.40 | Weak |
| F | < 0.40 | Deficient |

---

## 8. Query Engine

`pv query` provides O(1) semantic search across all 165+ contracts
AND their consumer projects. Inspired by `pmat query` from
paiml-mcp-agent-toolkit. Full query architecture, index format, and
enrichment flags in **[sub/query.md](sub/query.md)**.

### Index Architecture

Multi-index hybrid approach for sub-second lookups (modeled on
`pmat query`):

| Index | Data Structure | Lookup |
|---|---|---|
| Name index | `HashMap<stem, Vec<idx>>` | O(1) |
| Equation index | `HashMap<eq_name, Vec<idx>>` | O(1) |
| Full-text corpus | In-memory BM25 | O(n), n=165 |
| Dependency DAG | `BTreeMap<String, Vec<String>>` | O(1) |
| Score cache [IMPLEMENTED] | `HashMap<stem, f64>` | O(1) |
| **Cross-project index** | `HashMap<stem, Vec<ProjectRef>>` | O(1) |

### Cross-Project Search (Automatic)

`pv query` automatically discovers and indexes sibling project
directories. No configuration needed — it walks `../` for known
consumers:

```bash
pv query "softmax"                     # Shows contract + all call sites
pv query "rmsnorm" --project aprender  # Filter to one project
pv query --binding-gaps --all-projects # Gaps across entire stack
pv query --violations                  # Contracts violated in any project
```

Cross-project data sources:
- `#[contract(...)]` annotations in `.rs` files (aprender)
- `binding.yaml` entries per consumer crate
- KAIZEN ticket refs (`KAIZEN-NNN`, `C-*-NNN`) in commit messages + code
- `Cargo.toml` dependency declarations on `provable-contracts`

### Query Modes [IMPLEMENTED]

```bash
pv query "softmax stability"           # Semantic (BM25)
pv query --regex "SM-INV-\d+"         # Regex
pv query --literal "kani::proof"      # Exact match
pv query --obligation invariant       # Filter by type
pv query --unproven                   # Gaps only
pv query --depends-on softmax         # DAG traversal
pv query --min-score 0.5             # Score threshold
pv query --binding-gaps --binding b   # Unimplemented bindings
pv query --include-project ../trueno  # Explicit project path
```

### Enrichment [IMPLEMENTED]: `--score`, `--proof-status`, `--binding-info`,
`--graph`, `--paper`, `--diff`. Output: `-f text|json|markdown`.

---

## 9. Proc Macro

The `#[contract]` attribute from `provable-contracts-macros` provides
compile-time contract enforcement.

### Usage

```rust
#[provable_contracts_macros::contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    // Implementation must satisfy contract obligations
}
```

### Mechanism

1. **build.rs** in the consuming crate reads `binding.yaml` and sets
   `CONTRACT_<NAME>_<EQ>=<status>` env vars for each binding.

2. `#[contract("name", equation = "eq")]` expands to a const that reads
   the env var via `option_env!()`. Missing env var = soft warning
   (crates.io compat). `AllImplemented` policy in build.rs = hard
   failure for unbound equations not in `ALLOWED_GAPS`.

3. A static binding string registers the function for runtime audit
   traceability.

### Enforcement Policy

```rust
// build.rs: ALLOWED_GAPS whitelist
const ALLOWED_GAPS: &[(&str, &str)] = &[
    ("ssm-kernel-v1", "ssm_discretize"),
    ("ssm-kernel-v1", "ssm_scan"),
];
// Any not_implemented binding NOT in ALLOWED_GAPS = build failure
```

---

## 10. Kernel Contract Registry

Full registry of all 165 contracts, organized by tier and kernel
equivalence class, in **[sub/registry.md](sub/registry.md)**.

### Kernel Equivalence Classes

| Class | Kernels | Architecture |
|---|---|---|
| A | GQA + RMSNorm + SiLU + SwiGLU + RoPE | Llama, Mistral |
| B | MHA + LayerNorm + GELU + AbsPos | GPT-2, BERT |
| C | MHA + LayerNorm + GELU + ALiBi | BLOOM, MPT |
| D | LayerNorm + GELU + SiLU + GQA | Gemma |
| E | RMSNorm + SwiGLU + GQA | Qwen |

### Contract Tiers

| Tier | Scope | Count | Coverage |
|---|---|---|---|
| Tier 1 | Foundation kernels (softmax, rmsnorm, rope, silu) | 12 | 100% |
| Tier 2 | Composite kernels (attention, matmul, flash-attn) | 8 | 100% |
| Tier 3 | System kernels (kv-cache, sampling) | 15 | 90% |
| Tier 4 | Training kernels (adamw, cross-entropy, lora) | 25 | 85% |
| Tier 5 | Classical ML (kmeans, pagerank, pca, svm) | 40+ | 70% |
| Tier 6 | Model-specific (qwen2, qwen3, qwen3.5) | 12 | 100% |
| Tier 7 | Performance (KAIZEN buffer/GPU contracts) | 50+ | N/A |

---

## 11. Stack Integration

provable-contracts serves as the contract authority for the PAIML
Sovereign AI stack. Full integration details, consumer patterns, and
KAIZEN workflow in **[sub/integration.md](sub/integration.md)**.

### Consumer Projects

| Project | Bindings | Policy | Integration |
|---|---|---|---|
| aprender | 330 | **AllImplemented** | Level 3: `build.rs` + `#[contract]` proc macro |
| entrenar | 117 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| trueno | 42 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| realizar | 58 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| forjar | 13 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| presentar | 5 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| rmedia | 16 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |

### The KAIZEN Contract-First Workflow

Proven by 32+ KAIZEN tickets with measurable results:

```
1. Profile reveals bottleneck (entrenar CUDA training)
2. Write contract in provable-contracts (YAML with obligations)
3. Implement optimization in entrenar/trueno (code)
4. Measure improvement (profiling baseline)
5. Contract serves as regression gate going forward
```

### Evidence

| Ticket | Contract | Result |
|---|---|---|
| KAIZEN-048 | embed-grad-zero-copy-v1 | embed_bwd 145ms -> 0.55ms (263x) |
| KAIZEN-052 | C-XENT-003 | Eliminated 77.8MB/step GPU alloc |
| KAIZEN-066 | C-GPUNORM-001 | Eliminated CPU D2H->H2D round-trip |

---

## 12. References

### Methodology

1. Popper, K. (1959). *The Logic of Scientific Discovery.*
2. Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System.*
3. Meyer, B. (1988). *Object-Oriented Software Construction.*
4. Brady, E. (2017). *Type-Driven Development with Idris.*
5. King, A. (2019). "Parse, Don't Validate."
6. Williams, S. et al. (2009). "Roofline: An Insightful Visual Performance Model." CACM 52(4).

### ML Kernels

7. Vaswani, A. et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
8. Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization." arXiv:1910.10683.
9. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with RoPE." arXiv:2104.09864.
10. Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
11. Dao, T. et al. (2022). "FlashAttention." arXiv:2205.14135.
12. Dao, T. (2023). "FlashAttention-2." arXiv:2307.08691.

### Formal Verification

13. Kani Contributors (2022-2026). "Kani Rust Verifier." github.com/model-checking/kani
14. VanHattum, A. et al. (2022). "Verifying Dynamic Trait Objects in Rust." ICSE-SEIP 2022.
15. Rust Std Lib Verification (2025). arXiv:2510.01072.
16. Dubey, K. (2025). "Equivalence Checking of ML GPU Kernels." arXiv:2511.12638.
17. ProofWright (2025). "Agentic Formal Verification of CUDA." arXiv:2511.12294.
18. Dagstuhl Seminar 26031 (2026). "Software Contracts Meet System Contracts."
19. VerifyThisBench (2025). arXiv:2505.19271.

### Quality Gates and Toolchain Integration

20. OASIS (2020). *Static Analysis Results Interchange Format (SARIF) Version 2.1.0.*
21. Feist, J. et al. (2024). "Integrating Static Code Analysis Toolchains." arXiv:2403.05986.
22. Nachman, L. et al. (2025). "Dealing with SonarQube Cloud." arXiv:2508.18816.
23. Yang, J. et al. (2025). "CodeCureAgent: Automatic Classification and Repair of Static Analysis Warnings." arXiv:2509.11787.
24. Shestov, A. et al. (2025). "Augmenting LLMs with Static Code Analysis." arXiv:2506.10330.
25. Molnar, A. & Motogna, S. (2024). "Versioned Analysis of Software Quality Indicators." arXiv:2407.15967.
26. Singh, G. et al. (2022). "Interactive Abstract Interpretation." arXiv:2209.10445.
27. Li, Y. et al. (2025). "Do Large Language Models Respect Contracts?" arXiv:2510.12047.
28. Bruni, R. et al. (2026). "Agent Behavioral Contracts." arXiv:2602.22302.

### Gradual Typing and Verification

29. Siek, J. G. & Taha, W. (2006). "Gradual Typing for Functional Languages." Scheme and Functional Programming Workshop.
30. Bader, J., Aldrich, J. & Tanter, E. (2018). "Gradual Program Verification." VMCAI 2018. arXiv:1710.06422.
31. Lehmann, N. & Tanter, E. (2023). "Gradual Liquid Type Inference." OOPSLA 2023.
32. Garcia, R., Clark, A. & Tanter, E. (2016). "Abstracting Gradual Typing." POPL 2016.
33. Rondon, P. M., Kawaguci, M. & Jhala, R. (2008). "Liquid Types." PLDI 2008.

---

## 13. Escape-Proof Enforcement

**Sub-spec**: [sub/escape-proof-enforcement.md](sub/escape-proof-enforcement.md)

Six-stage pipeline where each stage gates the next. Skip one → compile error.
Equation (YAML) → Lean 4 proof (no sorry) → YAML validation (pv lint) →
build.rs codegen (debug_assert from preconditions) → #[contract] macro
(compile-time binding check) → test execution (falsification tests pass).

Zero runtime cost. Release binary identical to one built without contracts.
Inspired by SPARK/Ada (proof discharge), Eiffel (contract inheritance),
Dafny (verification conditions), Lean 4 (theorem proving).

---

## 15. Verification Extensions

**Sub-spec**: [sub/verification-extensions.md](sub/verification-extensions.md)

Six orthogonal verification approaches that complement the existing pipeline:

1. **Type Invariants** — `Invariant` trait (stable) or `#[contracts::invariant]` (nightly) with Kani preservation harnesses
2. **Coq Theorem Proving** — `pv coq` generates `.v` stubs; `coq-of-rust` bridges Rust → Coq for implementation-level proofs
3. **Coverage-Guided Fuzzing** — `pv fuzz` generates libfuzzer targets gated on contract preconditions
4. **Abstract Interpretation (MIRAI)** — `pv mirai` generates `precondition!`/`postcondition!` annotations for sound over-approximation
5. **Refinement Types (Flux)** — `pv flux` generates `#[flux::refined_by]` annotations for compile-time shape verification via SMT
6. **System-Level Model Checking (TLA+)** — `pv tla` generates TLA+ modules from the contract dependency DAG for pipeline-level safety/liveness

---

## 16. Bidirectional Coverage

**Sub-spec**: [sub/bidirectional-coverage.md](sub/bidirectional-coverage.md)

Current enforcement is unidirectional (binding → implementation). Bidirectional
coverage adds the reverse check: implementation → binding. Three mechanisms:

1. **`pv coverage --reverse`** — Static API diff: scan pub fns, report unbound
2. **`#[must_contract]`** — Compile-time lint for unannotated pub fns
3. **`pv infer`** — Semantic matching: suggest contracts for unbound functions

`pv lint` Gate 7 enforces reverse coverage threshold. This prevents
whack-a-mole: new functions cannot escape the contract system silently.

---

## 17. Gradual Enforcement

**Sub-spec**: [sub/gradual-enforcement.md](sub/gradual-enforcement.md)

Five enforcement gaps identified by falsifying against mypy, TypeScript,
Rust `#[forbid]`, C# nullable, JSpecify, Haskell/LiquidHaskell, ty, and Elm:

1. **Per-contract enforcement levels** — `metadata.enforcement_level: basic | standard | strict | proven`
   (pattern: mypy per-module, C# per-file `#nullable`, JSpecify `@NullMarked`)
2. **Stale suppression detection** — `PV-SUP-001` warns when suppressions become unnecessary
   (pattern: TypeScript `@ts-expect-error`, Rust `#[expect]`, ty `unused-ignore-comment`)
3. **Multi-stage pipeline** — Four verification tiers with progressive CI gates
   (pattern: C# `disable → warnings → annotations → enable`)
4. **Aggregate coverage metric** — `pv lint --coverage --min-coverage 0.70` with CI ratchet
   (pattern: TypeScript `type-coverage`, mypy typed def count)
5. **Irreversible level lock** — `metadata.locked_level: L3` cannot regress without `pv unlock`
   (pattern: Rust `#![forbid(unsafe_code)]`, Elm mandatory totality)

Key references: Bader et al. (2018) "Gradual Program Verification" arXiv:1710.06422;
Lehmann & Tanter (2023) "Gradual Liquid Type Inference" OOPSLA;
Meyer (2025) "Software engineering as a domain to formalize" arXiv:2502.11434.
