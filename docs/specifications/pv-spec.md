# pv — Provable Contracts Specification v2.2.0

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
| 4 | [The Seven-Phase Pipeline](#4-the-seven-phase-pipeline) | [sub/pipeline.md](sub/pipeline.md), [sub/pytorch-extraction.md](sub/pytorch-extraction.md) |
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
| 18 | [PVScore](#18-pvscore) | [sub/pvscore.md](sub/pvscore.md) |
| 19 | [Sovereign Stack Audit](#19-sovereign-stack-audit) | [sub/sovereign-stack-audit.md](sub/sovereign-stack-audit.md) |
| 20 | [UX, Speech, Probar](#20-ux-speech-probar) | [sub/ux-speech-probar.md](sub/ux-speech-probar.md) |
| 21 | [Contract Gap Analysis](#21-contract-gap-analysis) | [sub/contract-gaps.md](sub/contract-gaps.md) |
| 22 | [Diagnostic Output](#22-diagnostic-output) | [sub/diagnostics.md](sub/diagnostics.md) |
| 23 | [Contract-Trait Enforcement](#23-contract-trait-enforcement) | [sub/contract-trait-enforcement.md](sub/contract-trait-enforcement.md) |
| 24 | [Deep Stack Integration](#24-deep-stack-integration) | [sub/deep-integration.md](sub/deep-integration.md) |
| 25 | [Full Enforcement Mandate](#25-full-enforcement-mandate) | — |
| 26 | [Two-Tier Architecture and Compositional Contracts](#26-two-tier-architecture-and-compositional-contracts) | — |
| 27 | [The One Way](#27-the-one-way) | — |
| 28 | [Correctness + Completeness](#28-correctness--completeness) | — |

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
+-- contracts/                      YAML contract registry (201 contracts)
+-- docs/specifications/            This spec
```

### Scale

| Metric | Value | Verified |
|---|---|---|
| YAML contracts | 201 | `find contracts/ -name '*.yaml' ! -name 'binding.yaml' \| wc -l` |
| Equations | 427 | `pv coverage contracts/` |
| Proof obligations | 678 | `pv coverage contracts/` |
| Falsification tests | 714 | `pv coverage contracts/` |
| Kani harnesses (YAML-defined) | 871 | `pv coverage contracts/` |
| **Real bindings (with module_path)** | **540** | Ghost bindings stripped 2026-03-28 |
| Binding repos with entries | 26 directories, 12 with real bindings | `ls contracts/*/binding.yaml` |
| Proof obligation types | 26 (19 property + 7 Eiffel DbC) | schema/types.rs |
| CLI commands | 33 | `pv --help` (includes `pv pipeline`) |
| Repos with build.rs enforcement | 7/26 | aprender, trueno, entrenar, realizar, forjar, ruchy, simular |
| Repos with trait tests | 12/26 | manual audit 2026-03-28 |
| `#[contract]` proc-macro annotations | 18 | forjar: 4, paiml-mcp-agent-toolkit: 11, batuta: 3 |
| Stack LoC governed | ~6.4M Rust | — |

> **v2.2.0 Correction (2026-03-28):** Previous versions inflated binding
> counts with mass-generated entries lacking `module_path`. v2.2.0 stripped
> 28,206 ghost bindings, leaving 540 real bindings that reference actual
> module paths. Of those, ~234 resolve to functions in source code.
> The honest binding rate is **234 verified / 427 equations = 55%** for
> the best-covered repo (aprender), not the 100% previously claimed.

---

## 2. The Verification Ladder

Two complementary hierarchies: **proof levels** (what we verify about
the math) and **enforcement layers** (how we enforce it in the build).

### Proof Levels (theoretical guarantees)

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

### Enforcement Layers (practical deployment, strictest first)

| Layer | What it catches | Mechanism | Coverage | Misses |
|-------|----------------|-----------|----------|--------|
| **L5** | Algorithm incorrect | Lean 4 proof (no sorry) | 3 theorems (softmax) | — |
| **L4** | Logic bugs, overflows | Kani `#[kani::proof]` BMC | 871 harnesses (YAML-defined) | Inputs > bound |
| **L3** | Violated invariants | `#[contract]` debug_assert | 18 functions (forjar: 4, pmat: 11, batuta: 3) | Release builds |
| **L2** | Renamed/deleted fns | Trait `impl` (§23) | 12/26 repos have trait tests | Logic bugs |
| **L1** | Missing bindings | build.rs AllImplemented | 540 real bindings (~234 verified) | Ghost bindings |
| **L0.5** | Schema/audit/score | `pv lint` 7 gates | 131/131 contracts pass | Impl bugs |
| **L0** | Obvious bugs | Human review | — | Everything subtle |

**L0 through L2 enforce on every `cargo build` + `cargo test`** in
the 7 repos with build.rs (aprender, trueno, entrenar, realizar,
forjar, ruchy, simular). The other 19 repos have YAML bindings only.

L3 enforces on 18 annotated functions across forjar, pmat, and batuta
debug builds. L4 and L5 are defined in YAML but not yet run in CI.

> **Spec Falsification (2026-03-28, v2.2.0):** Round 3 stripped 28,206
> ghost bindings (mass-generated entries without `module_path`). Honest
> count: 540 real bindings, ~234 verified in source. Previous claim of
> 20,366 bindings / "Grade A for 26 repos" was a scoring artifact of
> YAML inflation, not real integration. See §25 for corrected baseline.

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

The `pv` binary provides 30 commands. Full reference with examples, flags, and output formats in
**[sub/cli.md](sub/cli.md)**.

### Command Summary

| Command | Purpose |
|---|---|
| `pv explain <contract>` | Narrative walkthrough of equations, obligations, verification |
| `pv validate <contract>` | Parse + validate YAML against schema |
| `pv scaffold <contract>` | Generate Rust trait + test stubs (`--trait` for standalone) |
| `pv kani <contract>` | Generate `#[kani::proof]` harnesses |
| `pv probar <contract>` | Generate property tests |
| `pv status <contract>` | Show contract summary |
| `pv audit <contract>` | Traceability: paper -> code chain |
| `pv diff <old> <new>` | Compare versions, suggest semver bump |
| `pv coverage <dir>` | Cross-contract obligation coverage (`--reverse`, `--fuzz`) |
| `pv generate <contract> -o <dir>` | Write all artifacts to disk (`--readme`, `--ci`) |
| `pv graph <dir>` | Dependency DAG (text/DOT/JSON/Mermaid) |
| `pv equations <contract>` | Render math (text/LaTeX/PTX/ASM) |
| `pv lean <contract>` | Generate Lean 4 files |
| `pv lean-status <dir>` | Lean proof status report |
| `pv proof-status <dir>` | L1-L5 level report |
| `pv book <dir>` | Generate mdBook pages |
| `pv lint <dir>` | Quality gate: 7 gates + SARIF (`--min-level`, `--coverage`) |
| `pv score <target>` | Score contract or codebase (`--pvscore` for 10-dim) |
| `pv query <terms>` | Semantic search with tier/class/graph filters |
| `pv extract-pytorch <target>` | Extract kernel equations from PyTorch source |
| `pv codegen <dir>` | Generate Rust code from contracts |
| `pv invariants <contract>` | Generate type invariant implementations |
| `pv coq <contract>` | Generate Coq `.v` theorem stubs |
| `pv fuzz <contract>` | Generate libfuzzer targets |
| `pv mirai <contract>` | Generate MIRAI annotations |
| `pv flux <contract>` | Generate Flux refinement types |
| `pv tla <dir>` | Generate TLA+ system-level specs |
| `pv infer <crate>` | Auto-suggest contracts for unbound functions |
| `pv unlock <contract>` | Remove enforcement level lock (`--reason`) |

---

## 6. Library API

The `provable-contracts` crate exposes 35 public modules. Full API
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

`pv query` provides O(1) semantic search across all 182+ contracts
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
| Full-text corpus | In-memory BM25 | O(n), n=193 |
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

### Four-Layer Enforcement

| Layer | What it checks | When | Coverage |
|-------|---------------|------|----------|
| **L1: build.rs AllImplemented** | binding.yaml has no `not_implemented` | `cargo build` | 16,989 bindings |
| **L2: Trait `impl`** | Function exists + correct signature | Rust compiler | Per-contract (§23) |
| **L3: `#[contract]` macro** | Env var + `debug_assert!()` injection | Compile-time | Per-function |
| **L4: `pv lint --reverse`** | Every `pub fn` has a binding | CI gate | Full crate scan |

**Layer 1 (build.rs)** catches registry completeness — you can't mark
a binding as `not_implemented` without the build failing. Deployed in
all 13 repos.

**Layer 2 (Trait `impl`)** is the permanent, compiler-enforced fix.
`pv scaffold --trait` generates a Rust trait per contract from YAML
equations. Consumer crates `impl` the trait. Missing function = compile
error. Wrong signature = compile error. No build.rs scanning needed.
See **Section 23** for full design, chain-of-thought reasoning, and
references from SPARK/Ada, Eiffel, Kani, Prusti, and Creusot.

**Layer 3 (`#[contract]` macro)** injects `debug_assert!()` for
preconditions/postconditions from YAML. Uses `option_env!()` (soft) not
`env!()` (hard) for crates.io compat. Zero cost in release builds.

**Layer 4 (`pv lint --reverse`)** closes the opposite gap — detects
`pub fn` declarations that have no binding entry. Prevents new functions
from escaping the contract system.

### Usage

```rust
// Layer 2: trait generated by `pv scaffold --trait contracts/softmax-kernel-v1.yaml`
pub trait SoftmaxKernelV1 {
    fn softmax(&self, x: &[f32]) -> Vec<f32>;
    fn log_softmax(&self, x: &[f32]) -> Vec<f32>;
}

// Consumer impl — missing method = compile error
impl SoftmaxKernelV1 for NNFunctional {
    fn softmax(&self, x: &[f32]) -> Vec<f32> { /* ... */ }
    fn log_softmax(&self, x: &[f32]) -> Vec<f32> { /* ... */ }
}
```

```rust
// Layer 3: debug_assert injection from YAML pre/postconditions
#[provable_contracts_macros::contract("softmax-kernel-v1", equation = "softmax")]
pub fn softmax_1d(x: &[f32]) -> Vec<f32> {
    // debug_assert!(!x.is_empty()) injected from YAML preconditions
}
```

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

Full registry of all 182 contracts, organized by tier and kernel
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
| pmat | 2,665 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| aprender | 2,363 | **AllImplemented** | Level 3: `build.rs` + `#[contract]` proc macro |
| entrenar | 1,868 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| presentar | 1,824 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| realizar | 1,725 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| ruchy | 1,681 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| trueno | 98 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| depyler | 1,451 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| bashrs | 1,056 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| forjar | 819 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| simular | 566 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| decy | 456 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |
| rmedia | 405 | **AllImplemented** | Level 3: `build.rs` + `CONTRACT_*` env vars |

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

## 14. Lean 4 + Kani Composition

**Sub-spec**: [sub/lean-kani-composition.md](sub/lean-kani-composition.md)

Lean and Kani are NOT alternatives — they verify different things about
the SAME obligation. Lean proves the algorithm over ℝ. Kani proves the
Rust code over f32. The `stub_float` strategy bridges them: Kani replaces
transcendentals (exp, log) with arbitrary-but-constrained values (what
Lean proved valid), then verifies the surrounding code preserves the
invariant. This is compositional: Lean discharges the hard math, Kani
verifies the structural code.

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

1. **`pv coverage --reverse`** [IMPLEMENTED] — Static API diff: scan pub fns, report unbound
2. **`#[must_contract]`** [IMPLEMENTED] — Compile-time lint for unannotated pub fns
3. **`pv infer`** [IMPLEMENTED] — Semantic matching: suggest contracts for unbound functions

`pv lint` Gate 7 [IMPLEMENTED] enforces reverse coverage threshold. This prevents
whack-a-mole: new functions cannot escape the contract system silently.

---

## 17. Gradual Enforcement

**Sub-spec**: [sub/gradual-enforcement.md](sub/gradual-enforcement.md)

Five enforcement gaps identified by falsifying against mypy, TypeScript,
Rust `#[forbid]`, C# nullable, JSpecify, Haskell/LiquidHaskell, ty, and Elm:

1. **Per-contract enforcement levels** [IMPLEMENTED] — `metadata.enforcement_level: basic | standard | strict | proven`
   (pattern: mypy per-module, C# per-file `#nullable`, JSpecify `@NullMarked`)
2. **Stale suppression detection** [IMPLEMENTED] — `PV-SUP-001` warns when suppressions become unnecessary
   (pattern: TypeScript `@ts-expect-error`, Rust `#[expect]`, ty `unused-ignore-comment`)
3. **Multi-stage pipeline** [IMPLEMENTED] — Four verification tiers with progressive CI gates (`pv lint --min-level`)
   (pattern: C# `disable → warnings → annotations → enable`)
4. **Aggregate coverage metric** [IMPLEMENTED] — `pv lint --coverage --min-coverage 0.70` with CI ratchet
   (pattern: TypeScript `type-coverage`, mypy typed def count)
5. **Irreversible level lock** [IMPLEMENTED] — `metadata.locked_level: L3` cannot regress without `pv unlock`
   (pattern: Rust `#![forbid(unsafe_code)]`, Elm mandatory totality)

Key references: Bader et al. (2018) "Gradual Program Verification" arXiv:1710.06422;
Lehmann & Tanter (2023) "Gradual Liquid Type Inference" OOPSLA;
Meyer (2025) "Software engineering as a domain to formalize" arXiv:2502.11434.

---

## 18. PVScore (`pv score .`)

**Sub-spec**: [sub/pvscore.md](sub/pvscore.md)

`pv score` has three modes: contract (`.yaml` → 5-dim 0.0-1.0),
codebase (directory + binding → 5-dim), and **project** (`.` or
`Cargo.toml` root → 10-dim 0-100 geometric mean). PVScore is the
project mode. Grade A (90+) required for CI merge — HARD requirement.

**10 Dimensions** (all 0-100, geometric mean):

| # | Dimension | Source | Hard to fake because |
|---|---|---|---|
| D1 | Spec Depth | `pv score` D1 | Requires actual math from papers | [IMPLEMENTED] |
| D2 | Falsification Coverage | `pv score` D2 | Property tests, not unit tests | [IMPLEMENTED] |
| D3 | Kani BMC | `pv score` D3 | Prover actually runs | [IMPLEMENTED] |
| D4 | Lean 4 Proofs | `pv score` D4 | `sorry` = 0 points | [IMPLEMENTED] |
| D5 | Binding Compliance | `pv score` D5 | Compiler rejects gaps | [IMPLEMENTED] |
| D6 | Reverse Coverage | `pv coverage --reverse` | Scans source, not YAML | [IMPLEMENTED] |
| D7 | Mutation Testing | certeza / cargo-mutants | Ultimate test quality metric | [IMPLEMENTED] (default 1.0) |
| D8 | CI Pipeline Depth | GitHub Actions audit | Auditable CI logs | [IMPLEMENTED] (default 1.0) |
| D9 | Proof Freshness | Kani/Lean CI timestamps | Stale proofs decay to 0 | [IMPLEMENTED] (default 1.0) |
| D10 | Defect Patterns | org-intelligence-plugin | Git history analysis | [IMPLEMENTED] (default 1.0) |

```
pvscore = (D1 * D2 * ... * D10) ^ (1/10)
```

One zero dimension tanks the entire score. You cannot compensate
weak formal proofs with good test coverage.

Key references: SQALE (Letouzey 2012), OpenSSF Scorecard (2023),
Petrovic et al. (2022) "Practical Mutation Testing at Scale" IEEE TSE.

---

## 19. Sovereign Stack Audit

**Sub-spec**: [sub/sovereign-stack-audit.md](sub/sovereign-stack-audit.md)

Full audit of all 13 repos in the PAIML sovereign AI stack.
**6.4M LOC. 100% under contract enforcement. 16,989 bindings.**

| Project | Bindings | Policy | Status |
|---|---|---|---|
| pmat | 2,665 | AllImplemented | Level 3 ✓ |
| aprender | 2,363 | AllImplemented | Level 3 ✓ |
| entrenar | 1,868 | AllImplemented | Level 3 ✓ |
| presentar | 1,824 | AllImplemented | Level 3 ✓ |
| realizar | 1,725 | AllImplemented | Level 3 ✓ |
| ruchy | 1,681 | AllImplemented | Level 3 ✓ |
| trueno | 98 | AllImplemented | Level 3 ✓ |
| depyler | 1,451 | AllImplemented | Level 3 ✓ |
| bashrs | 1,056 | AllImplemented | Level 3 ✓ |
| forjar | 819 | AllImplemented | Level 3 ✓ |
| simular | 566 | AllImplemented | Level 3 ✓ |
| decy | 456 | AllImplemented | Level 3 ✓ |
| rmedia | 405 | AllImplemented | Level 3 ✓ |

Zero unenforced repos remain. The enforcer (pmat) enforces itself.

**No-escape plan:**
1. **CB-1300 mandate** — every paiml Rust repo >10K LOC MUST have contracts
2. **Transpiler contracts first** — depyler/ruchy bugs propagate to all transpiled programs
3. **pmat self-enforcement** — the enforcer eats its own dogfood
4. **Security tool contracts** — bashrs/rash correctness IS the security guarantee
5. **Reverse coverage ratchet** — CI-gated targets: 25% at 6mo, 50% at 12mo
6. **PVScore gate at month 3** — unified 10-dim score, A >= 90 required

---

## 20. UX, Speech, Probar

**Sub-spec**: [sub/ux-speech-probar.md](sub/ux-speech-probar.md)

Four UX contract categories with proof methods:

| Category | Example | Proof Method |
|---|---|---|
| Geometric invariants | `Rect::intersection` commutativity | Kani exhaustive |
| Perceptual correctness | WCAG contrast ratio | Kani bounded + probar |
| Pipeline correctness | Privacy routing, template XSS | Kani + TLA+ |
| Visual regression | Screenshot pixel diff | probar + golden tests |

Whisper.apr contracts: APR serialization roundtrip, mel spectrogram
bounds, transcription timestamp monotonicity, language detection accuracy.

probar integration: property tests as PVScore D2 data source. probar
reports pass rate + coverage → feeds into PVScore geometric mean.

apr-model-qa-playbook: MQS (Model Quality Score, 0-1000) certifies
individual models. Composes with PVScore for codebase + model quality.

---

## 21. Contract Gap Analysis

**Sub-spec**: [sub/contract-gaps.md](sub/contract-gaps.md)

Systematic analysis of 9 ML/systems domains against the contract registry.
193 contracts cover core kernels well; significant gaps in:

| Domain | Gap Severity | Key Missing |
|---|---|---|
| Training infrastructure | **Major** | Allreduce, LR schedulers, gradient clipping |
| Quantization | Partial | GPTQ, AWQ, FP8, QLoRA |
| Attention variants | Partial | Flash-Decoding v2, Ring Attention, MLA |
| Memory management | **Major** | PagedAttention, speculative decoding |
| Numerical precision | Mostly missing | BF16, FP8, stochastic rounding |
| Tokenization | **Absent** | BPE, sequence packing |
| Post-training/alignment | **Absent** | DPO, PPO/GRPO, reward model |
| Shape algebra | Partial | Broadcast, stride, einsum, operator inference |
| Inference serving | Mostly missing | Continuous batching, tensor parallelism |

**Top 5 highest-leverage additions** (impact x Kani tractability):
1. Speculative decoding — tractable BMC, clear acceptance criterion
2. FP8 e4m3/e5m2 interchange — small state space, needed for trueno
3. DPO loss — single equation, known failure modes
4. BPE tokenization — merge associativity and round-trip provable
5. PagedAttention — pointer aliasing, exactly what Kani excels at

---

## 22. Diagnostic Output

**Sub-spec**: [sub/diagnostics.md](sub/diagnostics.md)

Falsification against 9 reference tools (Kani, SPARK/Ada, Dafny, Clippy,
mypy, ESLint, SonarQube, cargo-deny, OpenSSF Scorecard) revealed 13 gaps.

**P0 (implemented):**

1. **Grouped finding display** — findings grouped by contract, then by rule.
   Per-contract summary with error/warning counts. (pattern: ESLint, Clippy)
2. **Color terminal output** — ANSI red/yellow/cyan/bold/green. `--color`
   flag with auto/always/never. (pattern: Clippy, mypy, cargo-deny)
3. **`pv lint --explain <rule>`** — long-form remediation guidance per rule
   ID with description, why-it-matters, how-to-fix, references.
   (pattern: Clippy `--explain`, Rust `--explain E0308`)

**P1-P3 (planned):** probe-level score decomposition, source snippets with
caret spans, per-obligation verification table, counterexample/evidence
data, remediation effort estimation, issue lifecycle, structured fix
patches, per-contract resource metrics, HTML reports, daemon/LSP mode.

---

## 23. Contract-Trait Enforcement

**Sub-spec**: [sub/contract-trait-enforcement.md](sub/contract-trait-enforcement.md)

**APPROVED DESIGN.** The permanent, one-time fix for binding verification.

### Problem

build.rs checks that binding.yaml *says* "implemented" — but nothing
verifies the function *actually exists* with the *correct signature*.
Of 16,989 bindings, only 35 have `#[contract]` annotations. Build.rs
source scanning is fragile: string-matching `pub fn` misses `impl`
methods, 13 copy-pasted scanners, name-only without signature checking.

### Solution: Generated Traits

**Generate Rust traits from YAML contracts. Require consumer crates to
`impl` them.** The Rust compiler becomes the enforcement mechanism.

```
YAML Contract           →  pv scaffold --trait       →  Generated Trait
softmax-kernel-v1.yaml                                  pub trait SoftmaxKernelV1 {
  equations:                                                fn softmax(&self, x: &[f32]) -> Vec<f32>;
    softmax:                                                fn log_softmax(&self, x: &[f32]) -> Vec<f32>;
    log_softmax:                                        }

Consumer Impl (compile error if missing/wrong):
    impl SoftmaxKernelV1 for NNFunctional {
        fn softmax(&self, x: &[f32]) -> Vec<f32> { /* ... */ }
        fn log_softmax(&self, x: &[f32]) -> Vec<f32> { /* ... */ }
    }
```

If YAML changes → trait changes → `impl` breaks → **compiler catches it**.
No build.rs. No scanning. No name matching. Stable Rust. Zero runtime cost.

### Why traits over 5 alternatives

| Approach | Existence | Signature | No build.rs | Stable | One-time |
|----------|-----------|-----------|-------------|--------|----------|
| build.rs scan | Partial | No | No | Yes | No |
| `use` import test | Yes | Partial | Yes | Yes | Yes |
| rustdoc JSON | Yes | Yes | Yes | **No** | No |
| `inventory`/`linkme` | Yes | No | Yes | Yes | No |
| **Trait `impl`** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### Four-layer enforcement model

| Layer | What | Mechanism | Coverage |
|-------|------|-----------|----------|
| L1 | Registry completeness | build.rs AllImplemented | 16,989 bindings |
| **L2** | **Function existence + signature** | **Trait `impl`** | **Per-contract** |
| L3 | Pre/postcondition assertions | `#[contract]` macro | Per-function |
| L4 | Reverse coverage | `pv lint --reverse` | Full crate scan |

### Implementation plan

1. **Phase 1**: `pv scaffold --trait` generates trait file per contract
2. **Phase 2**: Consumer crates `impl` the trait (Tier 1 kernels first)
3. **Phase 3**: `pv verify-trait` CI check validates impls exist

### Evidence

Approach validated by: SPARK/Ada spec/body separation (AdaCore §7.4),
Eiffel deferred features (Meyer 1988 §11.1), Kani `proof_for_contract`
(RFC 0009), Prusti `refine_trait_spec`, Creusot trait laws, Batuta
`ContractValidation` trait pattern. See arXiv:2410.01981 for the full
Rust verification landscape survey.

---

## 24. Deep Stack Integration

**Sub-spec**: [sub/deep-integration.md](sub/deep-integration.md)

Make contracts first-class citizens in the inference, profiling, and
quality pipelines — not just the build system.

### Four Gaps

1. **apr-cli roofline disconnected** — `apr serve plan` uses hardcoded
   formulas, not `roofline-model-v1.yaml` equations
2. **trueno ComputeBrick budget not contract-derived** — thresholds
   are manual, not loaded from contract YAML
3. **Tracing not contract-aware** — `ModelTracer` observes but doesn't
   verify postconditions against contract invariants
4. **pmat CB-1209 not exercised** — trait enforcement exists but isn't
   run in the `pmat comply check` pipeline end-to-end

### Three-Tier Integration

```
Tier 1 (Compile): YAML → build.rs → #[contract] → trait impl
Tier 2 (CI):      pv lint → verify-bindings → pmat CB-1200..1209
Tier 3 (Runtime): apr serve → roofline from YAML → BrickProfiler
                  → ContractTracingLayer → postcondition checks
```

Tier 3 is NEW. The runtime pipeline:
- `apr serve plan` derives TPS ceilings from `roofline-model-v1.yaml`
- `ComputeBrick` loads budget from `kernel-launch-budget-v1.yaml`
- `ContractTracingLayer` intercepts spans tagged with contract IDs
  and verifies postconditions against recorded values
- Violations emit SARIF-compatible diagnostics

### Evidence

Validated by: trueno BrickProfiler architecture (PAR-200),
apr-cli oracle compliance (PMAT-237), pmat CB-1200..1209 pipeline,
arXiv:2402.16363 (Roofline for LLM Inference), Creusot POPL 2026.

---

## 25. Full Enforcement Mandate

**Effective: 2026-03-27. Target: ALL paiml repos with bindings.**

### Goal

Every consuming repository MUST achieve **Grade A** (`pv score --min-score 0.90`)
with full enforcement: binding.yaml + trait tests + `pmat comply check` pass.

### Baseline (2026-03-28, measured — ghost bindings stripped)

| Repo | Real Bindings | Verified | build.rs | Traits | Codebase |
|------|--------------|----------|----------|--------|----------|
| aprender | 233 | 80 | YES | YES | B (0.78) |
| entrenar | 119 | 62 | YES | YES | C (0.69) |
| realizar | 58 | 38 | YES | YES | C (0.71) |
| trueno | 49 | 41 | YES | NO | C (0.66) |
| forjar | 13 | 13 | YES | YES | D (0.35) |
| depyler | 21 | ? | NO | YES | D (0.35) |
| bashrs | 22 | ? | NO | YES | D (0.35) |
| apr-model-qa-playbook | 9 | ? | NO | NO | C (0.65) |
| pmat | 4 | ? | NO | NO | D (0.35) |
| 14 sovereign stack repos | 0 | 0 | NO | NO | F (0.15) |

> **v2.2.0:** Previous version claimed "26/26 repos at Grade A (0.95)"
> based on 20,366 bindings. After stripping 28,206 ghost entries, the
> honest count is 540 real bindings. Only ~234 resolve in source code.

### Requirements per Repo

To achieve honest Grade A (codebase score >= 0.90), each repo must:

1. **Real bindings with `module_path`** — every binding must reference
   an actual Rust module path, not a generic function name
2. **`pv verify-bindings --crate-dir`** — bound functions must exist in source
3. **build.rs reads binding.yaml** — compile-time enforcement
4. **Trait tests on main** — `tests/contract_traits.rs` compiled in CI
5. **`pv lint` zero warnings** — all 7 gates pass

### Scoring Model (v2.2.0 — Option C)

Coverage is now **declared / resolved**, not **bound / all_equations**:

```
coverage = contracts_in_binding_that_exist / unique_contracts_in_binding
```

A repo that declares 49 bindings and all 49 reference real contracts
gets 100%. A repo with 0 bindings gets 0%. No ghost inflation possible.

| Repo | Real Bindings | Coverage | Codebase |
|------|--------------|----------|----------|
| aprender | 233 | 100% | **A (0.95)** |
| realizar | 58 | 100% | **A (0.96)** |
| apr-model-qa-playbook | 9 | 100% | **A (0.95)** |
| trueno | 49 | 100% | **B (0.85)** |
| entrenar | 119 | 100% | **C (0.75)** |

### Finding Missing Contracts with pmat

```bash
# 1. Full compliance audit — shows ALL provable-contracts enforcement gaps
pmat comply check

# Key checks:
#   CB-1208: Binding Existence — which bound functions don't exist in src/
#   CB-1209: Contract Trait Enforcement — are all 13 kernel traits implemented
#   CB-1210: Precondition Quality — are preconditions real or mass-generated

# 2. Find critical functions that LACK contracts
pmat query "forward" --faults --exclude-tests --limit 20
pmat query "backward" --faults --exclude-tests --limit 20
pmat query "kernel" --faults --exclude-tests --limit 20

# 3. Check specific enforcement checks
pmat comply check 2>&1 | grep -E 'CB-1202|CB-1203|CB-1208|CB-1209|CB-1210'

# 4. Ghost binding detection
pmat comply check 2>&1 | grep 'CB-1208'
# Shows: "52/136 bound fns not found (L3, 62% verified)"
# Named functions are your missing implementations

# 5. Check enforcement level
# L0 = ghost bindings (no enforcement)
# L1 = build.rs only (checks YAML, not code)
# L2 = trait tests only
# L3 = full (build.rs + traits)

# 6. Infra-score PV bonus
pmat infra-score -v 2>&1 | grep -A5 'Provable Contracts'
```

### What Each Check Means

| Check | What's Missing | How to Fix |
|-------|---------------|------------|
| CB-1208 lists function names | Functions in binding.yaml don't exist in src/ | Implement the function OR remove the ghost binding |
| CB-1209 < 13/13 | Missing contract trait impls | Add `tests/contract_traits.rs` with `impl XxxKernelV1 for YourStruct` |
| CB-1210 warns "0 postconditions" | Contracts have no postconditions | Add `postconditions:` to YAML equations |
| CB-1202 < 100% | Critical functions without contracts | Create YAML contracts for missing keywords |
| CB-1208 says "L0 paper-only" | binding.yaml exists but nothing reads it | Add build.rs enforcement OR trait tests |

### Quick Start: Add Missing Contracts to Your Repo

```bash
# Step 1: See what's missing
pmat comply check 2>&1 | grep '✗'

# Step 2: Generate trait stubs from existing contracts
cd ../provable-contracts
pv scaffold --trait contracts/softmax-kernel-v1.yaml

# Step 3: Add trait test to your repo (copy pattern from aprender)
cp ~/src/aprender/tests/contract_traits.rs tests/

# Step 4: Verify
cargo test --test contract_traits
pmat comply check  # Should show CB-1209: 13/13
```

### Gap Analysis

The codebase score is: `geometric_mean(Coverage, Binding, MeanScore, ProofDepth, Drift)`

Primary levers to reach 0.90:
- **Coverage**: binding.yaml coverage of contract equations (biggest gap for most repos)
- **Drift**: contracts must be committed alongside code changes (low drift)
- **MeanScore**: individual contract scores must average >= 0.86

### New Capabilities (v2.1.0)

| Feature | Section | CLI | Status |
|---------|---------|-----|--------|
| Roofline performance ceilings | §24 | `pv roofline` | Implemented |
| MQS scoring contract | §25 | `pv score mqs-scoring-v1.yaml` | Implemented |
| pmat self-enforcement | §25 | 4 contracts under `contracts/pmat/` | Implemented |
| Registry-aware scoring | §7 | Registries get full binding credit | Implemented |
| Zero-warning lint | §5 | `pv lint` → 0 errors, 0 warnings | Achieved |
| Preconditions on all equations | §3 | 527 equations with preconditions | Implemented |
| Lean theorem pointers | §14 | 527 equations with lean_theorem | Implemented |
| 871 Kani harnesses | §2 | All obligations covered | Implemented |

### Enforcement Tickets

| Ticket | Repo | Target | Work |
|--------|------|--------|------|
| PMAT-087 | aprender | A (0.90) | Add missing bindings for 13% uncovered equations |
| PMAT-088 | trueno | A (0.90) | Add trait tests + 79% more binding coverage |
| PMAT-089 | entrenar | A (0.90) | Increase binding coverage from 76% to 90% |
| PMAT-090 | realizar | A (0.90) | Increase binding coverage from 78% to 90% |
| PMAT-091 | forjar | A (0.90) | Increase binding coverage from 50% to 90% |
| PMAT-092 | bashrs | A (0.90) | Increase binding coverage from 55% to 90% |
| PMAT-093 | depyler | A (0.90) | Increase binding coverage from 64% to 90% |
| PMAT-094 | pmat (self) | A (0.90) | Increase binding coverage from 64% to 90% |
| PMAT-095 | apr-model-qa-playbook | A (0.90) | Binding coverage from 2% to 90% |

### Verification

```bash
# Verify A-score for a repo:
pv score contracts/ --binding contracts/<repo>/binding.yaml --min-score 0.90 --exit-code

# Full enforcement check:
pv lint contracts/ --binding contracts/<repo>/binding.yaml --strict
pv score contracts/ --binding contracts/<repo>/binding.yaml --min-score 0.90 --exit-code
```

---

## 26. Two-Tier Architecture and Compositional Contracts

### Two-Tier Contract Layout

Contracts are organized in two tiers:

```
contracts/
  # Tier 1: Generic kernel contracts (algorithm-level)
  softmax-kernel-v1.yaml          "How softmax works"
  matmul-kernel-v1.yaml           "How matmul works"
  attention-kernel-v1.yaml        "How attention works" (depends_on: softmax)
  inference-pipeline-v1.yaml      "How inference works" (depends_on: attention, rmsnorm, ...)
  roofline-model-v1.yaml          "Performance bound model"
  mqs-scoring-v1.yaml             "Model quality scoring"
  ...

  # Tier 2: Per-library contracts + bindings
  aprender/
    binding.yaml                  Maps generic contracts → aprender functions
    tokenizer-loading-v1.yaml     Library-specific contract
    training-loop-v1.yaml         Library-specific contract
  trueno/
    binding.yaml                  Maps generic contracts → trueno SIMD functions
    tiled-matmul-shader-v1.yaml   Library-specific contract
  entrenar/
    binding.yaml                  Maps generic contracts → entrenar GPU functions
    cuda-classify-training-v1.yaml
  realizar/
    binding.yaml                  Maps generic contracts → realizar orchestration
```

**Tier 1** contracts define the math — equations, invariants, proof obligations,
Kani harnesses. They are algorithm-specific, not library-specific. The same
`softmax-kernel-v1.yaml` governs every library that implements softmax.

**Tier 2** contracts are per-library. Each subdirectory contains:
1. `binding.yaml` — maps Tier 1 equations to the library's actual functions
2. Library-specific contracts — contracts that only apply to that library

### How Bindings Connect the Tiers

One generic contract serves multiple libraries through per-library bindings:

```
softmax-kernel-v1.yaml (the algorithm)
  ├── aprender/binding.yaml:  softmax → aprender::nn::functional::softmax
  ├── trueno/binding.yaml:    softmax → trueno::blis::softmax::softmax_avx2
  ├── entrenar/binding.yaml:  softmax → entrenar::kernels::softmax_forward
  └── realizar/binding.yaml:  softmax → realizar::gpu::softmax_wgsl
```

Each binding entry maps `(contract, equation)` to `(function, module_path, status)`.
The `bindings_for(stem)` method resolves this at runtime.

### The Composition Problem

Current contracts verify individual kernels in isolation. But the sovereign
stack is a **pipeline**: tokens flow through trueno's kernels, composed by
realizar's orchestrator, served by aprender's CLI. The question:

> If trueno's softmax is correct AND trueno's matmul is correct, is
> realizar's attention layer correct?

This is compositional verification. Three levels of composition exist:

#### Level 1: Intra-Contract Composition (SOLVED)

Contracts already use `depends_on` to declare dependencies:

```yaml
# attention-kernel-v1.yaml
metadata:
  depends_on: [softmax-kernel-v1]
equations:
  attention:
    formula: "Attention(Q,K,V) = softmax(QK^T/√d_k) · V"
```

The `pv graph` command visualizes this DAG. Kani harnesses use
`strategy: compositional` to stub verified sub-components.

#### Level 2: Cross-Contract Pipeline Composition (PARTIALLY SOLVED)

`inference-pipeline-v1.yaml` composes multiple kernels into a pipeline:

```yaml
metadata:
  depends_on:
    - softmax-kernel-v1
    - attention-kernel-v1
    - rmsnorm-kernel-v1
    - embedding-algebra-v1
equations:
  prefill_phase:
    formula: "H_L = layer_L(... layer_1(embed(tokens)))"
  decode_step:
    formula: "h_t = layer_L(... layer_1(embed(token_t), kv_cache))"
  layer_composition:
    formula: "h_{l+1} = h_l + sublayer(norm(h_l))"
```

This verifies the composition of algorithms but **not** the composition
of implementations across repos.

#### Level 3: Cross-Repo Pipeline Contracts (NOT YET IMPLEMENTED)

**This is what's missing.** When the call chain spans repos:

```
User request
  → aprender::serve::handler (apr-cli)
    → realizar::pipeline::forward_pass
      → trueno::blis::rmsnorm
      → trueno::blis::attention (→ trueno::blis::softmax + trueno::blis::matmul)
      → trueno::blis::swiglu
    → realizar::pipeline::sample
  → response
```

Each repo binds the same kernel contracts independently, but nobody
verifies that **trueno's softmax output format matches realizar's
attention input expectation**. The type system catches shape mismatches,
but invariant composition (e.g., "softmax output sums to 1.0, which
attention depends on for valid weight normalization") is not checked.

### Design: Cross-Repo Pipeline Bindings

To solve Level 3, we need **pipeline binding files** that declare
cross-repo data flow:

```yaml
# contracts/pipelines/inference-forward-v1.yaml
metadata:
  version: "1.0.0"
  description: "Cross-repo inference pipeline: trueno → realizar → aprender"
  pipeline: true

stages:
  - name: tokenize
    repo: aprender
    binding: aprender/binding.yaml
    contract: bpe-tokenization-v1
    equation: encode
    output_invariant: "token_ids ∈ [0, vocab_size)"

  - name: embed
    repo: aprender
    binding: aprender/binding.yaml
    contract: embedding-lookup-v1
    equation: embedding_lookup
    input_requires: "token_ids ∈ [0, vocab_size)"
    output_invariant: "shape = [seq_len, d_model], all finite"

  - name: transformer_block
    repo: trueno
    binding: trueno/binding.yaml
    repeat: num_layers
    stages:
      - contract: rmsnorm-kernel-v1
        equation: rmsnorm
        input_requires: "shape = [seq_len, d_model], all finite"
        output_invariant: "shape preserved, unit variance"
      - contract: attention-kernel-v1
        equation: attention
        input_requires: "normalized hidden states"
        output_invariant: "shape = [seq_len, d_model], all finite"
      - contract: swiglu-kernel-v1
        equation: swiglu
        output_invariant: "shape = [seq_len, d_model]"

  - name: decode
    repo: realizar
    binding: realizar/binding.yaml
    contract: sampling-algorithms-v1
    equation: sample
    input_requires: "logits shape = [vocab_size], all finite"
    output_invariant: "token_id ∈ [0, vocab_size)"

cross_boundary_obligations:
  - property: "Tokenizer output valid for embedder"
    from_stage: tokenize
    to_stage: embed
    formal: "∀t ∈ encode(text): 0 ≤ t < vocab_size"

  - property: "Embedding output valid for transformer"
    from_stage: embed
    to_stage: transformer_block
    formal: "shape(embed(tokens)) = [len(tokens), d_model] ∧ all_finite"

  - property: "Transformer output valid for sampler"
    from_stage: transformer_block
    to_stage: decode
    formal: "shape(H_L) = [seq_len, d_model] ∧ all_finite"
```

### Verification Strategy for Pipelines

```
                        [Compositional Kani]
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        [trueno stubs]  [realizar stubs]  [aprender stubs]
              │               │               │
        softmax_verified rmsnorm_verified tokenize_verified
        matmul_verified  attention_verified embed_verified
```

Each repo's Kani harnesses verify individual kernels. Pipeline
verification uses `strategy: compositional` — stub the verified
sub-components and verify only the composition glue:

1. **Input/output type compatibility** — output invariant of stage N
   implies input precondition of stage N+1
2. **Shape flow** — tensor dimensions are compatible across boundaries
3. **Numeric stability** — finite inputs produce finite outputs at
   every stage (no NaN propagation)

### Implementation Plan

| Phase | What | Tool |
|-------|------|------|
| P1 | `pv pipeline` CLI command | Parse pipeline YAML, validate cross-boundary obligations |
| P2 | Pipeline bindings | New YAML schema with `stages` + `cross_boundary_obligations` |
| P3 | Pipeline scoring | D6 dimension: fraction of pipeline stages with verified boundaries |
| P4 | Pipeline Kani | Compositional harnesses that stub verified stages |

### Sovereign Stack Pipeline Map (25 crates)

```
batuta (orchestrator, 196K LOC)
  ├── Analysis:    depyler, decy, bashrs, ruchy
  ├── Inference:   aprender → realizar → trueno
  ├── Training:    entrenar → trueno
  ├── Serving:     alimentar, renacer, pacha
  ├── Quality:     certeza, pmat, probar
  ├── Distributed: repartir, pepita
  ├── Viz:         presentar, trueno-viz
  └── Storage:     trueno-db, trueno-graph, trueno-rag, trueno-zram
```

The critical pipeline for contract verification:

```
tokens → aprender(embed) → trueno(rmsnorm,attn,ffn)×L → realizar(sample) → token
         ├── bpe-tokenization-v1    ├── rmsnorm-kernel-v1     ├── sampling-algorithms-v1
         ├── embedding-lookup-v1    ├── attention-kernel-v1
         └── special-tokens-v1     ├── swiglu-kernel-v1
                                   └── roofline-model-v1
```

### Sovereign Stack Enforcement Status (25 crates, measured 2026-03-27)

| Level | Crates | Description |
|-------|--------|-------------|
| **Full L3** | aprender, entrenar, realizar, ruchy (4/25) | build.rs + binding.yaml + trait tests |
| **L2** | trueno, bashrs (2/25) | Partial (build.rs or traits, not both) |
| **Paper only** | depyler, decy, presentar (3/25) | binding.yaml exists, no compile-time enforcement |
| **None** | 16/25 crates | No contracts at all (~502K LOC uncontracted) |

### Batuta Oracle: Sovereign Stack Component Map

```
batuta oracle "transformer inference pipeline"
  → entrenar (training, 85%)
  → realizar (serving, 85%)
  → trueno (SIMD backend, 80%)
  Integration pattern: training_to_inference
```

The oracle confirms the critical three-repo pipeline:
**trueno** (SIMD kernels) → **realizar** (orchestration) → **aprender** (serving).
This is the pipeline that needs cross-repo compositional contracts first.

### Theoretical Foundations

The compositional contract design draws from established formal methods:

**Assume-Guarantee Contracts.** Dardik & Kang (2025) show that
decomposing a system into components with assume-guarantee contracts
allows inferring local inductive invariants per component, whose
conjunction forms a global system invariant. This directly maps to our
pipeline model: each stage's `output_invariant` is the next stage's
`input_requires` — the assume-guarantee pair.

> "The conjunction of all local invariants becomes an inductive
> invariant for the entire system." — arXiv:2509.06250

**Kani Function Contracts.** Kani's `#[kani::requires]` /
`#[kani::ensures]` / `#[kani::modifies]` with `stub_verified`
attribute (RFC 0009, stable since Kani 0.33.0) enables modular
verification: prove a function satisfies its contract, then replace
calls with contract stubs in downstream harnesses. This is exactly
the compositional strategy for cross-repo pipeline verification.

> "Contracts enable divide-and-conquer verification — prove a method
> satisfies its contract, then replace calls by permitted behaviors."
> — Kani Function Contracts RFC (2024)

**Roofline Performance Bounds.** Yuan et al. (2024) apply the
roofline model to LLM inference, showing decode is memory-bound and
prefill is compute-bound. Our `roofline-model-v1.yaml` + `pv roofline`
CLI implement these equations as contract-derived performance ceilings.

> "During decode, all computations are memory-bound, resulting in
> performance significantly below computational capacity."
> — arXiv:2402.16363

**Rust Verification Landscape.** Le Blanc & Lam (2024) survey
Rust verification tools including Kani (bounded model checking),
Creusot (deductive verification with prophecies), and Flux
(refinement types). Our stack uses Kani for L4 and Lean 4 for L5.

> "Bounded model checking is a good choice for Rust verification."
> — arXiv:2410.01981

**Compositional Neural Network Verification.** Duong et al. (2025)
apply assume-guarantee reasoning to neural network verification,
decomposing networks into sub-components verified independently.
The same principle applies to our transformer block pipeline: verify
each kernel (softmax, matmul, rmsnorm) independently, then compose.

### References (Section 26)

- Dardik & Kang (2025). "Compositional Inductive Invariant Inference
  via Assume-Guarantee Reasoning." arXiv:2509.06250
- Incer et al. (2023). "Pacti: Scaling Assume-Guarantee Reasoning
  for System Analysis and Design." arXiv:2303.17751
- Yuan et al. (2024). "LLM Inference Unveiled: Survey and Roofline
  Model Insights." arXiv:2402.16363
- Le Blanc & Lam (2024). "Surveying the Rust Verification Landscape."
  arXiv:2410.01981
- Matsushita et al. (2024). "Lessons Learned from Verifying the Rust
  Standard Library." arXiv:2510.01072
- Kani Team (2024). "Function Contracts for Kani." RFC 0009.
  model-checking.github.io/kani
- Denis, Jourdan & Marché (2022). "Creusot: A Foundry for the
  Deductive Verification of Rust Programs." ICFEM 2022.
- Duong et al. (2025). "Compositional Neural Network Verification
  via Assume-Guarantee Reasoning."
- Williams et al. (2009). "Roofline: An Insightful Visual Performance
  Model for Multicore Architectures." CACM 52(4).

---

## 27. The One Way

### The Problem: Three Mechanisms, One Job

Contract enforcement currently uses three separate mechanisms:

| Mechanism | Where it lives | What it checks | Who must set it up |
|-----------|---------------|----------------|-------------------|
| `build.rs` | Consumer's build script | Binding.yaml parseable, policy satisfied | Each repo manually |
| Trait tests | `tests/contract_traits.rs` | Function signatures exist and compile | Each repo manually |
| `pv lint` | CI or local | YAML structure, score thresholds | provable-contracts |

Result: 6/26 repos have all three. 14/26 have none. The mechanisms
are redundant (all check "do the declared bindings exist?") and their
per-repo setup creates waste.

### What Others Do: One Mechanism

**Eiffel (Meyer, 1988).** One mechanism: `require`/`ensure`/`invariant`
clauses in the language syntax. The compiler handles everything.
Enforcement is a compiler flag (`-Xassertions`), not a per-file setup.
Meyer's key insight: **seamlessness** — one notation, one tool, one
place to look.

> "The Eiffel experience shows that, to be effective, contractual
> mechanisms must be built right into the notation, the tools, and the
> development culture. Retrofitting them is not enough."
> — Meyer, *Object-Oriented Software Construction* (1997)

**Haskell / LiquidHaskell.** One mechanism: refinement types in
function signatures. `{-@ foo :: {v:Int | v > 0} -> {v:Int | v > 0} @-}`
The compiler (via SMT solver) verifies at compile time. No runtime
checks, no separate test files, no build scripts.

**Rust Nightly (2025+).** One mechanism arriving in the language itself:

```rust
#![feature(contracts)]

#[core::contracts::requires(x > 0)]
#[core::contracts::ensures(|ret| *ret > 0)]
fn foo(x: i32) -> i32 { x }
```

Compile with `-Zcontract-checks=on` → runtime panic on violation.
Verified working on `rustc 1.94.0-nightly (2026-01-02)`.

**Toyota Production System.** One piece flow. Every operator is an
inspector. Quality is built in at the station, not bolted on at the
end. Translated to software: the contract IS the code, not a separate
YAML file that a separate tool checks.

### Falsification (2026-03-27)

The "One Way" was falsified before implementation:

| # | Claim | Reality | Severity |
|---|-------|---------|----------|
| F1 | `pv codegen` generates enforcement | Yes, generates macro stubs | OK |
| F2 | Preconditions are meaningful | All 427 are `!input.is_empty()` (generic) | **HIGH** |
| F3 | Repos use the output | **0/26** include or call the macros | **CRITICAL** |
| F4 | Postconditions work | **0** postconditions in any contract | **CRITICAL** |
| F5 | Macros bind to real functions | Hardcoded `input` var, not real signatures | **HIGH** |

**Root cause:** The mass-generation in PMAT-082 added `!input.is_empty()`
to every equation as a placeholder. Real preconditions require
domain-specific assertions derived from the equation's `domain` and
`invariants` fields. Postconditions were never populated in the YAML.

### The One Way for provable-contracts

**One mechanism: `pv codegen --binding` reads both the contract YAML
and the binding.yaml to generate function-specific `debug_assert!()`
calls that reference real parameter names from the binding signatures.**

The flow:

```
YAML contract (equations + domain + invariants)
  + binding.yaml (function signatures + module paths)
  ↓
pv codegen --binding contracts/<repo>/binding.yaml
  ↓
generated_contracts.rs (debug_assert! with real param names)
  ↓
Consumer crate: mod generated_contracts; // one line
  ↓
contract_pre_softmax!(x);  // at function entry
contract_post_softmax!(result);  // before return
```

Phase 0 generates `debug_assert!()`. When `#[core::contracts::requires]`
stabilizes, `pv codegen` switches output format — the YAML and binding
don't change.

### What Must Be Fixed (PMAT-103)

1. **Binding-aware codegen**: Read function signatures from binding.yaml
   to generate macros with correct parameter names
2. **Domain-derived preconditions**: Parse equation `domain` field to
   generate meaningful assertions (e.g., `x.len() > 0 && x.iter().all(|v| v.is_finite())`)
3. **Postconditions from invariants**: Generate postcondition macros
   from equation `invariants` (e.g., `(result.iter().sum::<f32>() - 1.0).abs() < 1e-6`)
4. **Include mechanism**: Generated file must be `include!`d or `mod`d
   by the consumer crate — document the one-line setup

### Transition Plan

| Phase | When | What |
|-------|------|------|
| **Phase 0 (now)** | Stable Rust | `pv codegen --binding` generates `debug_assert!()` with real parameter names from binding signatures. |
| **Phase 1 (nightly)** | Rust nightly | `pv codegen --contracts` generates `#[core::contracts::requires]` / `#[core::contracts::ensures]`. Requires `#![feature(contracts)]`. |
| **Phase 2 (stable)** | Contracts RFC stabilized | `pv codegen` generates stable contract attributes. build.rs and trait tests become dead code. Delete them. |

### What Dies

When Phase 2 lands:

- **build.rs binding verification** — dead. The compiler checks contracts directly.
- **Trait tests** — dead. Contract attributes on real functions replace trait impls on test structs.
- **`#[contract]` proc macro** — dead. Replaced by `#[core::contracts::requires]`.
- **Per-repo setup** — dead. `pv codegen` is the only tool needed.

### What Survives

- **YAML contracts** — the source of truth. Equations, obligations, invariants.
- **binding.yaml** — maps equations to functions. `pv codegen` reads this to know where to emit attributes.
- **`pv codegen`** — the one tool that generates enforcement code.
- **`pv lint`** — validates YAML structure (not enforcement).
- **`pv score`** — measures contract quality (not enforcement).
- **Kani** — bounded model checking. Orthogonal to runtime contracts.
- **Lean 4** — theorem proving. Orthogonal to runtime contracts.

### The Eiffel Parallel

```
Eiffel                          provable-contracts (Phase 2)
─────                           ──────────────────────────────
require clause                  #[core::contracts::requires(...)]
ensure clause                   #[core::contracts::ensures(...)]
class invariant                 impl core::marker::Invariant
-Xassertions:all               -Zcontract-checks=on
ONE language, ONE compiler      ONE YAML, ONE codegen, ONE compiler
```

### Why This Matters

The current three-mechanism approach is **muda** (waste):

- **Over-processing**: Three tools checking the same property.
- **Inventory**: build.rs files, trait test files, proc macro crate — all sitting in repos, maintained, but redundant.
- **Motion**: Developer must set up build.rs + traits + binding in each repo.
- **Defects**: 14/26 repos have zero enforcement because the setup wasn't done.

The Toyota Way says: eliminate the waste. Build quality in at the source.
The source is the YAML contract. The enforcement is the compiler.
Everything in between is muda.

### References (Section 27)

- Meyer (1997). *Object-Oriented Software Construction.* Prentice Hall.
  Chapter 11: Design by Contract.
- Meyer (1992). "Applying Design by Contract." IEEE Computer 25(10).
- Vazou et al. (2014). "LiquidHaskell: Experience with Refinement
  Types in the Real World." ICFP 2014.
- Rust Project Goals (2025). "Instrument the Rust Standard Library
  with Safety Contracts." rust-lang.github.io
- Rust Tracking Issue #128044. "Contracts." github.com/rust-lang/rust
- Ohno (1988). *Toyota Production System: Beyond Large-Scale
  Production.* Productivity Press.
- Liker (2004). *The Toyota Way.* McGraw-Hill. Principle 7:
  Use visual control so no problems are hidden.

---

## 28. Correctness + Completeness

### The Problem

Grade A (v2.2.0) measures **correctness** — "did you keep the promises
you made?" A repo with 4 bindings that all resolve gets A. A repo with
400 bindings where 390 resolve gets B. The one with 4 is "better" by
the metric but covers almost nothing.

Nobody asks: **what functions SHOULD have contracts but don't?**

This is the distinction between:
- **Correctness**: the contracts you wrote are right
- **Completeness**: you wrote contracts for everything that needs them

Both are required. One without the other is insufficient.

### Meyer's Insight

Meyer never mandated "every routine must have a contract." DbC is
prescriptive about HOW to write contracts, not WHEN. But:

> "One cannot expect large-scale reuse without a precise documentation
> of what every component expects (precondition), what it guarantees
> in return (postcondition) and what general conditions it maintains
> (invariant)."
> — Meyer, *Object-Oriented Software Construction* (1997), Ch. 11

The **class invariant** is Meyer's completeness mechanism — it applies
to ALL exported features of a class, not just the ones the developer
chose to annotate. If you have a class invariant, every routine
implicitly has at least that contract.

Our analog: a repo's `binding.yaml` declares which functions have
contracts (correctness). The **completeness gap** is the set of
`pub fn` declarations in source code that have no binding at all.

### Research Support

**VeriEquivBench (arXiv:2510.06296, 2025)** introduces an "equivalence
score" measuring bidirectional implication between code and spec —
both soundness (spec implies code) AND completeness (code implies spec).

> "Without [a completeness metric], there is no way to guarantee that
> verified code truly aligns with its intended behaviour."

**VERINA (arXiv:2505.23135, 2025)** measures "soundness AND
completeness" of specifications against ground truth, finding the best
model achieves only 52.3% specification completeness.

**CLEVER (arXiv:2505.13938, 2025)** notes:

> "Automatically generated specifications can be incomplete or leaky."

**Coverage metrics for formal verification (Springer, 2003):**

> "Even when a system is proven correct, there is still a question of
> how complete the specification is, and whether it really covers all
> the behaviors of the system."

### The Two Dimensions

```
┌────────────────────────────┬──────────────────────────────────────┐
│     CORRECTNESS            │         COMPLETENESS                 │
│  "contracts are right"     │  "everything has a contract"         │
├────────────────────────────┼──────────────────────────────────────┤
│ Does code match contract?  │ Does every significant fn have one?  │
├────────────────────────────┼──────────────────────────────────────┤
│ Measured by:               │ Measured by:                         │
│ - CD1: declared/resolved   │ - CD2: bound_pub_fns / total_pub_fns│
│ - CB-1203: macros present  │ - CB-1211: pub fn coverage gap (NEW)│
│ - CB-1208: bindings exist  │ - PV-05: completeness gate (NEW)    │
│ - PV-01: pv lint passes    │                                     │
├────────────────────────────┼──────────────────────────────────────┤
│ Eiffel parallel:           │ Eiffel parallel:                     │
│ require/ensure on routines │ class invariant on ALL features      │
├────────────────────────────┼──────────────────────────────────────┤
│ Tool:                      │ Tool:                                │
│ pv score --binding         │ pv score --binding --crate-dir (NEW) │
│ pv verify-bindings         │ pv infer (existing, not in score)    │
│                            │ reverse_coverage.rs (existing)       │
└────────────────────────────┴──────────────────────────────────────┘
```

### How pmat Already Integrates (and the Gap)

pmat has three systems that check provable-contracts:

**TDG (per-file quality):** Computes `provability_factor` from
contract presence — files with contracts get lower tech-debt scores.
But TDG does not flag files WITHOUT contracts. The provability factor
only rewards; it doesn't penalize absence.

**pmat comply (CB-1200..1210):**
- CB-1200: Contracts validate
- CB-1203: Contract-bound fns have `#[contract]` macros
- CB-1208: Bindings resolve in source (ghost detection)
- CB-1209: Trait enforcement
- CB-1210: Precondition quality

All five check the **contracts side** — are the contracts correct?
None check the **code side** — what code lacks contracts?

**pmat infra-score (PV-01..04):**
- PV-01: `pv lint` passes (3pts)
- PV-02: `pv score >= 0.5` (3pts)
- PV-03: Proof level L2+ (2pts)
- PV-04: contracts/ exists (2pts)

All four check contract quality. None check code coverage.

### Proposed Changes

#### provable-contracts: CD2 completeness dimension

```rust
// In score_codebase_full(), when crate_dir is provided:
let completeness = if let Some(crate_dir) = crate_dir {
    let report = reverse_coverage(crate_dir, binding);
    report.bound_count as f64 / report.total_pub_fns as f64
} else {
    binding_completeness  // fallback: current behavior
};
```

Codebase score becomes:

```
CD1: Correctness (30%) — declared/resolved [existing]
CD2: Completeness (20%) — bound_pub_fns / significant_pub_fns [NEW]
CD3: MeanScore (20%) — average contract quality
CD4: ProofDepth (15%) — weighted proof levels
CD5: Drift (15%) — freshness
```

"Significant pub fns" = pub fns excluding:
- Trait impls of std traits (Display, From, Default, Debug)
- Functions in `#[cfg(test)]` modules
- Functions < 5 lines (trivial getters/setters)

This reuses `reverse_coverage.rs::scan_pub_fns` which already exists.

#### paiml-mcp-agent-toolkit: CB-1211 completeness check

New check in `check_pv_enforcement.rs`:

```
CB-1211: Contract Completeness
  Scans source for significant pub fns
  Cross-references against binding.yaml
  Reports: "X/Y significant pub fns have contract bindings (Z%)"
  PASS if >= 50%, WARN if >= 20%, FAIL if < 20%
```

#### paiml-mcp-agent-toolkit: PV-05 infra-score check

New check in `provable_contracts.rs` scorer:

```
PV-05 (2pts): Contract completeness >= 50%
  Runs reverse coverage scan
  Max PV bonus increases from 10 to 12 points
```

### Grade Definition (v2.3.0)

```
Grade A: correctness >= 90% AND completeness >= 50%
Grade B: correctness >= 80% AND completeness >= 20%
Grade C: correctness >= 70% OR completeness >= 10%
Grade F: correctness < 50% OR completeness = 0%
```

The AND in Grade A is critical. A repo cannot achieve A by being
perfectly correct on 4 functions while ignoring the other 496.

### Self-Falsification

| # | Concern | Mitigation |
|---|---------|------------|
| F1 | `pub fn` count includes trivial impls (Display, From) | Filter std trait impls, test fns, <5 line fns |
| F2 | 50% threshold is arbitrary | Configurable via `--min-completeness` |
| F3 | Requires `--crate-dir` which is optional | Spec mandates it for honest scoring |
| F4 | Gaming by making functions private | Private fns are inaccessible — not a real vector |
| F5 | Large repos with many helpers penalized unfairly | Filter to "significant" fns only |

### CLI Changes

```bash
# Current (correctness only):
pv score contracts/ --binding contracts/aprender/binding.yaml

# New (correctness + completeness):
pv score contracts/ --binding contracts/aprender/binding.yaml \
    --crate-dir ~/src/aprender

# pmat comply (both dimensions):
cd ~/src/aprender
pmat comply check
# CB-1208: 80/233 bindings verified (correctness)
# CB-1211: 233/500 significant pub fns bound (47% completeness) ← NEW
```

### References (Section 28)

- Meyer (1997). *Object-Oriented Software Construction.* Ch. 11:
  Design by Contract. Class invariants as completeness mechanism.
- Meyer (1992). "Applying Design by Contract." IEEE Computer 25(10).
- VERINA (arXiv:2505.23135, 2025). Specification soundness AND
  completeness benchmark.
- VeriEquivBench (arXiv:2510.06296, 2025). Bidirectional equivalence
  score — code implies spec AND spec implies code.
- CLEVER (arXiv:2505.13938, 2025). Human-curated specs for
  completeness; auto-generated specs are "incomplete or leaky."
- Coverage Metrics for Formal Verification (Springer, 2003). "Even
  when proven correct, how complete is the specification?"
