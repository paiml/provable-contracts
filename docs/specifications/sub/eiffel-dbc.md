# Sub-spec: Eiffel Design by Contract Extensions

**Parent:** [pv-spec.md](../pv-spec.md) Section 3

---

## 1. Motivation

Bertrand Meyer's Design by Contract (DbC) framework, introduced in
*Object-Oriented Software Construction* (1988, 2nd ed. 1997) and
implemented in the Eiffel language, defines a richer contract vocabulary
than our current proof obligation types capture. Our 19 obligation types
express *properties of functions* (invariant, monotonicity, bound, etc.)
but do not model the **caller/callee contract pair** that is central to
Meyer's framework.

This sub-spec defines seven new obligation types derived from the Eiffel
DbC tradition. These extend the existing type system without replacing
it — every current obligation type remains valid.

### What Already Exists

The codebase has already adopted Eiffel DbC concepts at two levels:

1. **Equation-level pre/postconditions.** The `Equation` struct has
   `preconditions: Vec<String>` and `postconditions: Vec<String>` fields
   containing Rust expressions that compile to `debug_assert!()` via
   `build.rs` codegen (see [escape-proof-enforcement.md](escape-proof-enforcement.md)).

2. **Proc macro enforcement.** Three attribute macros implement Meyer's
   core keywords directly:
   - `#[requires(pred)]` → Eiffel `require` → `debug_assert!` at entry
   - `#[ensures(pred)]` → Eiffel `ensure` → `debug_assert!` before return
   - `#[invariant(pred)]` → Eiffel `invariant` → assert pre + post

3. **`codegen.rs` module.** Generates `contract_pre_*!()` and
   `contract_post_*!()` assertion macros from YAML equations.

4. **Gate 5 (Enforce).** `pv lint` checks that equations have
   `preconditions`, `postconditions`, and `lean_theorem` fields.

### The Remaining Gap

The existing implementation puts pre/postconditions on **equations**
(implementation-facing, Rust expressions, `debug_assert!` enforcement).
This sub-spec proposes them as **proof obligation types** (specification-
facing, formal predicates, Kani/Lean verifiable). These are
complementary layers:

| Layer | Where | What | Verified by |
|---|---|---|---|
| Equation pre/post | `equations.<name>.preconditions` | Rust expressions | `debug_assert!` (runtime, debug) |
| Obligation pre/post | `proof_obligations[].type: precondition` | Formal predicates | Kani (L4), Lean (L5), probar (L3) |

The equation-level pre/postconditions are *enforcement*. The obligation-
level pre/postconditions are *specification*. Meyer's framework requires
both: the formal spec (what we prove) and the runtime check (what we
enforce). The five remaining types (`frame`, `loop_invariant`,
`loop_variant`, `old_state`, `subcontract`) have no equivalent at
either level today.

---

## 2. New Obligation Types

### 2.1. `precondition`

**Eiffel keyword:** `require`

**Pattern:** `P(input) — conditions that must hold before the kernel executes`

A precondition defines the caller's responsibility. If the caller
violates the precondition, the kernel's behavior is undefined. This is
distinct from a `bound` (which constrains the *output*) and from an
`invariant` (which asserts a property of the *function itself*).

```yaml
proof_obligations:
  - type: precondition
    property: "Input vector is finite and non-empty"
    formal: "∀i: ¬isNaN(x_i) ∧ ¬isInf(x_i) ∧ len(x) > 0"
    if_fails: "Kernel may produce NaN or panic on empty slice"
```

**Codegen implications:**
- probar: generate `proptest` that asserts the kernel *does* panic or
  return `Err` when the precondition is violated (negative testing)
- Kani: assume the precondition (`kani::assume`) and verify the
  postcondition
- Lean: precondition becomes a hypothesis in the theorem statement

**Relationship to `equations.<name>.preconditions`:** The equation-
level field contains *Rust expressions* for runtime `debug_assert!`.
The obligation-level `precondition` type contains a *formal predicate*
for static verification. They encode the same intent at different
abstraction levels. `pv generate` should emit both: the formal
predicate for Kani/Lean, and a Rust translation for `debug_assert!`.

**Relationship to existing types:** A `bound` obligation like
`0 <= f(x)_i <= 1` is implicitly a postcondition. Adding explicit
`precondition` makes the conditional structure visible:
`require P(x) ensure 0 <= f(x)_i <= 1`.

### 2.2. `postcondition`

**Eiffel keyword:** `ensure`

**Pattern:** `P(input) → Q(output) — what the kernel guarantees if preconditions hold`

A postcondition defines the kernel's responsibility. Unlike our current
`invariant` type (which asserts `∀x: P(f(x))`), a postcondition is
explicitly conditional on the precondition holding.

```yaml
proof_obligations:
  - type: postcondition
    property: "Output sums to 1.0 within tolerance"
    formal: "|Σ softmax(x)_i - 1.0| < ε"
    tolerance: 1.0e-6
    requires: "SM-PRE-001"   # Links to a precondition obligation
    if_fails: "Normalization numerically unstable for this input"
```

**The `requires` field:** A postcondition MAY reference one or more
precondition obligation IDs. This makes the `require → ensure` pair
explicit and machine-traceable.

**Relationship to `equations.<name>.postconditions`:** Same layering
as preconditions. The equation-level field holds Rust expressions for
`debug_assert!` (enforced by `#[ensures]` / `#[contract]` macros). The
obligation-level `postcondition` type holds the formal predicate for
Kani/Lean verification. Both are generated from the same contract.

**Codegen implications:**
- probar: generate test that first asserts precondition, then checks
  postcondition
- Kani: `kani::assume(precondition); kani::assert(postcondition);`
- Lean: `theorem softmax_sums_to_one (h : precondition x) : postcondition (softmax x)`

### 2.3. `frame`

**Eiffel keyword:** `only` (Eiffel 2005+ frame specification)

**Pattern:** `modifies(S) ∧ preserves(T \ S) — what state the kernel may change`

A frame condition specifies what the kernel is *allowed* to modify.
Everything not listed is implicitly preserved. This is critical for
kernels that operate on shared buffers (KV cache, weight tensors,
activation buffers).

```yaml
proof_obligations:
  - type: frame
    property: "Only output buffer is modified; input and weights unchanged"
    formal: "modifies(output) ∧ preserves(input, weights)"
    if_fails: "Kernel corrupts input buffer or weight tensor"
```

**Why this matters for ML kernels:**
- In-place operations (LayerNorm, RMSNorm) must not corrupt input
- KV cache append must not modify existing entries
- Quantization must not modify the original weights
- SIMD kernels with overlapping loads must not write beyond bounds

**Codegen implications:**
- probar: snapshot input/weights before call, assert equality after
- Kani: clone inputs, run kernel, assert originals unchanged
- Lean: frame condition as a conjunction in the postcondition

### 2.4. `loop_invariant`

**Eiffel keyword:** `invariant` (within `from...until...loop...end`)

**Pattern:** `∀ iteration i: P(state_i) — property maintained across iterations`

Distinct from our top-level `invariant` (a property of the function),
a loop invariant is a property of the *iterative state* at each step.
Relevant for iterative kernels: AdamW, LBFGS, CMA-ES, PageRank,
online softmax.

```yaml
proof_obligations:
  - type: loop_invariant
    property: "Running max is true max of elements seen so far"
    formal: "∀k ≤ i: running_max ≥ x_k"
    applies_to: "online_softmax.find_max"
    if_fails: "Online softmax numerically unstable — max tracking diverges"
```

**The `applies_to` field:** References a specific phase from the
contract's `kernel_structure.phases[]`, anchoring the invariant to a
concrete loop in the implementation.

**Codegen implications:**
- probar: instrument the loop body, assert invariant after each iteration
  for a bounded number of steps
- Kani: unroll loop to `bound`, assert invariant at each unrolling step
- Lean: induction proof over loop iterations

### 2.5. `loop_variant`

**Eiffel keyword:** `variant`

**Pattern:** `V(state_i) ∈ ℕ ∧ V(state_{i+1}) < V(state_i) — strictly decreasing natural number expression`

A loop variant is a witness function that proves termination. It must
be a non-negative integer expression that strictly decreases with each
iteration. This is more precise than our existing `termination` type,
which asserts termination without providing a witness.

```yaml
proof_obligations:
  - type: loop_variant
    property: "Remaining elements decreases each iteration"
    formal: "V(state) = n - i, V ≥ 0, V decreases"
    applies_to: "online_softmax.accumulate"
    if_fails: "Infinite loop — iteration counter not advancing"
```

**Relationship to `termination`:** A `loop_variant` *implies*
`termination` but is strictly stronger — it provides the proof witness.
Contracts MAY use `termination` when no clean variant expression exists
(e.g., convergence-based loops where termination depends on input
properties), but SHOULD prefer `loop_variant` when one exists.

**Codegen implications:**
- probar: compute variant before and after each iteration, assert
  strictly decreasing and non-negative
- Kani: assert `variant_after < variant_before` and `variant_after >= 0`
  at each unrolling step
- Lean: well-founded recursion or `Nat.lt_wfRel`

### 2.6. `old_state`

**Eiffel keyword:** `old` (expression)

**Pattern:** `Q(old(state), new(state)) — postcondition referencing pre-call values`

An old-state obligation relates the output to the input's *original*
value. This is more expressive than `conservation` (which asserts
`Q(before) = Q(after)` for a single quantity). Old-state obligations
can express arbitrary relationships between pre-state and post-state.

```yaml
proof_obligations:
  - type: old_state
    property: "KV cache length increases by exactly seq_len"
    formal: "new(cache.len) = old(cache.len) + seq_len"
    if_fails: "Cache append wrote wrong number of entries"
```

**Relationship to `conservation`:** Conservation is a special case of
old-state where the relationship is equality of a derived quantity.
Old-state is more general:

| Obligation | Expressible? |
|---|---|
| `conservation`: `sum(before) = sum(after)` | `old_state` can express this |
| `old_state`: `len(after) = len(before) + n` | `conservation` cannot |
| `old_state`: `∀i < old(len): after[i] = before[i]` | `conservation` cannot |

**Codegen implications:**
- probar: snapshot state before call, compare with state after
- Kani: clone state, run kernel, assert relationship between clone and result
- Lean: `old` values become universally quantified variables in theorem

### 2.7. `subcontract`

**Eiffel keyword:** Inheritance-based subcontracting rules

**Pattern:** `weaken(pre) ∧ strengthen(post) — contract refinement under substitution`

A subcontract obligation asserts that a derived contract is a valid
behavioral subtype of its parent. This enforces Meyer's subcontracting
rules, which align with the Liskov Substitution Principle:

1. Preconditions may only be **weakened** (OR'd with parent)
2. Postconditions may only be **strengthened** (AND'd with parent)
3. Invariants are **accumulated** (AND'd with parent)

```yaml
proof_obligations:
  - type: subcontract
    property: "GQA attention is a valid refinement of MHA attention"
    formal: "pre(MHA) → pre(GQA) ∧ post(GQA) → post(MHA)"
    parent_contract: "attention-kernel-v1"
    if_fails: "GQA cannot be substituted for MHA — contract violation"
```

**The `parent_contract` field:** References the contract stem that this
contract refines. Must be present in `metadata.depends_on`.

**Relationship to `depends_on`:** The existing `depends_on` field
declares a dependency. A `subcontract` obligation goes further: it
asserts that the dependency is a *behavioral subtyping* relationship,
not just a compositional one.

**Codegen implications:**
- probar: load both contracts, generate test that any input satisfying
  the parent's precondition also satisfies the child's, and the child's
  postcondition implies the parent's
- Kani: verify the weakening/strengthening relationship symbolically
- Lean: subtyping proof via implication chains

---

## 3. Schema Changes

### 3.1. Existing Pre/Post Infrastructure

The `Equation` struct already has pre/postcondition fields (added in
the escape-proof enforcement work):

```rust
pub struct Equation {
    pub formula: String,
    pub preconditions: Vec<String>,    // Rust exprs → debug_assert!
    pub postconditions: Vec<String>,   // Rust exprs → debug_assert!
    pub lean_theorem: Option<String>,  // Lean 4 proof reference
    // ...
}
```

These are *implementation-level* assertions (Rust code, runtime checks).
The new `ObligationType` variants add *specification-level* predicates
(formal logic, static verification via Kani/Lean).

### 3.2. New `ObligationType` Variants

Add to the `ObligationType` enum in `crates/provable-contracts/src/schema/types.rs`:

```rust
pub enum ObligationType {
    // ... existing 19 variants ...
    Precondition,
    Postcondition,
    Frame,
    LoopInvariant,
    LoopVariant,
    OldState,
    Subcontract,
}
```

### 3.3. New Optional Fields on `ProofObligation`

```yaml
proof_obligations:
  - type: precondition | postcondition | frame | ...
    property: "..."           # REQUIRED (existing)
    formal: "..."             # Formal predicate (existing)
    tolerance: 1.0e-6         # Numerical tolerance (existing)
    # --- new fields ---
    requires: "OB-ID"         # Postcondition: links to precondition
    applies_to: "phase_name"  # Loop invariant/variant: kernel_structure phase
    parent_contract: "stem"   # Subcontract: contract being refined
```

All new fields are optional and only meaningful for their respective
obligation types. Validation (`pv validate`) will reject:
- `requires` on any type other than `postcondition`
- `applies_to` on any type other than `loop_invariant` or `loop_variant`
- `parent_contract` on any type other than `subcontract`
- `parent_contract` values not present in `metadata.depends_on`

### 3.4. Backward Compatibility

All new types and fields are additive. Existing contracts are
unaffected. The `ObligationType` enum already uses `#[serde(rename)]`
for `state_machine`; new multi-word types follow the same pattern:

```rust
#[serde(rename = "loop_invariant")]
LoopInvariant,
#[serde(rename = "loop_variant")]
LoopVariant,
#[serde(rename = "old_state")]
OldState,
```

---

## 4. Proof Obligation Type Reference (Updated)

Extends [schema.md Section 2](schema.md#2-proof-obligation-type-reference)
with the Eiffel DbC types:

| Type | Pattern | Example |
|---|---|---|
| `precondition` | P(input) must hold before call | input is finite, non-empty |
| `postcondition` | P(in) → Q(out) conditional guarantee | given finite input, output sums to 1 |
| `frame` | modifies(S), preserves(T\S) | only output buffer written |
| `loop_invariant` | ∀ iter i: P(state_i) | running max tracks true max |
| `loop_variant` | V(state) ∈ N, strictly decreasing | remaining elements = n - i |
| `old_state` | Q(old(state), new(state)) | cache.len grows by seq_len |
| `subcontract` | weaken(pre) ∧ strengthen(post) | GQA refines MHA |

---

## 5. Codegen Strategy

**Integration with existing pipeline.** The codegen for new obligation
types plugs into the existing six-stage escape-proof enforcement
pipeline (see [escape-proof-enforcement.md](escape-proof-enforcement.md)):
- Stage D (`build.rs` codegen) already generates `debug_assert!` from
  equation-level `preconditions`/`postconditions`
- Stage E (`#[contract]` macro) already reads `CONTRACT_*_PRE_*` and
  `CONTRACT_*_POST_*` env vars and injects assertions
- The new obligation types extend **Stages C (validation), D (codegen),
  and F (test execution)** with generators for the five types not yet
  covered: `frame`, `loop_invariant`, `loop_variant`, `old_state`,
  `subcontract`

**Integration with Lean-Kani composition.** The `stub_float` bridge
(see [lean-kani-composition.md](lean-kani-composition.md)) applies to
the new types: Lean proves the obligation over ℝ, Kani verifies the
Rust code over f32, and `#[kani::stub_verified]` connects them. For
example, a `loop_invariant` can be proved by induction in Lean and
verified for bounded iterations in Kani.

### 5.1. Probar (Property Tests)

| Type | Test Strategy |
|---|---|
| `precondition` | Negative test: violate precondition, assert panic/Err |
| `postcondition` | Conditional test: assume pre, assert post |
| `frame` | Snapshot-compare: clone inputs, run kernel, assert inputs unchanged |
| `loop_invariant` | Instrumented loop: assert property at each iteration |
| `loop_variant` | Instrumented loop: assert decreasing and non-negative |
| `old_state` | Snapshot-compare: clone state, run, assert relationship |
| `subcontract` | Cross-contract: load parent, verify pre weakening + post strengthening |

### 5.2. Kani (Bounded Model Checking)

| Type | Kani Pattern |
|---|---|
| `precondition` | `kani::assume(pre); /* verify postcondition */` |
| `postcondition` | `kani::assume(pre); kani::assert(post);` |
| `frame` | Clone inputs; run kernel; `kani::assert(inputs_unchanged)` |
| `loop_invariant` | Unroll to bound; assert invariant at each step |
| `loop_variant` | Unroll to bound; `kani::assert(v_after < v_before && v_after >= 0)` |
| `old_state` | Clone state; run kernel; assert pre/post relationship |
| `subcontract` | `kani::assume(parent_pre); kani::assert(child_pre);` + reverse for post |

### 5.3. Lean 4 (Theorem Proving)

| Type | Lean Pattern |
|---|---|
| `precondition` | Hypothesis `h : P x` in theorem statement |
| `postcondition` | `theorem name (h : pre x) : post (f x)` |
| `frame` | Conjunction: `∧ ∀ i, old_input[i] = new_input[i]` |
| `loop_invariant` | Induction: `∀ n, inv (iter^n state)` |
| `loop_variant` | Well-founded recursion via `Nat.lt_wfRel` |
| `old_state` | Universally quantified pre-state variable |
| `subcontract` | Implication chain: `parent_pre → child_pre` |

---

## 6. Type Invariants

Meyer's Design by Contract places *class invariants* as the third pillar
alongside preconditions and postconditions. A class invariant is a
predicate that must hold for every instance of a type at every stable
state — after construction and after every public method returns.

Our current `invariant` obligation type asserts properties of
*functions*. Type invariants assert properties of *data structures*.

### 6.1. YAML Schema Extension

```yaml
type_invariants:
  - name: tensor_validity
    type: "ValidatedTensor"
    predicate: "self.dims.iter().product::<usize>() == self.data.len()"
    description: "Data length equals product of dimensions"
  - name: tensor_non_empty
    type: "ValidatedTensor"
    predicate: "!self.dims.is_empty()"
    description: "At least one dimension"
```

Add `type_invariants: Vec<TypeInvariant>` to the `Contract` struct:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInvariant {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub predicate: String,
    #[serde(default)]
    pub description: Option<String>,
}
```

### 6.2. Rust Implementation Paths

**Path A: Stable Rust — `Invariant` trait pattern.**

This is the seL4 Rust verification pattern. Works today with Kani:

```rust
pub trait Invariant {
    fn is_valid(&self) -> bool;
}

impl Invariant for ValidatedTensor {
    fn is_valid(&self) -> bool {
        !self.dims.is_empty()
            && self.dims.iter().product::<usize>() == self.data.len()
    }
}
```

`pv scaffold` generates the `Invariant` impl from `type_invariants`.
`pv kani` generates preservation harnesses:

```rust
#[kani::proof]
fn verify_tensor_invariant_preserved_by_reshape() {
    let tensor: ValidatedTensor = kani::any();
    kani::assume(tensor.is_valid());
    let reshaped = tensor.reshape(&new_dims);
    assert!(reshaped.is_valid());
}
```

**Path B: Nightly Rust — `#[contracts::invariant]`.**

RFC #128044 adds native type invariants to the compiler. When stable,
`pv scaffold --nightly-invariants` would generate:

```rust
#![feature(contracts)]

#[contracts::invariant(self.dims.iter().product::<usize>() == self.data.len())]
#[contracts::invariant(!self.dims.is_empty())]
pub struct ValidatedTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}
```

### 6.3. Codegen: `pv invariants`

New CLI command `pv invariants <contract.yaml>` generates:

1. `Invariant` trait impl for each type with declared invariants
2. Kani preservation harnesses for every function in the contract's
   binding that takes or returns the invariant-bearing type
3. `debug_assert!(self.is_valid())` calls in constructors (via
   `#[contract]` macro integration)

### 6.4. Relationship to Existing Obligations

| Concept | Scope | Checked when |
|---|---|---|
| `invariant` obligation | Function property | After function returns |
| `type_invariants` | Data structure property | After construction + every public method |
| `precondition` obligation | Function input | Before function body |
| `postcondition` obligation | Function output | Before function returns |
| `frame` obligation | Mutation scope | After function returns |

Type invariants are *orthogonal* to function-level obligations. A
function can satisfy all its pre/postconditions while violating a type
invariant on the return value. The preservation harness catches this:
`kani::assume(input.is_valid()); /* operation */; assert!(output.is_valid())`.

### 6.5. Escape-Proof Enforcement for Type Invariants

The six-stage pipeline extends naturally:

| Stage | Type Invariant Enforcement |
|---|---|
| A (YAML) | `type_invariants[]` section in contract |
| B (Lean) | Invariant as a `Prop` on the type; preservation theorems |
| C (Lint) | `pv lint`: every type with invariants has preservation harnesses |
| D (build.rs) | Generate `Invariant` trait impl |
| E (Macro) | `#[contract]` inserts `debug_assert!(result.is_valid())` |
| F (Tests) | Kani preservation harnesses + proptest |

---

## 7. Coq Integration

The current verification ladder has Lean 4 at L5 (unbounded proof) and
Kani at L4 (bounded model checking). Adding Coq provides a second
path to unbounded proofs with different strengths:

| Prover | Strengths | Ecosystem |
|---|---|---|
| **Lean 4** | Mathlib (analysis, algebra), tactic automation, `sorry` tracking | Active, growing |
| **Coq** | CompCert (verified C compiler), Fiat Crypto, `coq-of-rust` bridge | Mature, battle-tested |

The two are NOT interchangeable — they verify different aspects:
- **Lean** excels at mathematical properties (softmax sums to 1 over ℝ)
- **Coq** excels at implementation verification (`coq-of-rust` translates
  actual Rust code to Coq, proving properties of *the code itself*)

### 7.1. YAML Schema Extension

```yaml
coq_spec:
  module: "SoftmaxSpec"
  imports:
    - "Require Import Reals."
    - "Require Import List."
  definitions:
    - name: "softmax_sum_to_one"
      statement: |
        Theorem softmax_partition_of_unity : forall (xs : list R),
          xs <> [] ->
          fold_left Rplus (map softmax_fn xs) 0 = 1.
  obligations:
    - links_to: "SM-INV-001"
      coq_lemma: "softmax_partition_of_unity"
      status: "proved|admitted|stub"
```

### 7.2. The `coq-of-rust` Bridge

`coq-of-rust` translates Rust code directly to Coq, giving a formal
model of the *actual implementation* — not a parallel specification.
This means Kani-verified bounds and Coq proofs refer to the *same
code*:

```bash
# In Makefile
coq-verify:
    coq-of-rust translate crates/aprender/src/kernels/softmax.rs \
        --output generated/coq/softmax.v
    coqc generated/coq/softmax.v
```

### 7.3. Codegen: `pv coq`

New CLI command `pv coq <contract.yaml>` generates:

1. `.v` file with `Require Import` statements from `coq_spec.imports`
2. Definitions from equations (translated to Coq `Definition`)
3. Theorem stubs from proof obligations (with `admit.` placeholders)
4. Obligation cross-references via `(** Obligation: SM-INV-001 *)`

```coq
(* Generated from softmax-kernel-v1 v1.0.0 *)
Require Import Reals.
Require Import List.

(** Equation: softmax *)
Definition softmax (xs : list R) : list R :=
  (* TODO: formalize *) nil.

(** Obligation: Output sums to 1 [invariant] *)
(** Paper: Bridle (1990) *)
Theorem softmax_partition_of_unity :
  forall (xs : list R),
    xs <> [] ->
    fold_left Rplus (softmax xs) 0 = 1.
Proof.
  admit. (* replace with proof *)
Qed.
```

### 7.4. Tiered Proof Strategy

Full Coq proofs require mathematician time. The practical approach
(CompCert, seL4, Fiat Cryptography) is tiered:

```
Tier 1: Kani (automated, bounded)      ← current, L4
Tier 2: Lean 4 (semi-automated, ℝ)     ← current, L5
Tier 3: Coq stubs (generated, admit)   ← new: pv coq
Tier 4: Coq proofs (human-verified)    ← manual, L5+
Tier 5: coq-of-rust (implementation)   ← automated translation
```

### 7.5. Audit Integration

`pv audit --coq` reports which obligations have:

| Status | Meaning |
|---|---|
| `kani_only` | Bounded verification (L4), no proof |
| `lean_proved` | Lean theorem over ℝ (L5) |
| `coq_stub` | Coq theorem generated but unproved (`admit`) |
| `coq_proved` | Coq theorem fully discharged |
| `coq_of_rust` | Implementation translated and verified |

### 7.6. Relationship to Lean-Kani Composition

The `stub_float` bridge works identically with Coq:
- Coq proves `exp > 0` (over ℝ)
- Kani stubs `f32::exp()` with `kani::any()` constrained by Coq's proof
- Kani verifies the surrounding Rust code preserves the invariant

The choice between Lean and Coq is per-obligation:
- Use Lean when Mathlib has the required analysis lemmas
- Use Coq when `coq-of-rust` can directly verify the implementation
- Use both when maximum assurance is needed (defense-in-depth)

### 7.7. Migration Path

1. Add `coq_spec` and `type_invariants` as optional schema sections
2. Implement `pv coq` command (generates `.v` stubs)
3. Implement `pv invariants` command (generates `Invariant` trait)
4. Extend `pv audit` with `--coq` flag
5. Extend `pv explain` to render type invariants and Coq status
6. Write exemplar Coq proofs for Tier 1 contracts (softmax, matmul)
7. Integrate `coq-of-rust` for implementation-level verification

### 7.8. References

1. Leroy, X. (2009). "Formal verification of a realistic compiler." *CACM* 52(7). (CompCert)
2. Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel." *SOSP 2009.*
3. Erbsen, A. et al. (2019). "Simple High-Level Code For Cryptographic Arithmetic." *S&P 2019.* (Fiat Crypto)
4. `coq-of-rust` — github.com/formal-land/coq-of-rust
5. Rust RFC #128044 — `core::contracts` type invariants (nightly tracking)
6. Lattuada, A. et al. (2023). "Verus: Verifying Rust Programs using Linear Ghost Types." arXiv:2303.05491.

---

## 8. Domain Applicability

### 8.1. Meyer's Universality Argument

Meyer was emphatic in OOSC (1997, Ch. 11) that Design by Contract is
not a systems programming technique — it is a *software correctness*
technique that applies to every domain. His key claims:

1. **Abstract Data Types are universal.** Every domain has them. A
   `Stack` has `push`/`pop` contracts. A `Window` has display
   invariants. A `Transaction` has ACID properties. The mathematical
   structure differs but the contract *form* is identical.

2. **The most valuable contracts encode domain axioms.** `softmax sums
   to 1` is a *domain* contract (from the math). `output buffer not
   corrupted` is a *code* contract. Domain contracts catch deeper bugs
   because they encode what the software *means*, not just what it
   *does*.

3. **Command-Query Separation (CQS)** applies everywhere: functions
   that return values must not have side effects; procedures that change
   state must not return values. This principle is domain-independent.

4. **Seamless development:** contracts flow from *analysis* (the domain
   expert's model) through *design* to *implementation*. The domain
   expert's constraints become the developer's preconditions and
   postconditions directly — no translation layer.

### 8.2. Current Domain Coverage

Our contract system is heavily weighted toward **scientific/numerical
kernels**:

| Project | Domain | Contract Focus |
|---|---|---|
| aprender | ML inference library | Kernel math (softmax, attention, RoPE) |
| entrenar | Training pipeline | Performance budgets (KAIZEN), grad computation |
| trueno | SIMD/CUDA kernels | Low-level numerical equivalence (SIMD = scalar) |
| realizar | Task execution | Pipeline orchestration |
| bashrs | SSC pipeline | Encoder/classifier contracts |
| simular | Simulation engine | Energy conservation, checkpoint roundtrip, gradient stability |
| probar | WASM/TUI testing | (no contracts yet) GUI coverage, accessibility, visual regression |
| forjar | Infrastructure as Code | DAG ordering, atomic writes, recipe determinism, codegen dispatch |
| batuta | Orchestration / transpilation | (no contracts yet) Pipeline stages, semantic equivalence |
| pmat | Code quality / AI context | (no contracts yet) Uniform interface contracts, analysis invariants |
| rmedia | Headless video editor | (no contracts yet) Codec parity, determinism, SVG quality, animation timing |

Simular already consumes provable-contracts (3 YAML contracts +
`provable-contracts-macros` dependency). Forjar is also a direct
consumer (`#[contract]` macros on DAG ordering, atomic state writes,
recipe expansion, codegen dispatch + `build.rs` binding verification
with `WarnOnGaps` policy). Batuta and PMAT do not reference provable-
contracts but both operate in domains with rich contractable properties.
Probar does not yet reference provable-contracts but operates in a
domain — GUI/UX correctness — where Meyer's DbC types are maximally
applicable.

This covers Meyer's "scientific computing" domain thoroughly. But the
stack spans domains where our current property-only obligation types
are insufficient without the Eiffel DbC extensions.

### 8.3. Domain Contract Patterns

#### Presentation / UI (presentar + probar)

Presentar is the stack's WASM-first UI framework — a pure Rust
widget system with constraint-based layout (flexbox + CSS grid),
two-phase measure/layout engine, accessibility tree, real-time
WebSocket streaming, and the Brick Architecture (tests ARE the
interface). Probar tests presentar applications via CDP browser
automation, pixel coverage tracking, and state machine playbooks.

This domain has contracts rooted in *geometric invariants*,
*accessibility standards*, and *event semantics* — not numerical
equations. The concrete architecture maps to DbC as follows.

**Dimensions and Layout Contracts:**

The layout engine runs a two-phase algorithm: measure (bottom-up,
computing intrinsic sizes given `Constraints`) then layout (top-down,
assigning final `Rect` positions). Each phase has contractable
properties:

| Obligation Type | Layout Example |
|---|---|
| `precondition` | `Constraints` are valid: `min ≤ max`, both finite or explicitly unbounded |
| `postcondition` | After measure, returned `Size` satisfies constraints: `constraints.constrain(size) == size` |
| `invariant` | Every visible widget has `width > 0 ∧ height > 0` after layout |
| `frame` | Measure phase reads widget tree only; does not mutate positions or state |
| `conservation` | Flex layout: `Σ child_widths + Σ gaps = parent_content_width` (no pixels lost) |
| `bound` | Total measure + layout time < 16ms (BrickBudget: `measure_ms + layout_ms + paint_ms ≤ 16`) |
| `loop_invariant` | During flex distribution: remaining space ≥ 0 at each child allocation |
| `loop_variant` | Remaining unpositioned children = `total - positioned`, strictly decreasing |
| `old_state` | After resize event, `layout_tree.bounds(widget) ≠ old(layout_tree.bounds(widget))` for affected widgets |
| `determinism` | Same widget tree + same constraints → identical `LayoutTree` |

Concrete types from presentar that carry contracts:

| Type | Contract Surface |
|---|---|
| `Constraints { min_width, max_width, min_height, max_height }` | `tight()` pre: min == max. `loose()` pre: min ≤ max. `constrain()` post: result in [min, max] |
| `Size { width, height }` | Post: `width ≥ 0 ∧ height ≥ 0`. `area()` post: `= width × height` |
| `Rect { x, y, width, height }` | `contains(point)` post: `x ≤ p.x ≤ x+w ∧ y ≤ p.y ≤ y+h` |
| `FlexItem { grow, shrink, basis }` | Pre: `grow ≥ 0`, `shrink ≥ 0`. `collapse_if_empty` post: size = 0 when no content |
| `GridTemplate { columns, rows, gap }` | Pre: all `TrackSize` values ≥ 0. Post: `Σ column_widths + Σ gaps = available_width` |

**Element Contracts (Menu, Footer, Pane, Widget):**

Presentar's `Widget` trait (which extends `Brick`) defines the
verify-measure-layout-paint lifecycle. Each widget type has specific
contracts:

| Widget Element | Key Contracts |
|---|---|
| **Menu** | Pre: items list non-empty. Post: exactly one item has `selected` state. State machine: `closed → open → item_hover → selected → closed`. Frame: opening menu doesn't modify parent layout |
| **Footer** | Invariant: always pinned to viewport bottom (`y + height == viewport.height`). Frame: content updates preserve height unless content overflows |
| **Pane / Panel** | Pre: split ratio ∈ (0, 1). Conservation: `left_width + divider + right_width = parent_width`. Old-state: resizing divider changes ratio but preserves total width |
| **DataTable** | Pre: column count > 0. Post: sorted column satisfies ordering obligation. Idempotency: sorting already-sorted column is no-op. Virtual scroll invariant: only `visible_range + overscan` rows rendered |
| **TextInput** | Pre: `is_focusable() == true`. Post: after `TextInput` event, `value.len() = old(value.len()) + input.len()`. State machine: `unfocused → focused → editing → validated → unfocused` |
| **Button** | Pre: `accessible_name().is_some()`. Post: `click` event emitted only when `enabled`. Subcontract: `IconButton` refines `Button` (same click semantics, adds icon rendering) |
| **Scroll Container** | Loop invariant: `offset ≥ 0 ∧ offset ≤ content_height - viewport_height` at every scroll step. Old-state: `offset_after = old(offset) + delta` clamped to bounds |
| **Border** | Frame: adding border modifies rendered appearance only; child widget's content rect unchanged. Post: `inner_rect = outer_rect.deflate(border_width)` |

**Behavior Contracts (WebSocket, Forms, Events):**

| Behavior | Obligation Types |
|---|---|
| **WebSocket (StreamMessage)** | State machine: `Disconnected → Connecting → Connected → Reconnecting → Failed`. Pre: `ws_url` is valid URI. Post: after `Subscribe`, server sends `Data` messages with matching `id`. Invariant: `seq` numbers strictly increasing per stream. Frame: receiving messages doesn't modify application state until processed by `update()`. Loop invariant: reconnect backoff ≤ `max_backoff` at every retry. Loop variant: retry_count remaining = `max_retries - attempt` |
| **Form Validation** | Pre: all required fields bound via `Binding`. Post: `submit()` only succeeds if all validators pass. State machine: `pristine → dirty → validating → valid \| invalid → submitted`. Old-state: after validation, `errors.len()` either decreases (fixes) or increases (new violations). Frame: validation reads field values only; doesn't modify them |
| **Event Dispatch** | Pre: `event.target` is a valid widget ID in the tree. Post: exactly one widget handles each event (no duplicate dispatch). Frame: event handling modifies only the target widget's state and its ancestors (bubbling). Determinism: same event + same state → same `update()` result |
| **Two-Way Binding** | Invariant: `widget.value == state[binding.source_path]` at all quiescent states. Roundtrip: `set(get(path)) = get(path)` (setting the current value is a no-op). Frame: binding update modifies only the bound property; other state preserved |
| **Virtual Scrolling** | Pre: `estimated_item_height > 0`. Post: only items in `visible_range + overscan` are rendered. Conservation: total scrollable height = `Σ item_heights` (no gaps). Bound: render count ≤ `visible_count + 2 × overscan_count` |
| **Device Emulation** | Pre: `viewport.width > 0 ∧ viewport.height > 0`. Post: `device_scale_factor` applied to all coordinates. Frame: emulation changes viewport only; doesn't modify DOM content. Subcontract: `MobileDevice` refines `Device` (adds touch events, is_mobile = true) |

**Accessibility Contracts:**

| Obligation Type | Accessibility Example |
|---|---|
| `completeness` | Every `is_interactive()` widget has `accessible_name().is_some()` |
| `invariant` | Contrast ratio ≥ 4.5:1 for normal text, ≥ 3.0:1 for large text (WCAG 2.1 AA) |
| `bound` | Flash rate < 3 per second (WCAG 2.3.1 photosensitivity) |
| `postcondition` | After focus change, `AccessibleNode.focused == true` for exactly one node |
| `conservation` | Focus tab order: `Σ focusable_elements` unchanged after re-render |
| `frame` | Accessibility tree update modifies only changed nodes; unchanged nodes preserve all properties |
| `state_machine` | Live region: `Off → Polite → Assertive` transitions; `Assertive` interrupts screen reader immediately |

**Brick Architecture as DbC:**

Probar's Brick Architecture is the purest expression of Meyer's
DbC in the stack. Each `Brick` IS a contract:

```
Brick = {
    assertions: Vec<BrickAssertion>,  ← postconditions
    budget: BrickBudget,               ← bound obligations
    verify() → BrickVerification,      ← contract checker
}

Widget extends Brick:
    VERIFY  → check assertions     ← precondition gate
    MEASURE → compute intrinsic    ← pure function (frame: no mutation)
    LAYOUT  → assign positions     ← postcondition: fits constraints
    PAINT   → generate commands    ← only executes if VERIFY passed
```

The Jidoka principle (stop-the-line if any assertion fails) IS
Meyer's "exception on contract violation" — if a `BrickAssertion`
fails, the widget does not paint. This is the Eiffel `check`
instruction made architectural.

**BrickHouse composition** maps to `subcontract`: each child brick's
budget must sum to ≤ parent's total budget. This is a `conservation`
obligation on performance:

```yaml
proof_obligations:
  - type: conservation
    property: "Sum of child budgets ≤ BrickHouse total budget"
    formal: "Σ brick_budget_ms(child_i) ≤ house_budget_ms"
```

**Key insight:** presentation contracts derive from *design system
axioms* (WCAG, flexbox spec, grid spec) and *accessibility standards*,
just as kernel contracts derive from *paper equations*. The pipeline
phases map directly:

```
WCAG 2.1 / Flexbox Spec / Grid Spec  (≈ arXiv paper)
  → Layout Invariants + A11y Rules    (≈ equations)
    → Contract YAML                    (≈ contract)
      → Widget Trait + Brick           (≈ kernel trait)
        → presentar Implementation     (≈ scalar/SIMD kernel)
          → probar Property Tests      (≈ probar falsification)
            → Kani Model Checking      (≈ Kani harness)
```

#### Data Pipeline / ETL

Data pipelines have contracts rooted in schema conformance, data
quality, and transformation correctness.

| Obligation Type | Data Pipeline Example |
|---|---|
| `precondition` | Input schema matches expected version; no null primary keys |
| `postcondition` | Output row count = input row count (for map transforms) |
| `frame` | Transform modifies target columns only; source columns immutable |
| `invariant` | Foreign key references resolve at every pipeline stage |
| `old_state` | `output.row_count = old(input.row_count) - filtered_count` |
| `conservation` | Sum of monetary values preserved across currency conversion |
| `completeness` | All enum variants in source schema have mapping rules |
| `determinism` | Same input batch produces identical output regardless of parallelism |
| `roundtrip` | `unpivot(pivot(table)) = table` |
| `loop_invariant` | Streaming window: buffer size ≤ `max_window` at every step |

#### API / Service Layer

API contracts derive from protocol specifications, SLAs, and backward
compatibility guarantees.

| Obligation Type | API Example |
|---|---|
| `precondition` | Request body conforms to JSON schema v2.1; auth token valid |
| `postcondition` | Response status 200 implies body conforms to response schema |
| `frame` | GET requests modify no server state (HTTP idempotency) |
| `subcontract` | API v2 is a valid refinement of v1 — accepts all v1 requests, v2 responses are v1-compatible |
| `state_machine` | Order lifecycle: `created → paid → shipped → delivered → closed` |
| `old_state` | After PATCH, `version = old(version) + 1` |
| `bound` | Response latency p99 < 200ms |
| `idempotency` | Repeated PUT with same body produces identical state |
| `determinism` | Same request yields same response (for cacheable endpoints) |

#### Simulation (simular)

Simular is the stack's unified simulation engine covering physics
(orbital mechanics, N-body, rigid body, fluid dynamics), Monte Carlo
methods, optimization (Bayesian, GRASP), and ML training simulations.
It already consumes provable-contracts with 3 YAML contracts
(checkpoint, gradient, loss-functions). Simulation contracts derive
from *physics conservation laws* and *numerical integration theory*.

Meyer's Eiffel DbC maps naturally to simulation because simulations
are inherently stateful, iterative systems where *what changes* and
*what must not change* must be specified precisely.

| Obligation Type | Simulation Example |
|---|---|
| `precondition` | Initial state has finite energy; timestep `dt > 0` |
| `postcondition` | After N-body step, all positions are finite and within bounds |
| `invariant` | Total system energy conserved within drift tolerance |
| `frame` | Integration step modifies positions and velocities only; masses and gravitational constant unchanged |
| `loop_invariant` | Symplectic integrator preserves phase-space volume at every timestep |
| `loop_variant` | Remaining simulation steps = `total_steps - current_step`, strictly decreasing |
| `old_state` | `energy(state_new) - energy(old(state)) < ε_drift` per step |
| `conservation` | Total momentum conserved in closed N-body system |
| `determinism` | Same RNG seed + same initial state → identical trajectory |
| `roundtrip` | `deserialize(serialize(checkpoint)) = checkpoint` (checkpoint fidelity) |
| `bound` | Gradient norm ≤ `max_clip` after clipping |
| `equivalence` | Verlet integrator matches RK4 within tolerance for smooth potentials |
| `monotonicity` | Loss decreases monotonically for convex objectives with valid learning rate |
| `subcontract` | Custom integrator refines base `Integrator` contract |

**Key insight:** Simular's jidoka module (stop-on-error anomaly
detection) is fundamentally a *runtime contract checker*. It monitors
for NaN, Inf, energy drift, and constraint violations — these are
exactly the invariants and postconditions that should be declared in
YAML contracts and verified statically via Kani, not just caught at
runtime. The Eiffel DbC types make this declarative:

```
Runtime jidoka check               →  YAML contract equivalent
─────────────────────────────────────────────────────────────────
NaN/Inf detection                  →  postcondition: output is finite
Energy drift > threshold           →  old_state: |E_new - E_old| < ε
Constraint violation               →  invariant: constraint holds
State corruption                   →  frame: only specified fields change
```

#### Testing / QA (probar)

Probar is the stack's WASM and TUI testing framework — a Playwright-
compatible, zero-JavaScript, pure Rust testing tool for games,
simulations, and terminal UIs. It does not yet consume provable-
contracts, but its domain is rich with contractable properties.

Testing frameworks are a distinctive case for DbC because the
*framework itself* has contracts about what constitutes correct test
behavior. Meyer addressed this in OOSC Ch. 11 under "Who checks the
checker?" — the testing tool's own invariants must be at least as
rigorous as the code it tests.

| Obligation Type | Testing Framework Example |
|---|---|
| `precondition` | Browser/WASM runtime is connected before locator query |
| `postcondition` | After `click(element)`, element's click handler has been invoked |
| `invariant` | Pixel coverage map has dimensions matching viewport at all times |
| `frame` | Visual regression comparison modifies diff buffer only; reference image unchanged |
| `old_state` | After navigation, `history.length = old(history.length) + 1` |
| `state_machine` | Test lifecycle: `init → setup → running → teardown → complete \| failed` |
| `determinism` | Replay with same seed produces identical event sequence |
| `roundtrip` | `deserialize(serialize(playbook)) = playbook` (YAML playbook fidelity) |
| `completeness` | Every interactive element in DOM has at least one locator match |
| `idempotency` | Running same assertion twice yields same pass/fail result |
| `subcontract` | `TuiLocator` refines `Locator` — accepts same selectors, returns TUI-specific elements |
| `bound` | Screenshot capture completes within 500ms |
| `conservation` | Total pixel count in coverage heatmap = viewport width × height |
| `loop_invariant` | During auto-wait polling: timeout budget remaining ≥ 0 |
| `loop_variant` | Retry attempts remaining = `max_retries - attempt`, strictly decreasing |

**Key insight:** Probar's "Brick Architecture" pattern (tests ARE the
interface) is a direct expression of Meyer's seamless development:
the test specification IS the component contract. A brick's type-safe
selectors declare preconditions (element must exist), its assertions
declare postconditions (visual state after interaction), and its
composition rules (`BrickHouse` budgets) declare frame conditions
(total test time must not exceed budget). The gap is that these
contracts live in Rust code, not in declarative YAML contracts that
can be validated, scored, and verified independently.

#### Infrastructure as Code (forjar)

Forjar is the stack's sovereign IaC tool — a single-binary Rust
replacement for Terraform/Ansible that manages bare-metal machines and
containers over SSH using YAML configs, BLAKE3 content-addressed state,
and deterministic DAG execution. It already consumes provable-contracts
with `#[contract]` annotations on its core algorithms (DAG ordering,
atomic writes, recipe determinism, codegen dispatch).

Infrastructure provisioning is a domain where Meyer's DbC is
*especially* natural because the entire paradigm is already
contractual: a desired-state config is a *specification* (contract),
the planner computes a *diff* (obligation), and the executor
*converges* the system (fulfillment). Forjar's existing design
patterns map directly to Eiffel DbC concepts:

| Forjar Concept | Eiffel DbC Equivalent |
|---|---|
| Resource `when:` guards | `precondition` |
| Planner action (Create/Update/NoOp) | `postcondition` (desired state reached) |
| BLAKE3 state snapshot | `old_state` (hash comparison pre/post) |
| Jidoka failure isolation | `frame` (partial failure preserves unchanged resources) |
| DAG wave execution | `loop_invariant` (DAG ordering respected at every wave) |
| Idempotency contract (FJ-210) | `idempotency` + `postcondition` |
| Bashrs transport safety | `precondition` (script passes shell validation) |

| Obligation Type | Infrastructure as Code Example |
|---|---|
| `precondition` | Target machine reachable via SSH; bashrs validates generated script before execution |
| `postcondition` | After apply, resource hash in lock file matches desired-state hash |
| `invariant` | DAG ordering: no resource executes before its dependencies |
| `frame` | Applying resource R modifies only R's state; all other resources' lock entries preserved |
| `old_state` | `lock_hash(resource, after) ≠ lock_hash(resource, old(before))` implies action was Update |
| `idempotency` | Second apply on converged state produces zero changes (FJ-210) |
| `determinism` | Same config YAML + same lock state → identical execution plan |
| `loop_invariant` | During DAG wave execution: all completed resources have converged state |
| `loop_variant` | Remaining unprocessed resources in DAG = `total - completed`, strictly decreasing |
| `roundtrip` | `deserialize(serialize(lock_state)) = lock_state` (BLAKE3 lock fidelity) |
| `conservation` | Total resource count in plan = creates + updates + destroys + no-ops |
| `subcontract` | Pepita transport refines SSH transport — accepts same scripts, adds namespace isolation |
| `state_machine` | Resource lifecycle: `absent → creating → converged → drifted → updating → converged → destroying → absent` |
| `completeness` | Every resource in config has exactly one handler in the resource registry |
| `bound` | SSH retry backoff ≤ max 4 attempts; copia delta block size ≤ 1MB |

**Key insight:** Forjar's planner is already a contract evaluator. It
compares desired state (the "postcondition") against current state (the
"old_state") and computes the minimal set of actions to satisfy the
contract. The Eiffel DbC types make this structure *explicit in the
contract YAML* rather than implicit in Rust code. In particular:

- **`frame` is critical for IaC.** When forjar applies a package
  install, it must not disturb file resources, service resources, or
  other machines. This is the same "only modifies what it claims to
  modify" guarantee that Meyer's `only` clause provides. Forjar's
  jidoka policy (stop on first failure, preserve partial state) is a
  runtime enforcement of the frame condition.

- **`old_state` is the natural language of drift detection.** Forjar's
  tripwire module compares `hash(current)` against `hash(old(lock))`.
  This is exactly `Q(old(state), new(state))` — the DbC old-state
  obligation. Declaring this in the contract makes the drift detection
  contract auditable and testable via Kani.

- **`subcontract` captures transport substitutability.** Forjar's
  transport abstraction (local, SSH, container, pepita) is a textbook
  case of behavioral subtyping: each transport accepts the same script
  input (weakened precondition: pepita adds isolation but doesn't
  require it), and guarantees the same execution semantics (strengthened
  postcondition: pepita adds namespace isolation on top of base
  guarantees).

#### Orchestration / Transpilation (batuta)

Batuta is the stack's orchestration framework — a transpilation
pipeline that converts Python/C/Shell to Rust, coordinates multi-crate
releases, and routes ML workloads to backends (GPU/SIMD/scalar) via
cost-based selection. It applies Toyota Production System principles
(Jidoka, Poka-Yoke, Kaizen) to code transformation.

Transpilation is a domain where DbC provides *correctness guarantees
that are otherwise impossible to test exhaustively*. You cannot run
every possible Python program through a transpiler, but you can prove
structural properties of the transformation.

| Obligation Type | Orchestration / Transpilation Example |
|---|---|
| `precondition` | Input source is syntactically valid Python/C/Shell |
| `postcondition` | Output Rust compiles; semantically equivalent on test suite |
| `equivalence` | `transpile(source).eval(input) = source.eval(input)` for all test inputs |
| `frame` | Transpilation does not modify source files; only output directory written |
| `invariant` | Type safety preserved through all 5 pipeline stages (Analysis → Build) |
| `loop_invariant` | Pipeline context valid at each stage transition |
| `state_machine` | Pipeline stages: `analysis → transpilation → optimization → validation → build` |
| `subcontract` | `PyTorchConverter` refines `TranspilerPlugin` — accepts PyTorch subset, produces Realizar ops |
| `determinism` | Same source + same config → identical transpiled output |
| `completeness` | All NumPy ops in input have Trueno mappings; all sklearn algorithms have Aprender mappings |
| `bound` | Context generation < 5s for 10K LOC; memory < 500MB for 100K LOC |
| `conservation` | Number of functions in output = number of functions in input (no dropping) |
| `old_state` | After adding a converter plugin, `plugin_count = old(plugin_count) + 1` |

**Key insight:** Batuta's 5-phase pipeline with Jidoka validation gates
is a natural DbC structure. Each stage has implicit pre/postconditions
(the `PipelineStage` trait's `validate()` method). Making these
explicit as YAML contracts would let provable-contracts verify the
*transpiler itself* — not just the transpiled code.

The `BackendSelector` (MoE routing for GPU/SIMD/scalar selection) is
a particularly strong candidate for `postcondition` + `bound`
contracts: given an operation's complexity profile, the selected
backend must satisfy both correctness (equivalence to scalar) and
performance bounds (e.g., the 5× PCIe rule from Gregg & Hazelwood).

#### Code Quality / AI Context (pmat)

PMAT (paiml-mcp-agent-toolkit) is the stack's code quality analysis
and AI context generation system — the `pmat` binary used throughout
the stack for TDG scoring, semantic search (`pmat query`), mutation
testing, and MCP server hosting. It supports 17+ languages and
provides a "uniform contracts" architecture pattern for its 19+ MCP
tools.

PMAT's "uniform contracts" pattern is notable: `BaseAnalysisContract`
and its specializations (`AnalyzeComplexityContract`,
`AnalyzeSatdContract`, etc.) enforce identical parameter sets across
CLI, MCP, and HTTP interfaces. This is a *structural* application of
DbC — Meyer's class invariant ensuring interface uniformity — distinct
from our *mathematical* contracts but equally contractable.

| Obligation Type | Code Quality / Analysis Example |
|---|---|
| `precondition` | Project path exists and contains supported language files |
| `postcondition` | Output format matches requested format; all metrics are finite |
| `invariant` | TDG grade is monotonic: improving code never worsens the grade |
| `frame` | Analysis never modifies the analyzed codebase (read-only) |
| `determinism` | Same codebase state → same TDG grade, same complexity scores |
| `idempotency` | Running analysis twice produces identical output |
| `bound` | Analysis completes in < 5s for 10K LOC |
| `completeness` | All files matching language filter are analyzed |
| `conservation` | Total LOC reported = sum of per-file LOC |
| `equivalence` | CLI output matches MCP output matches HTTP output (uniform contracts) |
| `roundtrip` | `deserialize(serialize(analysis_result)) = analysis_result` |
| `subcontract` | `AnalyzeComplexityContract` refines `BaseAnalysisContract` — same base params, adds complexity thresholds |
| `old_state` | After mutation testing, `killed_mutants ≥ old(killed_mutants)` (monotonic improvement tracking) |
| `loop_invariant` | During multi-file analysis: partial results consistent with final aggregate |

**Key insight:** PMAT's uniform contracts pattern is Meyer's class
invariant in disguise. The `BaseAnalysisContract` struct with
`#[serde(flatten)]` inheritance ensures that every interface (CLI, MCP,
HTTP) receives identical parameters — a *structural guarantee* that
could be formalized as a `subcontract` obligation: each specialized
contract (complexity, SATD, TDG) refines the base contract by adding
fields without removing any.

PMAT is also the stack's *quality oracle*. Its TDG scores and mutation
coverage data could *feed into* provable-contracts' scoring system as
an additional dimension — binding provable-contracts' formal
verification metrics with PMAT's empirical quality metrics for a
complete picture.

#### Media Asset Pipeline (rmedia)

Rmedia is the stack's pure Rust headless video editor — an 8-crate
workspace that transforms SVG frame sequences, audio, and SRT
transcripts into rendered MP4 videos with deterministic, scorable
output. It has a 7-dimension render pipeline score (speed, efficiency,
determinism, observability, reliability, quality, pipeline health)
and machine-enforced SVG visual quality floors.

Media asset production is a domain where DbC types provide guarantees
that are otherwise *impossible to test manually*. You cannot watch
every rendered video frame-by-frame, but you can prove codec parity,
duration bounds, determinism, and visual quality minimums.

**Rendering contracts:**

| Obligation Type | Rendering Example |
|---|---|
| `precondition` | Input SVG frames are valid (parseable, 1920x1080 viewBox); SRT file exists and parses |
| `postcondition` | Output MP4 has correct codec (h264), resolution (1920x1080), channels (stereo), sample_rate (48kHz) |
| `equivalence` | `ffprobe(rmedia_output) = ffprobe(melt_output)` for codec, resolution, channels, sample_rate |
| `determinism` | Same inputs → identical file hash (SRT locked via SHA-256) |
| `bound` | RTX (real-time factor) ≥ 1.5x; render time < total_frames / fps / 1.5 |
| `frame` | Rendering modifies output directory only; input SVGs, SRT, and audio unchanged |
| `invariant` | YUV420P color space maintained through entire pipeline (no unnecessary swscale) |
| `loop_invariant` | During frame pipeline: bounded channel depth ≤ 16 at every producer step |
| `loop_variant` | Remaining frames = total_frames - rendered_frames, strictly decreasing |
| `old_state` | SRT lock: `sha256(srt_content) = old(sha256_lock)` — transcript hasn't drifted |
| `conservation` | Output frame count = ceil(transcript_duration × fps) ± 1 frame |
| `roundtrip` | `decode(encode(yuv_frame)) ≈ yuv_frame` within CRF tolerance |

**Animation contracts:**

| Obligation Type | Animation Example |
|---|---|
| `precondition` | Keyframe sequence has ≥ 2 frames; easing function is valid enum variant |
| `postcondition` | Interpolated value at t=0.0 equals start value; at t=1.0 equals end value |
| `invariant` | Easing parameter t ∈ [0, 1] at every interpolation step |
| `bound` | Animation timing aligns to SRT within ±2 frames |
| `monotonicity` | For linear easing: t₁ < t₂ → lerp(t₁) ≤ lerp(t₂) |
| `idempotency` | Rendering same animation plan twice produces identical output |

**SVG quality contracts (machine-enforced visual floors):**

| Obligation Type | SVG Quality Example |
|---|---|
| `bound` | Fill opacity ≥ 0.20 for content, ≥ 0.10 for backgrounds |
| `bound` | Stroke width ≥ 3.0 for icon outlines, ≥ 2.0 for details |
| `bound` | Font size ≥ 36px for hero titles, ≥ 14px for labels |
| `bound` | Icon bounding box ≥ 80×80 for hero banners |
| `completeness` | Full-canvas background rect present as first child |
| `invariant` | No `<text>` elements in no-text images (logo, marketing, nav) |

**Course pipeline contracts:**

| Obligation Type | Course Pipeline Example |
|---|---|
| `completeness` | All lessons rendered with valid video files |
| `conservation` | Number of output videos = number of input lesson directories |
| `postcondition` | Aggregate score computed via harmonic mean; any zero dimension → F grade |
| `old_state` | SRT lock hash matches across multiple renders |
| `state_machine` | Pipeline phases: `discover → validate → render → score → generate_marketing` |
| `frame` | Course pipeline modifies output directory only; source lessons unchanged |

**Key insight:** Rmedia's obsession with reproducibility — SHA-256 SRT
locks, integer-only compositing (`(a * (256-p) + b * p) >> 8`), no
f32 intermediates, deterministic frame output — makes it the most
naturally contractable non-kernel domain in the stack. Every property
is already quantified; the contracts just need to be formalized in
YAML. The 7-dimension render pipeline score is essentially a scoring
contract waiting to be extracted.

The `pv generate` pipeline should produce deterministic README.md
and CI workflow files for rmedia that validate:
- Codec parity with melt (`make bench-parity`)
- Determinism (two renders produce identical output)
- SVG quality floors (fill opacity, stroke width, font size)
- SRT lock integrity (SHA-256 hash match)
- Render pipeline score ≥ threshold (per-dimension and aggregate)

#### Configuration / Infrastructure (General)

Infrastructure contracts beyond IaC — deployment invariants and
resource constraints for orchestration platforms.

| Obligation Type | Infrastructure Example |
|---|---|
| `precondition` | Available memory ≥ model size + KV cache budget before load |
| `postcondition` | After deployment, health check returns 200 within 30s |
| `invariant` | Replica count ≥ `min_replicas` at all times |
| `frame` | Rolling update modifies at most `max_surge` pods; others unchanged |
| `conservation` | Total allocated GPU memory ≤ physical GPU memory |
| `bound` | Container CPU limit ≤ node capacity |
| `loop_variant` | Rolling restart: remaining pods = `total - restarted`, strictly decreasing |

### 8.4. Domain-Specific `references` and `equations`

Meyer's seamless development principle means the `metadata.references`
field and `equations` section should point to the *domain authority*,
not just arXiv papers:

| Domain | `references` Source | `equations` Encode |
|---|---|---|
| ML kernels | arXiv papers | Governing math (softmax formula) |
| Simulation | Physics textbooks, numerical methods papers | Conservation laws, integrator formulas |
| Testing/QA | Playwright docs, WCAG 2.2, W3C specs | Coverage metrics, assertion semantics |
| IaC (forjar) | POSIX spec, SSH RFC 4253, Nix papers | DAG ordering, BLAKE3 hash semantics, convergence |
| Orchestration (batuta) | Language specs (Python, POSIX), PL theory | Transpilation semantics, cost models |
| Code quality (pmat) | McCabe (1976), Halstead (1977), SATD literature | Complexity metrics, TDG scoring formulas |
| Media (rmedia) | FFmpeg docs, H.264/H.265 specs, MLT format, WCAG contrast | Codec parity, compositing math, easing curves, SVG quality floors |
| Presentation | WCAG 2.2, Material Design spec | Layout algorithms (flexbox distribution) |
| Data pipeline | Schema registry, data dictionary | Transform rules (join semantics) |
| API | OpenAPI spec, RFC 7231 | Request/response schemas |
| Infrastructure | Kubernetes API spec, cloud SLAs | Resource models (roofline) |

This is a natural extension: the pipeline's Phase 1 (EXTRACT) already
says "arXiv PDF → canonical math." For non-kernel domains, Phase 1
becomes "domain specification → canonical rules."

### 8.5. When DbC Types Matter Most by Domain

Not every obligation type is equally useful in every domain. The Eiffel
DbC types have different gravity depending on the domain:

| Type | Kernels | Simulation | IaC | Media | Orchestration | Quality | Testing | Presentation | Data | API |
|---|---|---|---|---|---|---|---|---|---|---|
| `precondition` | Medium | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** |
| `postcondition` | Medium | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** |
| `frame` | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | Medium |
| `loop_invariant` | **High** | **High** | **High** | **High** | **High** | Medium | Medium | Medium | Medium | Low |
| `loop_variant` | Medium | **High** | **High** | **High** | Medium | Low | Medium | Low | Low | Low |
| `old_state` | Medium | **High** | **High** | **High** | Medium | Medium | Medium | **High** | **High** | **High** |
| `subcontract` | Medium | Medium | **High** | Medium | **High** | **High** | **High** | **High** | Medium | **High** |

**Pattern:** The pre/post/frame/old-state cluster is *most* valuable
outside the kernel domain. Kernel contracts can often get away with
algebraic property types (`invariant`, `bound`, `monotonicity`) because
the math is self-contained. But simulation, IaC, orchestration, code
quality, testing, presentation, data, and API contracts inherently
describe *interactions between components* — exactly the caller/callee
relationship that Meyer's framework models.

Simulation and IaC are the two domains where *all seven* DbC types
are high-value. Simulations are stateful (frame, old-state), iterative
(loop_invariant, loop_variant), have strict input requirements
(precondition), must guarantee physical plausibility (postcondition),
and compose via integrator substitution (subcontract). IaC shares
this profile: infrastructure is stateful (frame — don't touch other
resources), convergence is iterative (DAG waves with loop_invariant/
variant), inputs must be validated (precondition — SSH reachable,
bashrs valid), outputs must be verified (postcondition — resource
converged), state comparison is the core operation (old_state — hash
diffing), and transports are substitutable (subcontract — pepita
refines SSH).

### 8.6. Cross-Project Dependency Graph and Contract Flow

The PAIML stack has a layered dependency structure. Contracts flow
*upward* through the dependency graph — a contract on trueno's SIMD
kernel propagates obligations to every consumer (aprender, entrenar,
realizar, presentar, probar, pmat, batuta, renacer).

```
Level 0 (Foundation)     provable-contracts ←──── trueno
                                |                    ↓
Level 1 (Direct)          forjar ◄──┘     ┌── aprender ──┬── entrenar
                                          │              │
                                          ├── presentar  ├── renacer
                                          │      ↓       │
                                          ├── probar     └── pmat
                                          │
Level 2 (Composite)                       ├── realizar ── pacha
                                          │
Level 3 (Orchestration)                   └── batuta ─── simular
```

**Current contract binding coverage:**

| Project | binding.yaml | `#[contract]` macros | Tier |
|---|---|---|---|
| trueno | Yes (22 bindings) | — | 1-2 |
| aprender | Yes (301 bindings) | 38 annotations | 1-5 |
| entrenar | Yes (96 bindings) | — | 4 |
| realizar | Yes (23 bindings) | — | 3 |
| forjar | Yes | 4 annotations | 9 (IaC) |
| simular | Yes (3 contracts) | macros dep | 8 (simulation) |
| presentar | **No** | — | 13 (presentation) |
| probar | **No** | — | 12 (testing) |
| batuta | **No** | — | 10 (orchestration) |
| pmat | **No** | — | 11 (code quality) |
| renacer | **No** | — | — |
| pacha | **No** | — | — |

**Cross-project contract obligations:**

The Eiffel DbC types create new cross-project contract relationships
that don't exist with property-only types:

**1. Subcontract chains across the dependency graph.**

When trueno exposes a `Kernel` trait and aprender implements it, the
implementation is a behavioral subtype. A `subcontract` obligation
makes this explicit:

```yaml
# In aprender's binding
proof_obligations:
  - type: subcontract
    property: "aprender::softmax refines trueno Kernel trait"
    formal: "pre(Kernel::execute) → pre(aprender::softmax)"
    parent_contract: "softmax-kernel-v1"
```

This propagates: if realizar wraps aprender's softmax, it inherits
the subcontract chain. `pv validate` can verify the entire chain
from trueno → aprender → realizar.

**2. Frame conditions at API boundaries.**

When entrenar calls trueno's GPU kernels, the frame condition must
hold across the FFI boundary: trueno's kernel must not corrupt
entrenar's training state. This is a cross-project frame obligation:

```yaml
# In entrenar's contract
proof_obligations:
  - type: frame
    property: "trueno kernel modifies output buffer only"
    formal: "modifies(output) ∧ preserves(weights, gradients, optimizer_state)"
```

**3. Precondition propagation through the stack.**

A precondition on trueno's matmul (input dimensions must match)
propagates to every consumer. Each layer can *weaken* the
precondition (Meyer's `require else`):

```
trueno:    require dimensions_match(A, B)
aprender:  require else dimensions_broadcastable(A, B)  # weaker
realizar:  require else model.expects_shape(input)       # weaker still
```

**4. Performance bound composition (BrickBudget flow).**

Presentar's `BrickBudget` (16ms per frame) decomposes into trueno
GPU kernel bounds. A `bound` obligation on the widget level requires
corresponding bounds on the compute level:

```
presentar Widget:   bound(total_render < 16ms)
  └── trueno GPU:   bound(kernel_dispatch < 2ms)
  └── trueno SIMD:  bound(scalar_fallback < 8ms)
  └── probar:       bound(assertion_check < 1ms)
```

This is a `conservation`-like obligation: the sum of component bounds
must not exceed the parent's budget.

**5. Tracing and profiling contracts (renacer integration).**

Trueno's renacer integration defines golden trace baselines with max
10% deviation. These are `bound` + `old_state` obligations:

```yaml
proof_obligations:
  - type: old_state
    property: "Performance within 10% of golden baseline"
    formal: "|metric(current) - metric(old(golden))| / metric(old(golden)) < 0.10"
  - type: bound
    property: "Matrix operation syscall budget"
    formal: "syscall_count(matrix_ops) ≤ 200"
```

**6. PTX bug detection as contract verification.**

Trueno-explain's `PtxBugClass` classification (SharedMemU64Addressing,
LoopBranchToEnd, MissingBarrierSync) maps directly to `postcondition`
obligations on PTX kernel generation:

```yaml
proof_obligations:
  - type: postcondition
    property: "Generated PTX has no P0 critical bugs"
    formal: "∀ bug ∈ analyze_ptx(output): bug.severity ≠ Critical"
  - type: invariant
    property: "Shared memory accesses use 32-bit addressing"
    formal: "¬∃ instr ∈ ptx: is_shared_mem(instr) ∧ is_64bit_addr(instr)"
```

### 8.7. Implications for the Stack

Extending provable-contracts to non-kernel domains requires:

1. **No schema changes beyond Section 3.** The 7 new obligation types
   and 3 new fields already support all domains above.

2. **New contract tiers.** Add Tier 8+ for non-kernel domains:

   | Tier | Scope | Domain | Primary Consumer |
   |---|---|---|---|
   | Tier 8 | Simulation contracts | Physics conservation, integrators, checkpoints | simular |
   | Tier 9 | IaC contracts | DAG ordering, convergence, drift, transport safety | forjar |
   | Tier 10 | Orchestration contracts | Transpilation semantics, pipeline stages, cost models | batuta |
   | Tier 11 | Code quality contracts | Analysis invariants, TDG scoring, uniform interfaces | pmat |
   | Tier 12 | Media asset contracts | Codec parity, determinism, SVG quality, animation timing | rmedia |
   | Tier 13 | Testing contracts | Coverage, assertions, replay, accessibility | probar |
   | Tier 14 | Presentation contracts | UI layout, accessibility, animation | presentar |
   | Tier 15 | Data pipeline contracts | Schema, transform, quality | — |
   | Tier 16 | API contracts | Protocol, SLA, versioning | — |
   | Tier 17 | Infrastructure contracts | Deployment, resource, scaling | — |

3. **New kernel equivalence classes.** The existing classes (A-E) cover
   ML architectures. Non-kernel domains need their own classification.

4. **Scoring adaptation.** The current 5 scoring dimensions (spec depth,
   falsification, Kani, Lean, binding) apply to non-kernel contracts
   without modification — the *content* of the contracts changes, but
   the quality rubric does not.

5. **Verification ladder adjustment.** L4 (Kani) and L5 (Lean) remain
   applicable but the "natural bound" concept differs:
   - Kernels: natural bound = SIMD width, super-block size
   - Simulation: natural bound = max particles, max timesteps per epoch
   - IaC: natural bound = max resources per config, max DAG depth
   - Media: natural bound = max frame count, max SVG element count, max SRT entries
   - Orchestration: natural bound = max pipeline stages, max converter ops
   - Code quality: natural bound = max files per analysis, max AST depth
   - Testing: natural bound = max DOM depth, max locator chain length
   - Presentation: natural bound = max component tree depth, max children
   - Data: natural bound = max batch size, max column count
   - API: natural bound = max request body size, max concurrent requests

6. **Simular is the nearest adoption target.** It already has 3 YAML
   contracts and a `provable-contracts-macros` dependency. Adding
   `precondition`/`postcondition` pairs to its existing gradient and
   checkpoint contracts, `frame` to its integration step, and
   `loop_invariant`/`loop_variant` to its simulation loops would
   demonstrate the full Eiffel DbC vocabulary on a real, stateful
   system.

7. **Forjar is equally ready.** It already has `#[contract]` annotations
   on DAG ordering, atomic writes, recipe determinism, and codegen
   dispatch. Its planner/executor pipeline is a textbook DbC system
   where desired state = postcondition, current state = old_state, and
   convergence = contract fulfillment. Adding `frame` to its resource
   apply (only the target resource changes), `precondition` to its
   transport safety (bashrs validation), and `old_state` to its drift
   detection (BLAKE3 hash comparison) would complete the picture.

8. **Probar's Brick Architecture is a natural contract surface.** Each
   brick is already a test component with implicit pre/post/frame
   conditions. Extracting these into YAML contracts would make probar
   both a *consumer* of provable-contracts (its own internal
   correctness) and a *tool* for verifying other projects' contracts
   at the property-test level (L3).

---

## 9. Migration Path

### Phase 1: Schema + Validation

1. Add 7 new variants to `ObligationType`
2. Add optional `requires`, `applies_to`, `parent_contract` fields to
   `ProofObligation`
3. Update `pv validate` to enforce field/type constraints
4. Update `pv query --obligation` to filter by new types

**Already done (partial):** Equation-level `preconditions`/
`postconditions` fields exist, `#[requires]`/`#[ensures]`/
`#[invariant]` proc macros exist, `codegen.rs` generates assertion
macros, Gate 5 checks for pre/post on equations. What remains is
adding the `ObligationType` variants and the obligation-level fields.

### Phase 2: Codegen

5. Add probar test generators for each new type (extend `probar_gen/`)
6. Add Kani harness generators for each new type (extend `kani_gen/`)
7. Add Lean 4 theorem generators for each new type (extend `lean_gen/`)
8. Wire obligation-level pre/post into the existing escape-proof
   enforcement pipeline (Stages D-E) alongside equation-level pre/post

### Phase 3: Kernel Adoption

8. Retrofit `precondition`/`postcondition` pairs onto Tier 1 contracts
   (softmax, rmsnorm, rope, silu) as exemplars
9. Add `frame` obligations to KV cache and in-place buffer contracts
10. Add `loop_invariant`/`loop_variant` to iterative contracts
    (online-softmax, adamw, lbfgs, cma-es, pagerank)
11. Add `subcontract` obligations where `depends_on` represents
    behavioral subtyping (GQA→attention, flash-attention→attention)

### Phase 4: Cross-Domain Expansion

12. **simular (Tier 8):** Retrofit existing 3 contracts (checkpoint,
    gradient, loss-functions) with `precondition`/`postcondition` pairs.
    Add `frame` obligations to integration steps. Add `loop_invariant`/
    `loop_variant` to simulation loops. Add `old_state` for energy
    drift tracking.
13. **forjar (Tier 9):** Retrofit existing `#[contract]` annotations
    with full YAML contracts. Add `frame` to resource apply (only
    target resource changes). Add `precondition` to transport safety
    (bashrs validation gate). Add `old_state` to drift detection
    (BLAKE3 hash comparison). Add `subcontract` for pepita→SSH
    transport refinement. Add `loop_invariant`/`loop_variant` to
    DAG wave execution.
14. **batuta (Tier 10):** Write contracts for the transpilation
    pipeline: `postcondition` (output compiles), `equivalence`
    (transpiled semantics match source on test suite), `frame`
    (source files unchanged), `subcontract` (PyTorchConverter
    refines TranspilerPlugin). Write contracts for BackendSelector
    cost model (`bound` on latency, `postcondition` on correctness).
15. **pmat (Tier 11):** Formalize the uniform contracts pattern as
    YAML: `subcontract` (each specialized contract refines base),
    `frame` (analysis is read-only), `determinism` (same input →
    same grade), `invariant` (TDG monotonic). Write contracts for
    the TDG scoring formula and mutation testing threshold.
16. **rmedia (Tier 12):** Extract render pipeline score dimensions as
    YAML contracts. Write contracts for codec parity (`equivalence`
    with melt), determinism (`determinism` — identical hash), SRT lock
    (`old_state` — SHA-256 match), SVG quality floors (`bound` on
    opacity/stroke/font), animation timing (`bound` on SRT alignment
    within ±2 frames), frame pipeline (`loop_invariant` — channel
    depth ≤ 16, `loop_variant` — remaining frames decreasing),
    compositing correctness (`invariant` — YUV420P maintained).
17. **probar (Tier 13):** Extract Brick Architecture implicit contracts
    into YAML. Write contracts for locator resolution (precondition:
    element exists), visual regression (frame: reference image
    unchanged), and playbook replay (determinism, roundtrip).
18. Write exemplar presentation contracts (layout, accessibility,
    animation) for presentar — demonstrate pre/post/frame/old-state
19. Write exemplar data pipeline and API contracts for future consumers
20. Update `pv scaffold` to generate domain-appropriate test skeletons
    (e.g., state snapshot tests for simulation, lock-file assertions
    for IaC, equivalence tests for transpilation, ffprobe assertions
    for media, DOM snapshot tests for presentation)
21. Update `pv generate` to produce deterministic README.md and
    GitHub Actions workflow files for consumer projects
22. Add Tier 8-17 to the contract registry classification

---

## 10. `pv explain` — Contract Narrative Command

Before implementing the 7 new DbC obligation types, we need the
ability to *explain* any contract in detail. The existing commands
provide counts (`pv status`), gap detection (`pv audit`), formulas
(`pv equations`), and reference pages (`pv generate` → `_book.md`).
None produces a narrative explanation of *what the contract means,
why each obligation exists, and how the verification chain works*.

### 10.1. Command Interface

```bash
pv explain <contract.yaml>                              # text narrative
pv explain <contract.yaml> --format markdown             # markdown with LaTeX
pv explain <contract.yaml> --format json                 # structured JSON
pv explain <contract.yaml> --binding binding.yaml        # include binding status
```

### 10.2. Output Structure

`pv explain` renders a **chain-of-thought narrative** organized into
these sections:

**1. What this contract specifies** — One-paragraph summary derived
from `metadata.description` and `metadata.references`. Names the
governing paper(s) and the domain.

**2. Governing equations** — For each equation in `equations`:
- Formula with domain/codomain
- Mathematical invariants (prose, not table)
- Preconditions (if present): what the caller must guarantee
- Postconditions (if present): what the kernel guarantees
- `lean_theorem` reference (if present): link to the L5 proof

**3. Proof obligations** — For each obligation in `proof_obligations`:
- Obligation type with its mathematical pattern
  (e.g., "invariant: ∀x ∈ Domain: P(f(x))")
- Property and formal predicate
- Tolerance (if numeric)
- **Why this matters** — prose explaining the obligation's purpose
- Lean proof status (proved/sorry/wip/not-applicable) with theorem
  name, dependencies, and mathlib imports
- Cross-references to falsification tests and Kani harnesses that
  verify this obligation

**4. Verification ladder** — Summary of verification coverage:
- L5 (Lean): N of M proved (percentage)
- L4 (Kani): N harnesses with strategy breakdown
- L3 (probar): N property tests
- L2 (falsification): N tests
- Overall proof level (L1-L5)

**5. Falsification tests** — For each test:
- Prediction (what should hold)
- Method (how it's tested)
- What it catches (root cause if it fails)
- Explain the Popperian structure: each test tries to *refute* the
  contract, not confirm it

**6. Kani bounded model checking** — For each harness:
- Strategy explanation:
  - `exhaustive`: verify for ALL inputs within bound
  - `stub_float`: assume Lean-proved postconditions on transcendentals,
    verify surrounding code
  - `compositional`: verify sub-kernels separately, compose proofs
  - `bounded_int`: integer-only verification
- Bound and solver

**7. Kernel execution phases** — If `kernel_structure` present:
- Sequential walkthrough of phases with invariants
- Explain how each phase's invariant feeds into the next

**8. SIMD dispatch** — If present: list dispatch targets

**9. Enforcement rules** — If present: list with severity

**10. Quality gate** — If present: explain pass criteria and mutation test

**11. Binding status** — If `--binding` provided:
- Which equations are implemented/partial/missing
- Which project(s) consume this contract

### 10.3. Example: Softmax

```
softmax-kernel-v1 (v1.0.0)
Softmax kernel — numerically stable exponential normalization

What this contract specifies
  This contract governs the softmax function, which normalizes a vector
  of real numbers into a probability distribution. It derives from
  Bridle (1990) and Milakov & Gimelshein (2018).

Governing equations

  softmax
    σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    Domain: x ∈ ℝ^n, n ≥ 1
    Range:  σ(x) ∈ (0,1)^n

    The equation uses the max-subtraction trick for numerical stability:
    subtracting max(x) before exponentiation prevents overflow while
    preserving the result (translation invariance).

    Invariants:
      1. Σ σ(x)_i = 1.0 — outputs form a probability distribution
      2. σ(x)_i > 0 — all outputs strictly positive (exp > 0)
      3. argmax(σ(x)) = argmax(x) — largest input → largest output

Proof obligations (6)

  1. [invariant] Output sums to 1
     Pattern: ∀x ∈ Domain: P(f(x)) — property holds for all inputs
     Formal:  |Σ σ(x)_i - 1.0| < ε  (tolerance: 1e-6)
     Why: The defining property of a probability distribution. If this
       fails, downstream sampling and cross-entropy loss produce garbage.
     Lean: Softmax.partition_of_unity (proved)
       Depends: Real.exp_pos, Finset.sum_div_distrib
       Note: Proof over reals; f32 gap addressed by error-bound lemma.
     Verified at: L2 (FALSIFY-SM-001), L4 (KANI-SM-001), L5 (Lean)

  2. [invariant] All outputs strictly positive
     Pattern: ∀x ∈ Domain: P(f(x))
     Formal:  σ(x)_i > 0 for all i
     Why: Logarithm of softmax output must be defined (log-softmax in
       cross-entropy). Zero outputs cause -inf and NaN propagation.
     Lean: Softmax.softmax_pos (proved)
     Verified at: L2 (FALSIFY-SM-002), L4 (KANI-SM-002), L5 (Lean)

  ...

Verification ladder
  L5 (Lean):  5/6 proved (83%)
  L4 (Kani):  3 harnesses (stub_float strategy)
  L2 (Tests): 6 falsification tests
  Level: L4

Falsification tests (Popperian)
  Each test tries to refute the contract. Survival = evidence for,
  not proof of, correctness.

  FALSIFY-SM-001: Normalization
    Predicts: sum(softmax(x)) ≈ 1.0 for random x ∈ [-1000, 1000]^n
    Method:   proptest with 10000 random vectors, dim 1..128
    Catches:  Missing or incorrect max-subtraction trick

  FALSIFY-SM-004: SIMD equivalence
    Predicts: |softmax_avx2(x) - softmax_scalar(x)| < 8 ULP
    Method:   proptest comparing scalar vs SIMD output
    Catches:  SIMD reduction order differs from scalar

Kani bounded model checking
  KANI-SM-001: verify_softmax_normalization (bound: 8, stub_float)
    Assumes Lean-proved postconditions on exp() (positive, finite),
    then verifies the sum-to-1 invariant holds for ALL vectors of
    length ≤ 8 regardless of exp's exact return value.

Kernel phases
  1. find_max      — Compute max(x) for stability [max ≥ x_i ∀i]
  2. exp_subtract  — exp(x_i - max) per element   [result ∈ (0,1]]
  3. sum_exp       — Σ exp(x_i - max)             [sum > 0]
  4. normalize     — Divide each exp by sum        [output_i = exp_i/sum]

SIMD dispatch
  softmax: scalar → softmax_scalar | avx2 → softmax_avx2 | ptx → softmax_ptx

Enforcement
  normalization — Output must sum to 1.0 (ERROR) → FALSIFY-SM-001
  positivity    — All outputs positive   (ERROR) → FALSIFY-SM-002

Quality gate: F-SM-001 Softmax Contract
  Pass: All 6 falsification tests + Kani harnesses verify
  Mutation: Introduce off-by-one in max reduction loop
```

### 10.4. Implementation

**Library module:** `crates/provable-contracts/src/explain.rs`

```rust
pub fn explain_contract(
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) -> String
```

Reuses existing building blocks:
- `schema::parse_contract()` — parse YAML
- `proof_status::compute_proof_level()` — L1-L5 level
- `probar_gen::obligation_pattern()` — pattern strings per type
- `binding::parse_binding()` — binding registry
- `latex::math_to_latex()` — math rendering (markdown format)

**CLI handler:** `crates/provable-contracts-cli/src/commands/explain.rs`

Adds `Explain` variant to `Commands` enum with `contract: PathBuf`,
`--format text|markdown|json`, and `--binding` option.

### 10.5. Relationship to DbC Types

When the 7 new obligation types are implemented, `pv explain` will
render their narratives using the same pattern:

| DbC Type | Explain Narrative |
|---|---|
| `precondition` | "Caller must guarantee: [formal]. If violated, kernel behavior is undefined." |
| `postcondition` | "Kernel guarantees: [formal], conditional on precondition [requires] holding." |
| `frame` | "This operation modifies [modifies set] only. All other state is preserved." |
| `loop_invariant` | "At every iteration of [applies_to]: [formal]. This is maintained inductively." |
| `loop_variant` | "Termination witness: [formal]. Strictly decreasing, proving the loop exits." |
| `old_state` | "Relates pre-call state to post-call state: [formal]." |
| `subcontract` | "This contract refines [parent_contract]. Preconditions weakened, postconditions strengthened." |

This makes `pv explain` the primary user-facing tool for understanding
contracts — including the new Eiffel DbC types as they are adopted.

### 10.6. Generated Artifacts: README.md and CI Workflows

`pv generate` already produces per-contract Rust artifacts (`_scaffold.rs`,
`_kani.rs`, `_probar.rs`, `_book.md`). It should also generate project-
level artifacts that consumer projects can adopt:

**`pv generate --readme`** — generates a `README.md` for a consumer project
that documents its contract coverage:

```bash
pv generate contracts/ --readme --binding contracts/aprender/binding.yaml \
    --output aprender/
```

Produces `aprender/CONTRACT-README.md` containing:

1. **Contract coverage badge** — "28/31 contracts bound (90.3%)"
2. **Bound contracts table** — contract stem, equation, binding function,
   proof level (L1-L5), Lean status
3. **Verification ladder summary** — how many obligations at each level
4. **Gap list** — unbound equations with priority and suggested action
5. **Build integration** — `build.rs` setup instructions for
   `#[contract]` enforcement
6. **CI integration** — reference to generated workflow file

**`pv generate --ci`** — generates a GitHub Actions workflow for contract
validation in the consumer project:

```bash
pv generate contracts/ --ci --output .github/workflows/
```

Produces `.github/workflows/contracts.yml`:

```yaml
name: Contract Validation
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    name: Contract Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/checkout@v4
        with:
          repository: paiml/provable-contracts
          path: provable-contracts

      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Install pv
        run: cargo install --path provable-contracts/crates/provable-contracts-cli

      - name: Validate contracts
        run: pv validate provable-contracts/contracts/*.yaml

      - name: Lint contracts
        run: pv lint provable-contracts/contracts/ --min-score 0.60

      - name: Verify bindings
        run: pv lint provable-contracts/contracts/
             --binding provable-contracts/contracts/$PROJECT/binding.yaml

      - name: Check Lean theorems (no sorry)
        run: |
          cd provable-contracts/lean
          lake build
          # Verify no sorry in proved theorems
          grep -r "sorry" ProvableContracts/Theorems/ && exit 1 || true

      - name: Run falsification tests
        run: cargo test --workspace
```

The workflow enforces the escape-proof pipeline at CI level:
- Gate C (YAML validation): `pv validate` + `pv lint`
- Gate D (binding check): `pv lint --binding`
- Gate B (Lean no-sorry): `lake build` + sorry grep
- Gate F (tests): `cargo test`

This makes contract enforcement automatic for any project that consumes
provable-contracts — no manual CI configuration needed.

**Determinism invariant:** All generated artifacts (README.md, CI
workflows, Rust files) must be **fully deterministic**. Same inputs →
identical byte-for-byte output. No timestamps, no git hashes, no
random IDs, no `created:` dates, no environment-dependent values.
This is a `determinism` obligation on the generator itself:

```yaml
proof_obligations:
  - type: determinism
    property: "pv generate output is deterministic"
    formal: "generate(contracts, binding) = generate(contracts, binding)"
```

Running `pv generate` twice with the same contracts and binding
must produce identical files. This enables `git diff` to detect real
changes (not regeneration noise) and makes CI caching effective.

**Cross-project CI matrix** — for the full stack, the generated workflow
can include a matrix strategy testing all bound projects:

```yaml
strategy:
  matrix:
    project: [aprender, entrenar, realizar, trueno, forjar, simular]
```

Each matrix entry validates that project's binding.yaml against the
contracts, ensuring no project ships with broken bindings.

---

## 11. Escape-Proof Enforcement by Domain

The escape-proof enforcement pipeline
([escape-proof-enforcement.md](escape-proof-enforcement.md)) defines
six stages where each gates the next. This section traces how each
Eiffel DbC obligation type flows through the pipeline, and how each
domain in the stack concretely enforces escape prevention at compile
time.

### 11.1. The Six Stages, Revisited for DbC Types

```
A. Equation YAML        equations.<name>.preconditions / postconditions
   ↓ must exist              ↓ formal predicates in proof_obligations[]
B. Lean 4 Proof          theorem proves the obligation over ℝ
   ↓ no sorry                ↓ precondition becomes hypothesis
C. YAML Validation       pv lint Gate 5: equations have pre/post/lean_theorem
   ↓ gates pass              ↓ Gate 1-4: obligations ↔ harnesses consistent
D. build.rs Codegen      generates CONTRACT_*_PRE_N / CONTRACT_*_POST_N env vars
   ↓ env vars set            ↓ generates debug_assert!() from Rust expressions
E. #[contract] Macro     reads env vars, injects debug_assert!() at entry/exit
   ↓ compile-time check      ↓ #[requires], #[ensures], #[invariant] macros
F. Test Execution        falsification tests + probar + Kani harnesses
   ↓ zero failures           ↓ cargo test blocks merge on any failure
```

For the five NEW DbC types, the pipeline extends as follows:

| DbC Type | Stage A (YAML) | Stage B (Lean) | Stage D (build.rs) | Stage E (Macro) | Stage F (Test) |
|---|---|---|---|---|---|
| `precondition` | `equations.<eq>.preconditions[]` | Hypothesis `h : P x` | `CONTRACT_*_PRE_N` env vars | `debug_assert!(pre)` at entry | Negative test: violate pre, assert panic |
| `postcondition` | `equations.<eq>.postconditions[]` | Goal `post (f x)` | `CONTRACT_*_POST_N` env vars | `let ret = {...}; debug_assert!(post)` | Conditional test: assume pre, assert post |
| `frame` | `proof_obligations[].type: frame` | `∧ ∀i, old[i] = new[i]` | Clone-and-compare codegen | Snapshot inputs pre-call, assert unchanged post-call | Mutation: corrupt input, assert frame test catches |
| `loop_invariant` | `proof_obligations[].type: loop_invariant` | Induction over iterations | Assert-per-iteration codegen | `debug_assert!(inv)` inside unrolled loop | Proptest: verify invariant at each step |
| `loop_variant` | `proof_obligations[].type: loop_variant` | Well-founded recursion | Assert-decreasing codegen | `debug_assert!(v_after < v_before)` | Proptest: verify strictly decreasing |
| `old_state` | `proof_obligations[].type: old_state` | Universally quantified pre-state | Clone-before codegen | `let old = state.clone(); ...; debug_assert!(Q(old, state))` | Snapshot-compare: assert relationship |
| `subcontract` | `proof_obligations[].type: subcontract` | Implication chain | Cross-contract env var check | Parent contract binding must exist | Load both contracts, verify weakening/strengthening |

### 11.2. Chain of Thought: Kernels (aprender, trueno, entrenar)

**Domain axiom:** `softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))`

**Stage A — Equation YAML.** The contract author writes:
```yaml
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))"
    preconditions:
      - "!logits.is_empty()"
      - "logits.iter().all(|x| x.is_finite())"
    postconditions:
      - "ret.len() == logits.len()"
      - "(ret.iter().sum::<f32>() - 1.0).abs() < 1e-6"
    lean_theorem: "ProvableContracts.Theorems.Softmax.PartitionOfUnity"
```
**Chain of thought:** The preconditions encode the *caller's* responsibility
(non-empty, finite input). The postconditions encode the *kernel's*
guarantee (output length matches, sums to 1). The formula is the
mathematical ground truth. The `lean_theorem` links to the L5 proof.

**Stage B — Lean 4 Proof.** `build.rs` verifies:
1. `PartitionOfUnity` exists in `lean/ProvableContracts/Theorems/Softmax/`
2. No `sorry` in the theorem or its dependencies
3. If either fails → `compile_error!` — the binary cannot be built

**Chain of thought:** This is Meyer's *seamless development* — the math
proof IS the gate. A developer cannot ship softmax without the
theorem being proved. The Lean proof takes the precondition as a
hypothesis: `theorem partition_of_unity (h : ∀ i, x[i].is_finite) : ...`

**Stage C — YAML Validation.** `pv lint` checks:
- Gate 1: YAML parses against schema (preconditions field exists)
- Gate 4: Falsification test IDs resolve to `fn test_*` in source
- Gate 5: Equation has preconditions, postconditions, lean_theorem

**Chain of thought:** If someone deletes a precondition from the YAML,
Gate 5 warns. If someone deletes a falsification test, Gate 4 errors.
The YAML contract is the single source of truth — tampering is visible.

**Stage D — build.rs Codegen.** `build.rs` reads the YAML and sets:
```
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=bound
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_COUNT=2
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_0=!logits.is_empty()
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_1=logits.iter().all(|x| x.is_finite())
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_COUNT=2
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_0=ret.len() == logits.len()
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_1=(ret.iter().sum::<f32>() - 1.0).abs() < 1e-6
```
**Chain of thought:** The env vars are the *bridge* between YAML and
Rust. They carry the precondition/postcondition strings from the
contract into the compiler's environment. If the YAML is deleted,
these vars are missing, and Stage E fails.

**Stage E — #[contract] Macro.** In aprender's softmax implementation:
```rust
#[contract("softmax-kernel-v1", equation = "softmax")]
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    // Macro expands to:
    // 1. const _: Option<&str> = option_env!("CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX");
    // 2. debug_assert!(!logits.is_empty(), "Pre-condition violated: ...");
    //    debug_assert!(logits.iter().all(|x| x.is_finite()), "...");
    // 3. let ret = { /* original body */ };
    // 4. debug_assert!(ret.len() == logits.len(), "Post-condition violated: ...");
    //    debug_assert!((ret.iter().sum::<f32>() - 1.0).abs() < 1e-6, "...");
    // 5. ret
}
```
**Chain of thought:** The macro reads the env vars at *compile time*
and generates `debug_assert!` calls. In debug builds, every call to
`softmax_1d` checks the preconditions on entry and postconditions on
exit. In release builds, `debug_assert!` expands to nothing — zero
runtime cost. If someone removes the `#[contract]` attribute, `pv lint`
Gate 4 catches the missing test reference. If someone deletes the
YAML, `option_env!` returns `None` and the binding const fails.

**Stage F — Test Execution.** `cargo test` runs:
- FALSIFY-SM-001: proptest with 10000 random vectors, verify sum ≈ 1.0
- FALSIFY-SM-004: proptest comparing scalar vs AVX2 output within 8 ULP
- Kani harnesses: `verify_softmax_normalization` (bounded ≤16 elements)

**Chain of thought:** The falsification tests are the Popperian gate —
they try to *refute* the postcondition. If any test finds an input
where sum ≠ 1.0, the contract is violated and `cargo test` fails.
CI blocks the merge. The kernel cannot ship with a broken postcondition.

**Escape analysis for kernels:**

| Attempted Escape | Gate That Catches It |
|---|---|
| Delete precondition from YAML | Gate 5: equation without precondition → WARNING |
| Delete `#[contract]` from function | Gate 4: test reference dangling → ERROR |
| Delete YAML file entirely | build.rs: `CONTRACT_*` env var missing → compile error |
| Leave `sorry` in Lean proof | build.rs: sorry detection → compile error |
| Weaken postcondition | Lean proof no longer matches → sorry required → compile error |
| Skip `cargo test` | CI workflow requires test pass |

### 11.3. Chain of Thought: Simulation (simular)

**Domain axiom:** `E(t+dt) = E(t) + O(dt^4)` (symplectic energy conservation)

**Stage A — Equation YAML.** Simular's checkpoint contract:
```yaml
equations:
  checkpoint_roundtrip:
    formula: "deserialize(serialize(state)) = state"
    preconditions:
      - "state.energy.is_finite()"
      - "state.particles.iter().all(|p| p.position.is_finite())"
    postconditions:
      - "(restored.energy - state.energy).abs() < f64::EPSILON"
    lean_theorem: "ProvableContracts.Theorems.Checkpoint.Roundtrip"
```
**Chain of thought:** Simulation preconditions constrain *physical
plausibility* (finite energy, finite positions). The postcondition
asserts *checkpoint fidelity* — serialization must not lose precision.
This is where `old_state` becomes essential: the postcondition
references `state` (the pre-call value) and `restored` (the post-call
value). Without `old_state` as a first-class concept, this relationship
is awkward to express.

**NEW: Frame obligation for integration step:**
```yaml
proof_obligations:
  - type: frame
    property: "Integration step modifies positions and velocities only"
    formal: "modifies(pos, vel) ∧ preserves(mass, G, dt)"
```
**Stage D codegen** for frame obligations generates:
```rust
// build.rs generates:
let old_mass = state.mass.clone();
let old_g = state.gravitational_constant;
// ... run integration step ...
debug_assert!(state.mass == old_mass, "Frame violated: mass changed");
debug_assert!(state.gravitational_constant == old_g, "Frame violated: G changed");
```
**Chain of thought:** The frame obligation is *critical* for simular
because the jidoka module (runtime anomaly detection) already catches
NaN and energy drift — but it cannot catch *silent parameter
corruption* where a bug modifies the gravitational constant during
integration. The frame obligation makes this a compile-time-enforced
invariant: if the integration step touches `G`, `debug_assert!` fires
in debug builds, and Kani can verify it statically.

**NEW: Loop invariant for N-body simulation:**
```yaml
proof_obligations:
  - type: loop_invariant
    property: "Total momentum conserved at every timestep"
    formal: "∀k ≤ i: |Σ m_j·v_j(k) - Σ m_j·v_j(0)| < ε_drift × k"
    applies_to: "nbody.integrate"
```
**Stage D codegen** generates an assertion inside the simulation loop:
```rust
for step in 0..n_steps {
    integrate_step(&mut state);
    debug_assert!(
        (total_momentum(&state) - initial_momentum).norm() < DRIFT_TOL * step as f64,
        "Loop invariant violated: momentum drift at step {step}"
    );
}
```
**Chain of thought:** This replaces simular's runtime jidoka check with
a *declarative contract*. The jidoka module catches drift at runtime
and halts — the loop invariant obligation lets Kani verify *before
deployment* that drift is bounded for all inputs within the natural
bound (max particles × max timesteps).

### 11.4. Chain of Thought: Infrastructure as Code (forjar)

**Domain axiom:** `converge(desired, current) → desired` (idempotent convergence)

**Stage A — Equation YAML.** Forjar's DAG ordering contract:
```yaml
equations:
  topological_sort:
    formula: "∀ (a,b) ∈ E: order(a) < order(b)"
    preconditions:
      - "!graph.has_cycle()"
      - "graph.node_count() > 0"
    postconditions:
      - "result.len() == graph.node_count()"
      - "result.windows(2).all(|w| !graph.has_edge(w[1], w[0]))"
    lean_theorem: "ProvableContracts.Theorems.DAG.TopologicalSort"
```
**Chain of thought:** IaC preconditions enforce *infrastructure
validity* — the resource dependency graph must be acyclic. This is
where precondition-as-obligation-type adds value: the *formal
predicate* `¬∃ cycle in G` can be verified by Kani for all graphs
up to N nodes, while the Rust expression `!graph.has_cycle()` is
a runtime check only.

**NEW: Frame obligation for resource apply:**
```yaml
proof_obligations:
  - type: frame
    property: "Applying resource R modifies only R's lock entry"
    formal: "modifies(lock[R]) ∧ preserves(lock[S] ∀ S ≠ R)"
```
**Stage D codegen** for forjar's apply:
```rust
// build.rs generates for each resource apply:
let old_lock = lock_state.clone();
apply_resource(&mut lock_state, resource_r);
for (name, entry) in &old_lock.entries {
    if name != resource_r.name() {
        debug_assert!(
            lock_state.entries[name] == *entry,
            "Frame violated: resource {} modified by apply of {}",
            name, resource_r.name()
        );
    }
}
```
**Chain of thought:** This is the *most important* DbC type for IaC.
Forjar's jidoka policy (stop on first failure, preserve partial state)
is a runtime enforcement of the frame condition. But without a frame
*contract*, a subtle bug where applying package `nginx` accidentally
modifies the lock entry for service `postgres` would go undetected
until production. The frame obligation makes this a compile-time-
verifiable property: Kani can exhaustively verify that `apply_resource`
only touches the target entry for all resource configurations within
the natural bound.

**NEW: Old-state for drift detection:**
```yaml
proof_obligations:
  - type: old_state
    property: "Drift = hash mismatch between lock and current state"
    formal: "drift(R) ↔ hash(current(R)) ≠ hash(old(lock(R)))"
```
**Stage E macro** on forjar's drift detection:
```rust
#[contract("drift-detection-v1", equation = "drift_hash")]
pub fn detect_drift(lock: &LockState, current: &SystemState) -> Vec<DriftResult> {
    // Macro injects:
    // PRE: debug_assert!(lock.entries.iter().all(|e| e.hash.len() == 32));
    // POST: debug_assert!(ret.iter().all(|d| d.resource_exists_in_lock));
}
```
**Chain of thought:** Forjar's tripwire module already computes
`hash(current) vs hash(lock)` — but the *contract* for what
constitutes drift is implicit in code. Making it an `old_state`
obligation means `pv lint` validates the contract exists, `build.rs`
generates assertions, and Kani can verify that the hash comparison
is correct for all states within bounds. The `old_state` type is
the formal language for "relating pre-state to post-state" — which
is exactly what drift detection does.

**NEW: Subcontract for transport substitutability:**
```yaml
proof_obligations:
  - type: subcontract
    property: "Pepita transport refines SSH transport"
    formal: "pre(SSH) → pre(Pepita) ∧ post(Pepita) → post(SSH)"
    parent_contract: "ssh-transport-v1"
```
**Stage F test** — cross-contract verification:
```rust
#[test]
fn pepita_refines_ssh() {
    // Load both contracts
    let ssh = parse_contract("ssh-transport-v1.yaml");
    let pepita = parse_contract("pepita-transport-v1.yaml");
    // Verify: every script accepted by SSH is also accepted by Pepita
    proptest!(|(script in valid_ssh_scripts())| {
        assert!(pepita.accepts(&script), "Pepita rejects script SSH accepts");
    });
    // Verify: Pepita's postconditions imply SSH's
    proptest!(|(script in valid_ssh_scripts(), state in any_state())| {
        let pepita_result = pepita.execute(script, state);
        assert!(ssh.postcondition_holds(&pepita_result),
            "Pepita result violates SSH postcondition");
    });
}
```
**Chain of thought:** This is Meyer's subcontracting rule made
operational. If a future forjar developer adds a new pepita check
that rejects a script SSH would accept (strengthening the precondition),
the cross-contract test fails immediately. The `subcontract` obligation
type makes the Liskov relationship *machine-checkable* — not just
documented in a comment.

### 9.5. Chain of Thought: Orchestration (batuta)

**Domain axiom:** `eval(transpile(source), input) = eval(source, input)` (semantic equivalence)

**Stage A — Equation YAML:**
```yaml
equations:
  transpile_equivalence:
    formula: "∀ input: eval(transpile(py_source), input) = eval(py_source, input)"
    preconditions:
      - "source.is_valid_python()"
      - "source.uses_only_supported_ops()"
    postconditions:
      - "output.compiles_as_rust()"
      - "output.function_count() == source.function_count()"
    lean_theorem: "ProvableContracts.Theorems.Transpile.Equivalence"
```
**Chain of thought:** Transpilation is the domain where postconditions
are *hardest to enforce* because semantic equivalence is undecidable
in general. The precondition narrows the domain to supported ops, and
the postcondition checks *structural* properties (compiles, same
function count) that are decidable. The full semantic equivalence
is verified at Stage F via test-suite comparison, not compile-time
assertion.

**NEW: Frame for transpilation:**
```yaml
proof_obligations:
  - type: frame
    property: "Transpilation does not modify source files"
    formal: "modifies(output_dir) ∧ preserves(source_dir)"
```
**Stage D codegen:**
```rust
let source_hash_before = blake3_hash_dir(&source_dir);
let result = transpile(&source_dir, &output_dir);
debug_assert!(
    blake3_hash_dir(&source_dir) == source_hash_before,
    "Frame violated: transpilation modified source directory"
);
```
**Chain of thought:** This is defensive but critical. A transpiler bug
that *modifies the source while reading it* would be catastrophic —
the user loses their original code. The frame obligation makes this
impossible to ship: `debug_assert!` catches it in testing, Kani can
verify it for bounded inputs.

**NEW: Loop invariant for pipeline stages:**
```yaml
proof_obligations:
  - type: loop_invariant
    property: "Pipeline context valid at each stage transition"
    formal: "∀k ≤ i: context.validate() == Ok(()) after stage k"
    applies_to: "pipeline.execute"
```
**Stage E** — the pipeline's `execute()` method:
```rust
#[contract("pipeline-v1", equation = "pipeline_stages")]
pub async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
    for stage in &self.stages {
        ctx = stage.execute(ctx).await?;
        // Macro inserts: debug_assert!(ctx.validate().is_ok(),
        //     "Loop invariant violated after stage {}", stage.name());
    }
    Ok(ctx)
}
```
**Chain of thought:** Batuta's `PipelineStage` trait already has a
`validate()` method — the loop invariant obligation makes the call to
`validate()` a *contractual requirement*, not an optional check. If
a new stage is added that breaks the context between stages, the
debug assertion fires immediately. This is Meyer's seamless development:
the Jidoka "stop on error" principle is *formalized as a loop invariant*.

### 9.6. Chain of Thought: Code Quality (pmat)

**Domain axiom:** `analyze(codebase) → metrics` where `metrics` are
deterministic, complete, and read-only.

**Stage A — Equation YAML:**
```yaml
equations:
  tdg_score:
    formula: "TDG = w₁·cyclomatic + w₂·cognitive + w₃·coverage + w₄·docs + w₅·satd + w₆·duplication"
    preconditions:
      - "path.exists()"
      - "path.is_dir()"
    postconditions:
      - "score >= 0.0 && score <= 100.0"
      - "grade.is_valid()"
    lean_theorem: "ProvableContracts.Theorems.TDG.BoundedScore"
```
**Chain of thought:** Code quality analysis has a surprising
*mathematical* foundation — the TDG formula is a weighted linear
combination. The postcondition (`score ∈ [0, 100]`) is provable
by Lean if each component is bounded. The precondition (`path exists`)
is the caller's responsibility.

**NEW: Frame for read-only analysis:**
```yaml
proof_obligations:
  - type: frame
    property: "Analysis never modifies the analyzed codebase"
    formal: "preserves(codebase_dir)"
```
**Stage D codegen:**
```rust
let codebase_hash_before = blake3_hash_dir(&path);
let result = analyze_tdg(&path);
debug_assert!(
    blake3_hash_dir(&path) == codebase_hash_before,
    "Frame violated: analysis modified codebase"
);
```
**Chain of thought:** An analysis tool that modifies the code it
analyzes would be a catastrophic trust violation. The frame obligation
makes this *contractually impossible* in debug builds. This is the
Hippocratic principle: "first, do no harm" — formalized as a DbC
frame condition.

**NEW: Subcontract for uniform interface contracts:**
```yaml
proof_obligations:
  - type: subcontract
    property: "AnalyzeComplexityContract refines BaseAnalysisContract"
    formal: "fields(Base) ⊂ fields(Complexity) ∧ semantics(Base) preserved"
    parent_contract: "base-analysis-v1"
```
**Stage F test:**
```rust
#[test]
fn complexity_contract_refines_base() {
    // Every field in BaseAnalysisContract exists in AnalyzeComplexityContract
    let base_fields = BaseAnalysisContract::field_names();
    let complexity_fields = AnalyzeComplexityContract::field_names();
    for field in &base_fields {
        assert!(complexity_fields.contains(field),
            "Subcontract violated: base field {} missing from complexity contract", field);
    }
}
```
**Chain of thought:** PMAT's uniform contracts pattern — where every
specialized contract `#[serde(flatten)]`s the base — is a structural
invariant that could break silently if someone removes a base field
from a specialization. The `subcontract` obligation makes this a
tested property. If a developer removes `format` from
`AnalyzeComplexityContract`, the test fails immediately.

### 9.7. Chain of Thought: Testing (probar)

**Domain axiom:** `assert(locator.find(element)) → element.exists_in_DOM`

**Stage A — Equation YAML:**
```yaml
equations:
  locator_resolution:
    formula: "find(selector, DOM) → element | timeout"
    preconditions:
      - "browser.is_connected()"
      - "!selector.is_empty()"
    postconditions:
      - "result.is_some() → result.unwrap().matches(selector)"
      - "result.is_none() → elapsed >= timeout"
```
**Chain of thought:** Testing framework contracts are *meta-contracts*
— they specify the behavior of the tool that checks other tools'
behavior. The precondition (browser connected) is the caller's
responsibility. The postcondition says: either we found the element
and it matches, or we timed out. There is no third state.

**NEW: Frame for visual regression:**
```yaml
proof_obligations:
  - type: frame
    property: "Visual regression comparison preserves reference image"
    formal: "modifies(diff_buffer) ∧ preserves(reference_image)"
```
**Stage D codegen:**
```rust
let ref_hash_before = blake3_hash(&reference_path);
let diff = visual_compare(&reference_path, &screenshot);
debug_assert!(
    blake3_hash(&reference_path) == ref_hash_before,
    "Frame violated: visual regression modified reference image"
);
```
**Chain of thought:** A visual regression tool that accidentally
overwrites the reference image when comparing would silently accept
all future regressions. The frame obligation prevents this: the
reference image must be byte-identical before and after comparison.

### 9.8. Chain of Thought: Infrastructure as Code — Forjar Transport

**Domain axiom:** `∀ script: SSH.accepts(script) → Pepita.accepts(script)` (Liskov)

This is a deeper trace through `subcontract` enforcement specifically,
showing how Meyer's subcontracting rules map to the compile pipeline.

**Stage A — Two YAML contracts with parent-child relationship:**
```yaml
# ssh-transport-v1.yaml
equations:
  exec_script:
    preconditions:
      - "!script.is_empty()"
      - "script.is_valid_posix()"
    postconditions:
      - "exit_code.is_some()"

# pepita-transport-v1.yaml
metadata:
  depends_on: ["ssh-transport-v1"]
equations:
  exec_script:
    preconditions:
      - "!script.is_empty()"         # Same (not strengthened)
      # NOTE: script.is_valid_posix() NOT required — Pepita accepts more
    postconditions:
      - "exit_code.is_some()"        # Same (not weakened)
      - "namespace.is_isolated()"    # Added (strengthened)
```
**Chain of thought:** Pepita *weakens* the precondition (doesn't
require POSIX validation — it runs in a namespace so non-POSIX is
safe) and *strengthens* the postcondition (adds namespace isolation
guarantee). This is exactly Meyer's `require else` / `ensure then`
pattern.

**Stage C — Validation:** `pv validate` checks:
1. `pepita-transport-v1.yaml` declares `depends_on: [ssh-transport-v1]`
2. The `subcontract` obligation references `parent_contract: ssh-transport-v1`
3. The parent contract exists in the registry

**Stage D — build.rs:** Generates cross-contract assertions:
```
CONTRACT_PEPITA_TRANSPORT_V1_EXEC_SCRIPT_PARENT=ssh-transport-v1
```

**Stage E — #[contract] macro** on pepita's `exec_script`:
```rust
#[contract("pepita-transport-v1", equation = "exec_script")]
pub fn exec_script(script: &str) -> ExecOutput {
    // Macro injects pepita's weaker preconditions
    // AND verifies parent contract binding exists
}
```

**Stage F — Cross-contract falsification test:**
```rust
#[test]
fn pepita_never_rejects_what_ssh_accepts() {
    proptest!(|(script in ssh_valid_scripts())| {
        // If SSH accepts this script, Pepita MUST also accept it
        assert!(pepita_accepts(&script),
            "Liskov violation: Pepita rejects '{}' but SSH accepts it", script);
    });
}
```

**Escape analysis for subcontracting:**

| Attempted Escape | Gate That Catches It |
|---|---|
| Pepita adds a precondition SSH doesn't have | Stage F: cross-contract proptest fails |
| Pepita removes a postcondition SSH guarantees | Stage F: cross-contract proptest fails |
| Developer removes `depends_on` | Stage C: `pv validate` — `parent_contract` references missing dep |
| Developer removes subcontract obligation | Stage C: `pv lint` — consistency check fails |

### 9.9. Summary: Escape Prevention by Stage

| Stage | What It Prevents | Enforcement Mechanism | Runtime Cost |
|---|---|---|---|
| A (YAML) | Missing specification | Schema validation, required fields | Zero |
| B (Lean) | Unproved mathematics | `sorry` detection → `compile_error!` | Zero |
| C (Lint) | Inconsistent contracts | `pv lint` non-zero exit → CI blocks | Zero |
| D (build.rs) | Missing assertions | Env var generation from YAML | Zero |
| E (Macro) | Unbound functions | `option_env!` check → `compile_error!` | Zero (debug_assert) |
| F (Tests) | Incorrect behavior | Falsification tests → CI blocks | Test-time only |

**Total runtime cost in release binary: zero.** The proof exists
in the build artifacts, not the shipped code. Like SPARK/Ada's
proof discharge, the verification is *consumed* during compilation
and leaves no trace in the deployed binary.

---

## 12. Falsification

Every claim in this spec must be falsifiable. The following tests can
refute the spec's core hypotheses. They follow the project's standard
Popperian pattern: prediction → test → if_fails.

### 12.1. Hypothesis: DbC Types Add Verification Power

**H1: Precondition/postcondition obligation types produce Kani
harnesses that catch bugs undetectable by property-only types.**

```yaml
- id: FALSIFY-DBC-001
  rule: "Pre/post catches what invariant misses"
  prediction: >
    A kernel with a precondition obligation (input finite, non-empty)
    produces a Kani harness with kani::assume(pre) that catches an
    empty-slice panic unreachable via invariant-only harnesses
  test: >
    Generate Kani harnesses for softmax-kernel-v1 with and without
    precondition obligations. Introduce a deliberate empty-slice bug.
    Verify the precondition harness catches it, invariant-only does not.
  if_fails: >
    Precondition obligation type adds no verification power beyond
    existing invariant type — remove from spec
```

**H2: Frame obligations detect state corruption undetectable by
conservation obligations.**

```yaml
- id: FALSIFY-DBC-002
  rule: "Frame detects what conservation misses"
  prediction: >
    A frame obligation (modifies output only) on kv-cache-equivalence-v1
    produces a probar test that catches input buffer corruption, while
    conservation (sum preserved) does not detect it
  test: >
    Generate tests for kv-cache with frame vs conservation-only
    obligations. Introduce a bug that corrupts input but preserves
    sum. Verify frame test catches it, conservation does not.
  if_fails: >
    Frame obligation adds no detection power beyond conservation —
    merge into conservation or remove
```

**H3: Loop invariant/variant obligations produce stronger termination
proofs than bare termination type.**

```yaml
- id: FALSIFY-DBC-003
  rule: "Loop variant proves termination with witness"
  prediction: >
    A loop_variant obligation on adamw-kernel-v1 produces a Kani
    harness that verifies decreasing iteration count, while a bare
    termination obligation produces only an assertion that the loop
    exits (no witness)
  test: >
    Generate Kani harnesses for adamw with loop_variant vs termination.
    Introduce a bug where the loop counter wraps around (non-decreasing).
    Verify loop_variant catches it, termination does not.
  if_fails: >
    Loop variant witness provides no advantage over bare termination
    assertion — simplify to termination only
```

### 12.2. Hypothesis: Two-Layer Pre/Post Model Is Necessary

**H4: Obligation-level pre/postconditions provide value beyond
equation-level debug_assert pre/postconditions.**

```yaml
- id: FALSIFY-DBC-004
  rule: "Formal predicates vs Rust expressions"
  prediction: >
    At least one obligation-level precondition (formal predicate)
    can be verified by Kani but cannot be expressed as a valid
    Rust debug_assert! expression in the equation-level field
  test: >
    Attempt to express all proposed precondition formal predicates
    as valid Rust expressions. Count how many require quantifiers
    (∀, ∃), mathematical notation, or cross-equation references
    that have no Rust equivalent.
  if_fails: >
    Every formal predicate is expressible as Rust debug_assert —
    obligation-level pre/post is redundant, equation-level suffices
```

### 12.3. Hypothesis: DbC Types Apply to Non-Kernel Domains

**H5: The frame obligation type is useful for at least 3 non-kernel
stack projects (simular, forjar, probar).**

```yaml
- id: FALSIFY-DBC-005
  rule: "Frame applicability across domains"
  prediction: >
    Writing a frame obligation for each of simular (integration step
    preserves masses), forjar (resource apply preserves other resources),
    and probar (visual regression preserves reference image) produces
    meaningful probar tests that detect real mutation bugs
  test: >
    Write 3 frame contracts (one per project). Run mutation testing
    (pmat mutate). Verify each frame test kills at least 1 mutant
    that no existing test catches.
  if_fails: >
    Frame obligations in non-kernel domains are tautological — they
    catch no bugs beyond what existing tests already cover
```

**H6: The subcontract obligation type detects Liskov violations in
the stack's transport/plugin abstractions.**

```yaml
- id: FALSIFY-DBC-006
  rule: "Subcontract detects substitution bugs"
  prediction: >
    A subcontract obligation (pepita refines SSH transport in forjar)
    produces a test that catches a pepita-only bug where the precondition
    is accidentally strengthened (rejecting scripts SSH accepts)
  test: >
    Write subcontract obligation for forjar pepita→SSH. Generate
    cross-contract test. Introduce a pepita bug that rejects valid
    SSH scripts. Verify the subcontract test catches it.
  if_fails: >
    Subcontract obligations are documentation-only — they produce
    no tests that detect real behavioral subtyping violations
```

### 12.4. Hypothesis: The Heat Map Predicts Adoption Value

**H7: Domains rated "High" for a DbC type benefit measurably more
from that type than domains rated "Low" or "Medium".**

```yaml
- id: FALSIFY-DBC-007
  rule: "Heat map predicts bug detection"
  prediction: >
    After implementing DbC types across 3+ projects, the number of
    unique bugs caught per obligation (bugs / contract) is at least
    2× higher in domains rated "High" vs domains rated "Low/Medium"
    for that obligation type
  test: >
    Track bugs caught by each new DbC obligation type across projects.
    After 6 months of adoption, compute bugs/contract ratio per
    domain-type pair. Compare High vs Low/Medium groups.
  if_fails: >
    The heat map does not predict adoption value — obligation type
    utility is domain-independent, and the per-domain recommendations
    should be removed
```

---

## 13. References

### Design by Contract Foundations

1. Meyer, B. (1988). *Object-Oriented Software Construction.* Prentice Hall.
2. Meyer, B. (1992). "Applying Design by Contract." *IEEE Computer* 25(10).
3. Meyer, B. (1997). *Object-Oriented Software Construction.* 2nd ed.
   Prentice Hall. Ch. 11 (DbC), Ch. 16 (Inheritance and contracts),
   Ch. 25 (GUI contracts via EiffelVision), Ch. 30 (Concurrency).
4. Meyer, B. (2009). *Touch of Class: Learning to Program Well.* Springer.
5. Meyer, B. (2022). "The Dependent Delegate Dilemma." *CACM* 65(4).
6. Liskov, B. & Wing, J. (1994). "A Behavioral Notion of Subtyping." *ACM TOPLAS* 16(6).
7. Findler, R.B. & Felleisen, M. (2002). "Contracts for Higher-Order Functions." *ICFP 2002.*
8. Hoare, C.A.R. (1969). "An Axiomatic Basis for Computer Programming." *CACM* 12(10).
9. Parnas, D.L. (1972). "On the Criteria to Be Used in Decomposing Systems into Modules." *CACM* 15(12).

### Domain-Specific Contracts

10. Barnett, M. et al. (2004). "Spec#: A Language for API Contracts." *CASSIS 2004.*
11. Meyer, B. (2003). "The Grand Challenge of Trusted Components." *ICSE 2003.*
12. Dagstuhl Seminar 26031 (2026). "Software Contracts Meet System Contracts."

### Rust Verification and Bounded Model Checking

13. Le Blanc, A. & Lam, P. (2024). "Surveying the Rust Verification Landscape." arXiv:2410.01981.
14. Lattuada, A. et al. (2023). "Verus: Verifying Rust Programs using Linear Ghost Types." arXiv:2303.05491.
15. Ayoun, S.-E. et al. (2024). "A Hybrid Approach to Semi-automated Rust Verification." arXiv:2403.15122.
16. Le Blanc, A. & Lam, P. (2025). "Lessons Learned from Verifying the Rust Standard Library." arXiv:2510.01072.
17. Kroening, D. et al. (2023). "CBMC: The C Bounded Model Checker." arXiv:2302.02384.
18. Amusuo, P.C. et al. (2025). "Do Unit Proofs Work? Compositional Bounded Model Checking." arXiv:2503.13762.

### Frame Conditions and Separation Logic

19. Eilers, M. et al. (2024). "Verification Algorithms for Automated Separation Logic Verifiers." arXiv:2405.10661.
20. Jacobs, B. (2025). "VeriFast's Separation Logic." arXiv:2505.04500.
21. Fasse, J. & Jacobs, B. (2022). "Modular Termination Verification with Higher-Order Concurrent Separation Logic." arXiv:2212.14126.

### Loop Invariants and Termination

22. Sarita, Y. et al. (2024). "Syndicate: Efficient Ranking Function-Based Termination Analysis." arXiv:2404.05951.
23. Liu, R. et al. (2024). "Enhancing Automated Loop Invariant Generation with Large Language Models." arXiv:2412.10483.
24. Liu, C. et al. (2023). "LIG-MM: Towards General Loop Invariant Generation." arXiv:2311.10483.
25. Akhond, M.R. et al. (2025). "LLM For Loop Invariant Generation: How Far Are We?" arXiv:2511.06552.

### Behavioral Subtyping and Refinement

26. Haehnle, R. et al. (2023). "Context-aware Trace Contracts." arXiv:2310.04384.
27. Dominguez, F. & Spiwack, A. (2025). "Refinement-Types Driven Development." arXiv:2509.15005.

### Contracts for ML and Infrastructure

28. Wong, S. et al. (2023). "MLGuard: Defend Your Machine Learning Model!" arXiv:2309.01379.
29. Jakeman, J.D. et al. (2025). "V&V for Trustworthy Scientific Machine Learning." arXiv:2502.15496.
30. Chiari, M. et al. (2022). "Static Analysis of Infrastructure as Code: A Survey." arXiv:2206.10344.
31. Jana, P. et al. (2026). "TerraFormer: Automated IaC with LLMs via Policy-Guided Verifier Feedback." arXiv:2601.08734.

### Pre/Postcondition Inference and LLM-Assisted Verification

32. Richter, C. & Wehrheim, H. (2025). "Beyond Postconditions: Can LLMs Infer Formal Contracts?" arXiv:2510.12702.
33. Faria, J.P. et al. (2026). "Automatic Generation of Formal Specification Using LLMs and Test Oracles." arXiv:2601.12845.
34. Wen, C. et al. (2024). "Enchanting Program Specification Synthesis by LLMs." arXiv:2404.00762.
35. Yang, A.Z.H. et al. (2024). "VERT: Verified Equivalent Rust Transpilation with LLMs." arXiv:2404.18852.
36. Liu, Y. et al. (2024). "PropertyGPT: LLM-driven Formal Verification." arXiv:2405.02580.
37. Lim, S. et al. (2025). "ContractEval: Evaluating Contract-Satisfying Assertions." arXiv:2510.12047.
38. Councilman, A. et al. (2025). "Towards Formal Verification of LLM-Generated Code." arXiv:2507.13290.

### Internal Cross-References

- [escape-proof-enforcement.md](escape-proof-enforcement.md) — Six-stage
  compile-time enforcement pipeline (equation pre/post → build.rs →
  `#[contract]` macro → test execution)
- [lean-kani-composition.md](lean-kani-composition.md) — How Lean (ℝ)
  and Kani (f32) compose via `stub_float` bridge
- [pytorch-extraction.md](pytorch-extraction.md) — `pv extract-pytorch`
  infers pre/postconditions from PyTorch docstrings
