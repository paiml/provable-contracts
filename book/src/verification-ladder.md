# The Verification Ladder

Every proof obligation in a contract is verified at multiple levels. Higher
levels subsume lower ones. The goal is to push every obligation as high as
practically possible.

```
Level   Method                  Tool            What it proves
─────   ──────                  ────            ──────────────
  5     Mathematical proof      Lean/Coq        True for ALL inputs. Period.
                                                (out of scope for this project)

  4     Bounded model check     Kani ←────────  True for ALL inputs up to size N.
        (formal verification)                   Exhaustive. No sampling. ACTUAL PROOF
                                                within the bound.        ← TARGET

  3     Property-based test     probar/proptest  True for ~10,000 random inputs.
        + metamorphic                            High confidence, not proof.

  2     Contract test           #[test]          True for specific edge cases
        (falsification)                          chosen by developer.

  1     Type system             rustc            True by construction.
        (Poka-Yoke)                              Compile error if violated.

  0     Code review             Human eyes       "Looks right to me."
```

## Where Each Tool Lives

| Obligation Type | Level 1 (Types) | Level 3 (probar) | Level 4 (Kani) |
|----------------|-----------------|-------------------|-----------------|
| Shape correctness | `ValidatedTensor` newtype | N/A (compile-time) | N/A (compile-time) |
| Softmax sums to 1 | N/A | proptest random vectors | `#[kani::proof]` all vectors <= 16 |
| SIMD = scalar | N/A | proptest random data | `#[kani::proof]` all data <= 256 |
| No overflow | N/A | proptest edge cases | Kani automatic (checks ALL paths) |
| Quantized bsums correct | N/A | proptest random blocks | `#[kani::proof]` all blocks (integer-exact) |
| Format isolation | `#[test]` cross-format | N/A | `#[kani::proof]` + `#[kani::should_panic]` |

## The Provability Claim

When we say a kernel is "provable," we mean:

1. **Level 1:** The type system prevents invalid construction (Poka-Yoke).
2. **Level 3:** probar has tested the property for 10,000+ random inputs.
3. **Level 4:** Kani has exhaustively verified the property for ALL inputs up
   to the kernel's natural bound (super-block size, SIMD width, etc.).

This is not Level 5 (full mathematical proof in Lean/Coq). But for fixed-size
kernel operations -- which is what ML inference IS -- bounded verification at
the natural bound IS exhaustive. A Q4_K super-block is always 256 elements.
Verifying for all 256-element inputs IS verifying for all inputs.
