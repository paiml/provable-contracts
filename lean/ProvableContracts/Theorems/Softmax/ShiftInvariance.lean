/-!
# Softmax Shift Invariance

Proves that softmax is invariant under translation: σ(x + c·1) = σ(x).

## Obligation

`SM-INV-006`: ∀ x ∈ ℝⁿ, ∀ c ∈ ℝ, softmax(x + c) = softmax(x)

This is the key property for numerical stability: subtracting max(x)
does not change the output. The proof is algebraic: exp(xᵢ+c)/Σexp(xⱼ+c)
= exp(xᵢ)·exp(c) / (exp(c)·Σexp(xⱼ)) = exp(xᵢ)/Σexp(xⱼ).
-/

import ProvableContracts.Defs.Softmax
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset

namespace ProvableContracts.Softmax

open Real Finset

/-- Shifted input vector: adds scalar c to every component. -/
noncomputable def shift {n : ℕ} (x : RVec n) (c : ℝ) : RVec n :=
  fun i => x i + c

-- Status: proved
/-- Softmax is invariant under uniform translation.
    exp(xᵢ+c) / Σⱼ exp(xⱼ+c) = exp(xᵢ)·e^c / (e^c · Σⱼ exp(xⱼ))
    = exp(xᵢ) / Σⱼ exp(xⱼ). -/
theorem shift_invariance {n : ℕ} (x : RVec n) (c : ℝ) (i : Fin n) :
    softmax (shift x c) i = softmax x i := by
  unfold softmax shift
  simp only [Real.exp_add]
  rw [Finset.sum_mul_distrib]
  -- Now: exp(xᵢ) * exp(c) / (Σⱼ exp(xⱼ) * exp(c))
  rw [mul_div_mul_right]
  exact ne_of_gt (Real.exp_pos c)

-- Helper: Finset.sum distributes multiplication
private theorem Finset.sum_mul_distrib {n : ℕ} (f : Fin n → ℝ) (c : ℝ) :
    univ.sum (fun j => f j * c) = univ.sum f * c := by
  rw [Finset.sum_mul]

end ProvableContracts.Softmax
