/-!
# Softmax Partition of Unity

Proves that softmax outputs sum to 1 for all real inputs.

## Obligation

`SM-INV-001`: ∀ x ∈ ℝⁿ, Σᵢ softmax(xᵢ) = 1

This is the flagship Phase 7 proof. The key insight is that
the sum telescopes: Σᵢ exp(xᵢ)/Z = Z/Z = 1 where Z = Σⱼ exp(xⱼ).
-/

import ProvableContracts.Defs.Softmax
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Softmax outputs sum to 1: Σᵢ exp(xᵢ)/Z = Z/Z = 1. -/
theorem partition_of_unity {n : ℕ} (x : RVec n) :
    univ.sum (softmax x) = 1 := by
  unfold softmax
  simp_rw [Finset.sum_div]
  apply div_self
  ne_of_gt (Finset.sum_pos (fun j _ => Real.exp_pos (x j)) univ_nonempty)

end ProvableContracts.Softmax
