/-!
# Softmax Monotonicity

Proves that softmax preserves input ordering.

## Obligation

`SM-INV-003`: ∀ x ∈ ℝⁿ, x_i > x_j → softmax(x)_i > softmax(x)_j
-/

import ProvableContracts.Defs.Softmax
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Softmax preserves input ordering: larger input → larger output.
    Since both terms share the same positive denominator Z, we reduce
    to showing exp(xᵢ) > exp(xⱼ), which follows from exp being monotone. -/
theorem monotone {n : ℕ} (x : RVec n) (i j : Fin n)
    (h : x i > x j) :
    softmax x i > softmax x j := by
  unfold softmax
  apply div_lt_div_of_pos_right
  · exact Real.exp_lt_exp.mpr h
  · apply Finset.sum_pos
    · intro k _
      exact Real.exp_pos (x k)
    · exact univ_nonempty

end ProvableContracts.Softmax
