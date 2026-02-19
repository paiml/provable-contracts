/-!
# Softmax Non-Negativity

Proves that softmax outputs are strictly positive for all real inputs.

## Obligation

`SM-INV-002`: ∀ x ∈ ℝⁿ, softmax(x)_i > 0

This is the starter proof — it follows directly from `Real.exp_pos`.
-/

import ProvableContracts.Defs.Softmax
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Softmax outputs are strictly positive.
    Follows from `exp > 0` and the denominator being a sum of positives. -/
theorem softmax_pos {n : ℕ} (x : RVec n) (i : Fin n) :
    softmax x i > 0 := by
  unfold softmax
  apply div_pos
  · exact Real.exp_pos (x i)
  · apply Finset.sum_pos
    · intro j _
      exact Real.exp_pos (x j)
    · exact univ_nonempty

end ProvableContracts.Softmax
