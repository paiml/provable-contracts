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

-- Status: sorry (proof pending)
theorem softmax_pos {n : ℕ} (x : RVec n) (i : Fin n) :
    softmax x i > 0 := by
  sorry

end ProvableContracts.Softmax
