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

-- Status: sorry (proof pending)
theorem monotone {n : ℕ} (x : RVec n) (i j : Fin n)
    (h : x i > x j) :
    softmax x i > softmax x j := by
  sorry

end ProvableContracts.Softmax
