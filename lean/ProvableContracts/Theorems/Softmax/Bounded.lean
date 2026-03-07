/-!
# Softmax Bounded

Proves that softmax outputs lie strictly in (0, 1).

## Obligation

`SM-BND-001`: ∀ x ∈ ℝⁿ, 0 < softmax(x)ᵢ < 1

The lower bound follows from `softmax_pos`. The upper bound follows
from `partition_of_unity`: since all terms are positive and sum to 1,
each term must be strictly less than 1.
-/

import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.NonNegativity
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Each softmax output is strictly less than 1.
    Since all outputs are positive and sum to 1, no single output
    can equal or exceed 1. -/
theorem softmax_lt_one {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) :
    softmax x i < 1 := by
  have hsum := partition_of_unity x
  have hpos : ∀ j, softmax x j > 0 := softmax_pos x
  -- softmax(x)_i = 1 - Σ_{j≠i} softmax(x)_j
  -- Since Σ_{j≠i} softmax(x)_j > 0 (at least one other term), we get < 1
  rw [show (1 : ℝ) = univ.sum (softmax x) from hsum.symm]
  apply Finset.single_lt_sum (Finset.mem_univ i)
    (fun j _ => le_of_lt (hpos j))
  exact ⟨i.succAbove 0, Finset.mem_univ _, Ne.symm (Fin.succAbove_ne i 0), hpos _⟩

/-- Softmax outputs are strictly bounded in (0, 1). -/
theorem softmax_bounded {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) :
    0 < softmax x i ∧ softmax x i < 1 :=
  ⟨softmax_pos x i, softmax_lt_one x i⟩

end ProvableContracts.Softmax
