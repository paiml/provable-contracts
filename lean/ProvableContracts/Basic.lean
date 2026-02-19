/-!
# ProvableContracts — Shared Definitions

Common notation and utility definitions shared across all kernel
theorem modules.

This module provides the foundation types and notation used by
the theorem-proving layer (Phase 7) of the provable-contracts
pipeline.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset

namespace ProvableContracts

/-- A finite vector of reals, indexed by `Fin n`. -/
abbrev RVec (n : ℕ) := Fin n → ℝ

/-- Sum of all elements in a real vector. -/
noncomputable def RVec.sum {n : ℕ} (v : RVec n) : ℝ :=
  Finset.univ.sum v

/-- Maximum element of a nonempty real vector. -/
noncomputable def RVec.max {n : ℕ} (v : RVec (n + 1)) : ℝ :=
  Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ v

end ProvableContracts
