/-!
# Softmax Definitions

Mathematical definition of softmax over real-valued vectors,
matching the `softmax-kernel-v1.yaml` contract equations.

## References

- Bridle, J.S. "Training Stochastic Model Recognition Algorithms
  as Networks can Lead to Maximum Mutual Information Estimation
  of Parameters." NeurIPS, 1990.
- Vaswani et al. "Attention Is All You Need." NeurIPS, 2017. Eq. 3.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import ProvableContracts.Basic

namespace ProvableContracts.Softmax

open Real Finset

/-- The softmax function: `softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`. -/
noncomputable def softmax {n : ℕ} (x : RVec n) (i : Fin n) : ℝ :=
  Real.exp (x i) / univ.sum (fun j => Real.exp (x j))

/-- Log-softmax: `log_softmax(x)_i = x_i - log(Σ_j exp(x_j))`. -/
noncomputable def log_softmax {n : ℕ} (x : RVec n) (i : Fin n) : ℝ :=
  x i - Real.log (univ.sum (fun j => Real.exp (x j)))

/-- Numerically stable softmax via max-subtraction:
    `stable_softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))`. -/
noncomputable def stable_softmax {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) : ℝ :=
  let m := RVec.max x
  Real.exp (x i - m) / univ.sum (fun j => Real.exp (x j - m))

end ProvableContracts.Softmax
