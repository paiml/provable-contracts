# adamw-kernel-v1

**Version:** 1.0.0

AdamW kernel — Adam optimizer with decoupled weight decay

## References

- Loshchilov & Hutter (2017) Decoupled Weight Decay Regularization
- Kingma & Ba (2014) Adam: A Method for Stochastic Optimization

## Equations

### adam_moments

$$
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
$$

**Domain:** $g_t in R^d, m_0 = 0, beta1 in (0, 1)$

**Codomain:** $m_t in R^d$

**Invariants:**

- $m_t is exponential moving average of gradients$
- $|m_t| bounded by max(|g_1|, ..., |g_t|) when beta1 < 1$

### adam_variance

$$
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
$$

**Domain:** $g_t in R^d, v_0 = 0, beta2 in (0, 1)$

**Codomain:** $v_t in R_>=0^d$

**Invariants:**

- $v_t >= 0 (non-negative second moment)$
- $v_t is exponential moving average of squared gradients$

### bias_correction

$$
m_hat_t = m_t / (1 - beta1^t), v_hat_t = v_t / (1 - beta2^t)
$$

**Domain:** $t >= 1, beta1 in (0,1), beta2 in (0,1)$

**Codomain:** $m_hat_t in R^d, v_hat_t in R_>=0^d$

**Invariants:**

- $Correction factor > 1 for all t >= 1$
- $Correction approaches 1 as t -> inf$

### weight_update

$$
theta_t = theta_{t-1} - lr * (m_hat_t / (\sqrt{v_hat_t} + eps) + lambda * theta_{t-1})
$$

**Domain:** $theta in R^d, lr > 0, lambda >= 0, eps > 0$

**Codomain:** $theta_t in R^d$

**Invariants:**

- $Weight decay applied AFTER Adam update (decoupled)$
- $Update finite when inputs finite and eps > 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Decoupled weight decay | $Weight decay term is lambda * theta, not lambda * theta in gradient$ |
| 2 | bound | Second moment non-negative | $v_t >= 0 for all t and all dimensions$ |
| 3 | bound | Bias-corrected moments finite | $m_hat_t and v_hat_t are finite when g_t is finite$ |
| 4 | invariant | Bias correction factor | $1 / (1 - beta^t) > 1 for t >= 1 and beta in (0, 1)$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **update_first_moment**: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t — *m_t is linear combination of m_{t-1} and g_t*
2. **update_second_moment**: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2 — *v_t >= 0*
3. **bias_correct**: Compute bias-corrected m_hat and v_hat — *Correction factor > 1*
4. **adam_step**: Compute lr * m_hat / (sqrt(v_hat) + eps) — *Step is finite when eps > 0*
5. **weight_decay**: Subtract lr * lambda * theta (decoupled) — *Decay applied to theta, not gradient*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| adamw | avx2 | `adamw_step_avx2` |
| adamw | scalar | `adamw_step_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-AW-001 | Decoupled weight decay | AdamW(g, lambda) != Adam(g + lambda*theta) for lambda > 0 | Weight decay is coupled (L2 reg instead of decoupled) |
| FALSIFY-AW-002 | Second moment non-negativity | v_t >= 0 for all t after random gradient updates | Floating-point underflow in EMA update |
| FALSIFY-AW-003 | Bias correction | 1/(1-beta^t) > 1 for t in [1, 10000] and beta in (0, 1) | Integer overflow in power computation or division by zero |
| FALSIFY-AW-004 | Update finiteness | theta_t is finite when g_t is finite and eps > 0 | Division by near-zero denominator when eps too small |
| FALSIFY-AW-005 | SIMD equivalence | \|adamw_avx2(args) - adamw_scalar(args)\| < 8 ULP | SIMD sqrt or reciprocal approximation differs |
| FALSIFY-AW-006 | Boundary - zero gradient | With g=0, only weight decay modifies theta | Bias correction or moment update incorrect at zero |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-AW-001 | AW-INV-001 | 4 | stub_float |
| KANI-AW-002 | AW-BND-001 | 8 | stub_float |
| KANI-AW-003 | AW-BND-002 | 4 | stub_float |

## QA Gate

**AdamW Contract** (F-AW-001)

Decoupled weight decay optimizer quality gate

**Checks:** decoupled_decay, moment_positivity, update_finiteness, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

