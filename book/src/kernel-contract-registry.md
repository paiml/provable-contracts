# Kernel Contract Registry

## Naming Convention

```
<operation>-kernel-v<version>.yaml
```

Examples:
- `softmax-kernel-v1.yaml`
- `rmsnorm-kernel-v1.yaml`
- `attention-kernel-v1.yaml`
- `rope-kernel-v1.yaml`
- `matmul-kernel-v1.yaml`

## Contract ID Convention

Falsification test IDs follow: `FALSIFY-<PREFIX>-NNN`

| Contract | Prefix |
|----------|--------|
| Softmax | SM |
| RMSNorm | RMS |
| Attention | ATTN |
| FlashAttention | FATTN |
| RoPE | ROPE |
| MatMul (GEMM/GEMV) | MM |
| SwiGLU/GeGLU | ACT |
| Quantized Dot Product | QDOT |
| Tensor Layout | LAYOUT |
| Layer Parity | PARITY |
| Kernel Fusion | FUSION |

## QA Gate ID Convention

`F-<PREFIX>-NNN` (matches certeza format).

---

## Existing Contracts (aprender)

These four contracts already exist in `aprender/contracts/` and serve as the
reference implementation of this specification:

### quantized-dot-product-v1.yaml

**Papers:** GPTQ (Frantar 2022, arXiv:2210.17323), LLM.int8() (Dettmers 2022),
GGML K-quant (ggerganov), Wulf & McKee 1995 (Memory Wall).

**Key equation:**
```
dot(W, x) = Σ_superblock [
  SCALE:  d_W * d_x * Σ_j(s_j * Σ_i(q_W_i * q_x_i))
  OFFSET: dmin_W * d_x * Σ_j(m_j * Σ_i(q_x_i))        ← bsums
]
```

**Key insight:** The offset term depends ONLY on activations (not weights),
so bsums can be precomputed once and reused across all weight rows.

**Falsification tests:** 5 (FALSIFY-QDOT-001 through 005).
**Format registry:** Q4_K, Q5_K, Q6_K, Q4_0, Q8_0 with full byte layouts.
**SIMD dispatch:** Exhaustive per format x ISA (scalar, AVX2, AVX-512 VNNI).

### tensor-layout-v1.yaml

**Theoretical basis:** Poka-Yoke (Shingo 1986), Popperian Falsificationism
(Popper 1959), Type-Driven Development (Brady 2017), Parse Don't Validate
(King 2019).

**Key principle:** `ValidatedTensor` newtypes make it IMPOSSIBLE (at compile
time) to use unvalidated data. Private inner fields + validated constructors.

**Falsification tests:** 8 (FALSIFY-001 through 008).
**Root cause:** PMAT-234 (SafeTensors 94.5% zeros passed structural checks).

### layer-parity-v1.yaml

**Problem:** 4 independent forward pass implementations (CPU SIMD, GPU
workspace, GPU graphed, GPU async) with no structural guarantee of equivalence.

**Key specification:** 14-step transformer layer forward pass with per-step
tolerance bounds.

**Enforcement:** `apr parity model.gguf` tool with cosine similarity >= 0.999,
KL divergence < 0.01, sigma >= 3.0, Cpk >= 1.33.

**Falsification tests:** 4 (PARITY-001 through 004).
**Root cause:** PMAT-232 (7B GPU garbage output).

### kernel-fusion-v1.yaml

**Theoretical basis:** Toyota Production System / Poka-Yoke (Shingo 1986),
Roofline Model (Williams et al. 2009), CUDA Graph Replay.

**Key principle:** Every fusion decision is documented with status (ACTIVE,
BLOCKED, PLANNED, REJECTED) and measurable benchmarks. No undocumented fusion.

**Root cause:** PAR-077 (fused kernel existed but was never wired in; when
tried, it was 3x slower due to shared memory overhead).

---

## Planned Contracts

Target kernels for aprender, ordered by dependency:

### Tier 1: Foundation Kernels (no dependencies)

| Contract | Paper | Key Equations |
|----------|-------|---------------|
| `softmax-kernel-v1.yaml` | Bridle 1990; Goodfellow 2016 | `s(x)_i = exp(x_i - max(x)) / S exp(x_j - max(x))` |
| `rmsnorm-kernel-v1.yaml` | Zhang & Sennrich 2019 | `RMSNorm(x) = x / RMS(x) * g, RMS = sqrt(mean(x^2) + e)` |
| `rope-kernel-v1.yaml` | Su et al. 2021 (arXiv:2104.09864) | `RoPE(x, m) = x * cos(mt) + rotate(x) * sin(mt)` |
| `activation-kernel-v1.yaml` | Shazeer 2020; Ramachandran 2017 | `SwiGLU(x,W,V,b,c) = Swish(xW+b) * (xV+c)` |

### Tier 2: Composite Kernels (depend on Tier 1)

| Contract | Paper | Key Equations |
|----------|-------|---------------|
| `attention-kernel-v1.yaml` | Vaswani et al. 2017 (arXiv:1706.03762) | `Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V` |
| `matmul-kernel-v1.yaml` | Standard linear algebra | `C = AB, C_{ij} = S_k A_{ik}B_{kj}` |
| `flash-attention-v1.yaml` | Dao et al. 2022 | Tiled attention with online softmax (Milakov 2018) |

### Tier 3: System Kernels (depend on Tier 1 + 2)

| Contract | Paper | Key Equations |
|----------|-------|---------------|
| `kv-cache-kernel-v1.yaml` | Pope et al. 2022 (arXiv:2210.09461) | Paged KV cache with block tables |
| `sampling-kernel-v1.yaml` | Holtzman et al. 2019 (arXiv:1904.09751) | Top-p (nucleus), top-k, temperature scaling |

### Dependency Graph

```
softmax ─────────────────┐
                         ├── attention ─── flash-attention
rope ────────────────────┤
                         ├── kv-cache
matmul ──────────────────┘
                              │
rmsnorm ─────────────────────┤
                              │
activation (SwiGLU) ─────────┤
                              │
quantized-dot-product ───────┤    (already exists)
                              │
tensor-layout ───────────────┤    (already exists)
                              │
layer-parity ────────────────┤    (already exists)
                              │
kernel-fusion ───────────────┘    (already exists)
```
