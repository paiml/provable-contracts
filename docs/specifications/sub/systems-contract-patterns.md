# Section 34: Systems Contract Patterns

> Parent: [pv-spec.md](../pv-spec.md) §34

## 34.1 Motivation

Sections 1–33 cover **mathematical** contracts (equations, bounds, roundtrips) and **interface**
contracts (CLI, MCP, HTTP). This section specifies contract patterns for **systems** concerns:
threading, concurrency, async, compute backend dispatch, memory lifecycle, and LLM architecture
invariants. These patterns cut across all sovereign stack repos.

### References

- Lamport (1978). Time, Clocks, and the Ordering of Events. CACM 21(7)
- Flanagan & Freund (2009). FastTrack: Efficient and Precise Dynamic Race Detection. PLDI
- Bader et al. (2021). Correctness of Automatic Differentiation via Pullbacks. NeurIPS
- Li & Tan (2023). Towards Verified GPU Kernels. arXiv:2303.08600
- Jiang et al. (2024). Formal Verification of SIMD Vectorization. PLDI
- Jung et al. (2020). RustBelt meets Relaxed Memory. POPL
- Dathathri et al. (2020). GENN: Verified GPU Code Generation for Spiking Neural Networks
- Liang et al. (2022). Verifying Quantized Neural Networks. CAV
- Xi & Harper (2001). Dependently Typed Data Structures. J. Functional Programming

## 34.2 Threading & Lock Ordering

### Contract Pattern: `lock-ordering`

Prevents deadlocks by enforcing a total order on lock acquisition.

```yaml
equations:
  lock_order_invariant:
    formula: |
      lock_order: (Mutex<A>, Mutex<B>) -> bool
        ∀ thread t: if t.holds(A) ∧ t.acquires(B) then order(A) < order(B)
        Violated: A.lock() then B.lock() where order(A) > order(B)
    invariants:
    - Lock acquisition follows total order (no cycles in wait-for graph)
    - Documented lock levels: L0 (index), L1 (cache), L2 (state), L3 (IO)
    preconditions:
    - All lockable resources have assigned levels
    postconditions:
    - No deadlock reachable from any interleaving
    lean_theorem: Theorems.LockOrderInvariant
```

**Proof obligation type**: `ordering`
**Falsification**: Spawn 100 threads acquiring locks A,B in both orders; detect panic or timeout.
**Applies to**: `pmat` (DashMap in index), `forjar` (state lock + config lock), `trueno` (graph + cache).

### Contract Pattern: `data-race-freedom`

```yaml
equations:
  race_freedom:
    formula: |
      race_free: Program -> bool
        ∀ memory location m, ∀ concurrent accesses (a₁, a₂) to m:
          a₁.is_write ∨ a₂.is_write → synchronized(a₁, a₂)
    invariants:
    - All shared mutable state behind Mutex, RwLock, or atomic
    - No raw pointer aliasing across thread boundaries
    preconditions:
    - Program compiles under Rust borrow checker (Send + Sync)
    postconditions:
    - MIRI detects zero data races under any scheduling
```

**Proof obligation type**: `soundness`
**Falsification**: Run under `cargo +nightly miri test` with `-Zmiri-preemption-rate=0.1`.

## 34.3 Async & Structured Concurrency

### Contract Pattern: `cancellation-safety`

Ensures async tasks release all resources when cancelled.

```yaml
equations:
  cancellation_safe:
    formula: |
      cancel: Task<T> -> ()
        ∀ resource r acquired by task:
          r.is_released() after cancel
        No leaked: file handles, temp files, network connections, semaphore permits
    invariants:
    - Drop impl releases all resources
    - select! branches are cancellation-safe (no partial state)
    preconditions:
    - Task is in Running state
    postconditions:
    - All acquired resources released within 1s of cancel signal
    lean_theorem: Theorems.CancellationSafe
```

**Proof obligation type**: `frame`
**Falsification**: Start analysis, cancel after 100ms via `tokio::time::timeout`, check for `.tmp` files and open fd count.

### Contract Pattern: `structured-concurrency`

```yaml
equations:
  structured_spawn:
    formula: |
      structured: (Parent, Vec<Child>) -> bool
        ∀ child in parent.spawned:
          child.lifetime ⊆ parent.lifetime
        parent.await => all children completed or cancelled
    invariants:
    - No orphan tasks (tasks that outlive their parent scope)
    - JoinSet/TaskSet owns all spawned work
    - Panic in child propagates to parent (not silently lost)
```

**Proof obligation type**: `frame`
**Applies to**: `pmat` (rayon parallel analysis), `realizar` (batched inference), `entrenar` (data loader workers).

### Contract Pattern: `channel-conservation`

```yaml
equations:
  channel_lossless:
    formula: |
      lossless: (Sender<T>, Receiver<T>, Bound) -> bool
        sent_count = received_count + pending_count + dropped_on_close
        ∀ msg: msg is received XOR sender was dropped before delivery
    invariants:
    - No silent message loss in bounded channels
    - Backpressure: send blocks when channel full (not drop)
    postconditions:
    - After close: receiver drains remaining pending messages
```

**Proof obligation type**: `conservation`

## 34.4 Compute Backend Dispatch

### Contract Pattern: `simd-scalar-parity`

The foundational compute contract: SIMD and GPU implementations must match scalar within ULP tolerance.

```yaml
equations:
  simd_scalar_parity:
    formula: |
      parity: (f_scalar, f_simd, x) -> bool
        |f_scalar(x) - f_simd(x)| ≤ ULP_TOLERANCE * ε_format
        Where:
          ε_f32 = 2^{-23} ≈ 1.19e-7
          ε_f16 = 2^{-10} ≈ 9.77e-4
          ULP_TOLERANCE = sqrt(n) for n-element reductions (FMA reassociation)
    domain: x ∈ ℝ^n, n ≤ MAX_VECTOR_LEN
    codomain: bool (parity within tolerance)
    invariants:
    - scalar is the reference implementation (ground truth)
    - SIMD may reassociate FMA operations (different rounding)
    - Tolerance is derived from arithmetic, not guessed
    preconditions:
    - x contains no NaN or Inf (unless testing those paths)
    - f_scalar and f_simd compute the same mathematical function
    postconditions:
    - max element-wise error ≤ ULP_TOLERANCE * ε_format
    lean_theorem: Theorems.SimdScalarParity
```

**Proof obligation type**: `equivalence`
**Tolerance selection** (from Jiang et al. 2024):

| Operation | n | ε_f32 Tolerance | Rationale |
|-----------|---|-----------------|-----------|
| Element-wise (relu, sigmoid) | 1 | 0 ULP | No reassociation |
| Dot product | n | sqrt(n) * ε | FMA reassociation |
| Softmax | n | n * ε | exp + division chain |
| MatMul (M×K×N) | K | sqrt(K) * ε | Reduction dimension |
| LayerNorm/RMSNorm | n | 2*sqrt(n) * ε | sqrt + division |

**Falsification**: Property test with 10K random vectors, compare scalar vs AVX2 vs NEON, assert max ULP error within bound.

### Contract Pattern: `gpu-cpu-parity`

Extends simd-scalar-parity to GPU backends (WGPU, CUDA, PTX).

```yaml
equations:
  gpu_cpu_parity:
    formula: |
      parity: (f_cpu, f_gpu, x) -> bool
        |f_cpu(x) - f_gpu(x)| ≤ GPU_TOLERANCE * ε_format
        Where GPU_TOLERANCE = max(ULP_TOLERANCE, WARP_REASSOCIATION_BOUND)
        WARP_REASSOCIATION_BOUND = log2(WARP_SIZE) * ε for reductions
    invariants:
    - CPU (scalar or SIMD) is reference; GPU must match
    - GPU may have wider reassociation window (warp-level reduce)
    - Different GPU architectures may have different rounding
    - PTX must match WGPU within 1 ULP (same hardware, different frontend)
    postconditions:
    - max element-wise error ≤ GPU_TOLERANCE * ε_format
    lean_theorem: Theorems.GpuCpuParity
```

**Additional GPU invariants**:

```yaml
  gpu_safety:
    formula: |
      safe: Kernel -> bool
        No out-of-bounds shared memory access
        No uninitialized shared memory read
        Barrier sync before shared memory read-after-write
        Workgroup size ≤ device.max_workgroup_size
    invariants:
    - Bounds checks on all shared memory indices
    - __syncthreads() / workgroupBarrier() before cross-lane reads
    - No divergent barriers (all threads in workgroup reach same barrier)
```

**Proof obligation type**: `equivalence` (parity), `soundness` (safety)

### Contract Pattern: `backend-dispatch-completeness`

```yaml
equations:
  dispatch_complete:
    formula: |
      dispatch: (Operation, Backend) -> Impl
        ∀ op in OperationSet, ∀ backend in {Scalar, AVX2, NEON, WGPU, CUDA, PTX}:
          if backend.is_available() then dispatch(op, backend) exists
        Fallback: unavailable backend → scalar (never panic)
    invariants:
    - Every operation has scalar fallback
    - Runtime detection: CPUID for SIMD, device enumeration for GPU
    - No compile-time only dispatch (must handle runtime absence)
    postconditions:
    - dispatch never returns None; always falls back to scalar
```

**Proof obligation type**: `completeness`
**Applies to**: `trueno` (SIMD dispatch), `realizar` (GPU inference), `aprender` (backend selection).

## 34.5 Memory Lifecycle Patterns

### Contract Pattern: `lazy-initialization`

```yaml
equations:
  lazy_init:
    formula: |
      lazy: LazyCell<T> -> T
        First access: T = init()
        Subsequent: T = cached_value (no re-init)
        init() called at most once (even under concurrency)
    invariants:
    - Initialization is idempotent (safe to race)
    - Once initialized, value never changes
    - Thread-safe: OnceCell/OnceLock semantics
    preconditions:
    - init function is pure (no side effects beyond allocation)
    postconditions:
    - All accesses after first return the same value
    lean_theorem: Theorems.LazyInit
```

**Proof obligation type**: `idempotency`
**Applies to**: `pmat` (lazy source loading in SQLite backend), `trueno` (lazy SIMD detection), `aprender` (lazy model loading).

### Contract Pattern: `lru-cache-correctness`

```yaml
equations:
  lru_correctness:
    formula: |
      lru: (Cache<K,V>, Capacity) -> bool
        cache.len() ≤ capacity always
        get(k) exists → k becomes most-recently-used
        put(k,v) when full → evicts least-recently-used
        evicted entries: refcount = 0, memory freed
    invariants:
    - Size invariant: len() ≤ capacity after any operation
    - Recency ordering: access list is permutation of key set
    - No phantom entries: evicted key not findable by get()
    postconditions:
    - After put at capacity: exactly one eviction
    lean_theorem: Theorems.LruCorrectness
```

**Proof obligation type**: `bound` (capacity), `ordering` (recency), `frame` (eviction cleanup)

### Contract Pattern: `arena-lifetime`

```yaml
equations:
  arena_contained:
    formula: |
      arena: Arena<'a> -> bool
        ∀ ref r allocated from arena: lifetime(r) ⊆ lifetime(arena)
        drop(arena) → all allocations freed
        No use-after-free: r is invalid after arena.reset()
    invariants:
    - Allocations cannot escape arena lifetime (enforced by Rust borrow checker)
    - Bulk deallocation on drop (O(1) per arena, not per object)
    postconditions:
    - After drop: heap usage returns to pre-arena level
```

**Proof obligation type**: `frame`

### Contract Pattern: `memory-budget`

```yaml
equations:
  budget_honored:
    formula: |
      budgeted: (Operation, Budget) -> Result<T, OOM>
        peak_memory(operation) ≤ budget
        If would exceed: returns Err(OOM) before allocation
        Never triggers OS OOM killer
    invariants:
    - Budget checked before large allocations (not after)
    - Partial results returned on budget exhaustion (not empty)
    - Budget tracking is conservative (overestimates, never underestimates)
    postconditions:
    - process RSS ≤ budget + baseline_overhead
```

**Proof obligation type**: `bound`
**Applies to**: `pmat` (index loading), `realizar` (model loading), `trueno-zram` (compression buffer).

## 34.6 LLM Architecture Contracts

### Contract Pattern: `attention-correctness`

```yaml
equations:
  attention:
    formula: |
      Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
      Where:
        Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{m×d_k}, V ∈ ℝ^{m×d_v}
        Output ∈ ℝ^{n×d_v}
    invariants:
    - Attention weights sum to 1.0 per query (softmax normalization)
    - Causal mask: attn[i,j] = 0 when j > i (autoregressive)
    - GQA: num_kv_heads divides num_heads evenly
    preconditions:
    - Q, K have matching d_k dimension
    - Sequence length ≤ max_seq_len
    postconditions:
    - Output shape = (batch, n, d_v)
    - attn_weights ∈ [0, 1] and rows sum to 1.0 ± ε
```

**Proof obligation type**: `bound` (weights), `invariant` (causal mask), `conservation` (normalization)

### Contract Pattern: `kv-cache-coherence`

```yaml
equations:
  kv_cache_coherent:
    formula: |
      cache_coherent: (KVCache, NewTokens) -> bool
        append(cache, new_kv) preserves all existing entries
        cache[pos] = kv_at_generation_time(pos) for all pos < current
        Paged: page_table[logical] → physical is bijective for active pages
    invariants:
    - Append-only during generation (no mutation of past KV pairs)
    - Paged attention: no two logical positions map to same physical page
    - Eviction: only evict positions that will never be attended to
    postconditions:
    - cache.len() = total_generated_tokens
    - Recompute from scratch matches cached result within ε
```

**Proof obligation type**: `invariant` (append-only), `equivalence` (recompute parity)

### Contract Pattern: `quantization-roundtrip`

```yaml
equations:
  quant_roundtrip:
    formula: |
      roundtrip: (Tensor, QuantConfig) -> f64
        error = max|dequant(quant(x)) - x| / max|x|
        error ≤ expected_error(config)
        Where:
          Q4_K:  expected ≤ 0.05 (5% relative)
          Q8_0:  expected ≤ 0.004 (0.4% relative)
          FP8:   expected ≤ 0.02 (2% relative)
          INT8:  expected ≤ 0.008 (0.8% relative)
    invariants:
    - Quantize then dequantize preserves value within tolerance
    - Zero is preserved exactly (quant(0) = 0, dequant(0) = 0)
    - Monotonicity: x₁ < x₂ → quant(x₁) ≤ quant(x₂)
    postconditions:
    - Relative error ≤ expected_error(config)
```

**Proof obligation type**: `roundtrip`, `bound`, `monotonicity`

### Contract Pattern: `tokenizer-correctness`

```yaml
equations:
  tokenizer_roundtrip:
    formula: |
      roundtrip: String -> bool
        decode(encode(text)) = text for all valid UTF-8
        encode(text).len() ≤ text.len() * MAX_EXPANSION_FACTOR
    invariants:
    - Encode-decode is identity for valid input
    - Unknown tokens get fallback (byte-level BPE, not panic)
    - Encoding is deterministic (same text → same tokens)
    postconditions:
    - Decoded output = original input
    - Token count is bounded
```

**Proof obligation type**: `roundtrip`, `determinism`, `bound`

### Contract Pattern: `sampling-distribution`

```yaml
equations:
  sampling_valid:
    formula: |
      sample: (Logits, SamplingParams) -> TokenId
        temperature > 0: softmax(logits / temperature) is valid distribution
        top_k > 0: only top k logits considered
        top_p ∈ (0, 1]: nucleus sampling threshold
        Greedy (temperature = 0): argmax(logits) deterministically
    invariants:
    - Sampled token is always a valid vocabulary index
    - Temperature 0 is deterministic (argmax)
    - Temperature > 0 with fixed seed is reproducible
    postconditions:
    - token_id ∈ [0, vocab_size)
    - Greedy sampling: token_id = argmax(logits)
```

**Proof obligation type**: `bound` (valid index), `determinism` (greedy/seeded)

## 34.7 Cross-Cutting Contract Patterns

### Pattern: `deterministic-parallel`

Parallel execution must produce identical results to sequential (modulo ordering).

```yaml
equations:
  parallel_determinism:
    formula: |
      sort(parallel_map(f, inputs)) = sort(sequential_map(f, inputs))
      Where f is a pure function (no shared mutable state)
    invariants:
    - Results are set-equal (same elements, possibly different order)
    - No result depends on scheduling order
    - HashMap iteration order does not leak into output
```

**Applies to**: All repos using rayon, tokio::spawn, or parallel iterators.

### Pattern: `timeout-bounded`

All long-running operations must honor a timeout.

```yaml
equations:
  timeout_honored:
    formula: |
      bounded: (Operation, Timeout) -> Result<T, TimeoutError>
        wall_clock(operation) ≤ timeout + epsilon
        epsilon = max(1s, cleanup_time)
    invariants:
    - Timeout cancels in-progress work (not just checked at start)
    - Partial results returned when available
    - Resources released on timeout (no leaks)
```

## 34.8 Summary: Contract Pattern Registry

| Pattern | Obligation Type | Domain | Repos |
|---------|----------------|--------|-------|
| lock-ordering | ordering | Threading | pmat, forjar, trueno |
| data-race-freedom | soundness | Threading | all Rust repos |
| cancellation-safety | frame | Async | pmat, realizar, entrenar |
| structured-concurrency | frame | Async | pmat, realizar, entrenar |
| channel-conservation | conservation | Async | pmat, forjar |
| simd-scalar-parity | equivalence | Compute | trueno, aprender, realizar |
| gpu-cpu-parity | equivalence | Compute | trueno, realizar |
| backend-dispatch-completeness | completeness | Compute | trueno, aprender, realizar |
| lazy-initialization | idempotency | Memory | pmat, trueno, aprender |
| lru-cache-correctness | bound+ordering | Memory | pmat, trueno |
| arena-lifetime | frame | Memory | pmat, aprender |
| memory-budget | bound | Memory | pmat, realizar, trueno-zram |
| attention-correctness | bound+invariant | LLM | aprender, realizar, entrenar |
| kv-cache-coherence | invariant+equiv | LLM | aprender, realizar |
| quantization-roundtrip | roundtrip+bound | LLM | aprender, trueno |
| tokenizer-correctness | roundtrip+determ | LLM | aprender |
| sampling-distribution | bound+determinism | LLM | aprender, realizar |
| deterministic-parallel | determinism | Cross | all parallel repos |
| timeout-bounded | bound | Cross | all CLI/server repos |

Each pattern maps to one or more of the 26 proof obligation types defined in pv-spec §3.
