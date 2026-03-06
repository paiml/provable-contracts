# gpu-decode-profiling-v1

**Version:** 2.0.0

GPU decode profiling contract — ensures BrickProfiler data reflects real GPU execution time AND report output faithfully represents profiler measurements.

## References

- REALIZAR-GPU-PERF-001 v2.10.0 §5 — BrickProfiler Decode Breakdown
- trueno BrickProfiler (`src/brick/profiler/mod.rs`) — SyncMode enum
- realizar executor\_api.rs — start\_brick\_id/stop\_brick\_id sync gates
- aprender `crates/apr-cli/src/commands/gguf.rs` — brick\_scores\_from\_profiler (B1-B5)
- aprender `crates/apr-cli/src/commands/cbtop_measure_batch.rs` — build\_and\_output\_report (B6-B13)
- aprender `crates/apr-cli/src/commands/cbtop_get_cpu_memory.rs` — simulated path (B14-B18)
- Hoefler & Belli SC'15 — Scientific Benchmarking of Parallel Computing Systems

## Problem Statement

The GPU decode profiling pipeline spans four repositories:

```
trueno (BrickProfiler) → realizar (CudaExecutor) → aprender (OwnedQuantizedModelCuda) → apr-cli (cbtop JSON)
```

Each boundary is a potential corruption point. This contract formalizes invariants at every stage — from kernel-level timing collection to final JSON report output.

### The Jidoka Incident (2026-03-06)

During 4090 serial benchmarking, cbtop reported LmHead at **1.9 µs** per call. The actual profiler measurement was **595 µs**. This 300x discrepancy triggered a Jidoka "STOP THE LINE" — all optimization work halted until profiling was provably correct.

Root cause analysis (Five Whys) traced the bug to apr-cli's last-mile conversion: `brick_scores_from_profiler()` was constructing `BrickScore` with hardcoded `score: 100, grade: "R", gap_factor: 1.0` instead of using the profiler's `stats.avg_us()` and `compute_brick_score()`. Full falsification found **18 bugs** across 3 files.

## Equations

### wall\_coverage

$$
coverage = \frac{\sum brick\_total\_ns}{wall\_clock\_ns}
$$

**Domain:** $brick\_total\_ns \geq 0, wall\_clock\_ns > 0$

**Codomain:** $coverage \in [0, 1]$

**Invariants:**

- $coverage \geq 0.85$ when profiling is valid (bricks account for ≥85% of wall time)
- $coverage < 0.50$ indicates CUDA graph replay hiding brick instrumentation
- $coverage > 1.0$ is impossible (bricks are subsets of wall time)

### sync\_verification

$$
is\_immediate = \frac{measured\_brick\_us}{expected\_brick\_us} > 0.5
$$

**Domain:** $measured\_brick\_us > 0, expected\_brick\_us = wall\_clock\_us / num\_bricks\_per\_token$

**Invariants:**

- Deferred sync: brick avg < 100µs regardless of kernel (CPU launch latency only)
- Immediate sync: brick avg correlates with kernel complexity (large GEMV > small norm)
- LmHead ($n=151936$) must be >10x RmsNorm in Immediate mode

### graph\_disable

$$
valid\_profiling \implies \neg has\_decode\_graph
$$

**Invariants:**

- CUDA graph replay executes all kernels in one opaque launch
- Bricks instrumented during graph CAPTURE only (first token), not REPLAY
- Profiling with graphs enabled measures $1/N$ of actual decode time

### report\_fidelity

$$
\forall b: JSON.actual\_us(b) = profiler.avg\_us(b)
$$

**Domain:** All bricks emitted in JSON output, $profiler.avg\_us > 0$

**Codomain:** $relative\_error \in [0, 0.01]$

**Invariants:**

- JSON `actual_us` must equal profiler per-call avg (not per-element, not per-token)
- JSON `score` must equal `compute_brick_score(actual_us, budget_us)` — never hardcoded
- JSON `grade` must equal `score_to_grade(score)` — never hardcoded
- JSON `gap_factor` must equal `actual_us / budget_us` — never 1.0 unless actual == budget
- No `BrickScore` field may be a compile-time constant (`score: 100`, `grade: 'R'`, `gap: 1.0`)

### report\_completeness

$$
|JSON.brick\_scores| = |profiler.all\_brick\_stats()|
$$

**Invariants:**

- Every profiler brick appears in JSON output — no silent truncation
- Aggregate `brick_score` uses all N bricks — zip with fixed-length array forbidden
- `FalsificationSummary.total_points == len(JSON.brick_scores)`
- `FalsificationSummary.passed + failed == total_points`

### report\_denominator

$$
decoded\_tokens = LmHead.count
$$

**Domain:** $LmHead.count > 0$ (exactly 1 LmHead per decoded token)

**Invariants:**

- `per_decoded_tok_us(b) = (b.count * b.avg_us) / decoded_tokens`
- `profiler.total_tokens` counts brick ELEMENTS — must NEVER be used as decoded token count
- Dividing `total_ns` by `profiler.total_tokens` produces values 100–300x too small

### report\_metadata

$$
\forall f \in \{rust\_project\_score, tdg\_score, cuda\_tdg\_score\}: f = 0.0
$$

(unless computed by pmat in this run)

**Invariants:**

- `FalsificationSummary` must derive from actual pass/fail counts, not constants
- No hardcoded magic numbers: 137, 173.9, 98.1, 95.2, 976.0

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| GDP-001 | Wall coverage | ≥85% brick coverage | CUDA graph hiding instrumentation |
| GDP-002 | Graph disable | <50% with graphs enabled | Unexpected graph-path instrumentation |
| GDP-003 | Sync mode | Immediate LmHead >100µs, Deferred <100µs | sync mode not propagating |
| GDP-004 | LmHead count | LmHead.count == decoded\_tokens | Missing instrumentation |
| GDP-005 | Brick ordering | LmHead > Gate > RmsNorm (per-call) | Not Immediate sync |
| GDP-006 | Token accounting | profiler.total\_tokens >> decoded\_tokens | API misuse risk |
| GDP-007 | Coverage bound | sum(brick\_total) ≤ wall\_clock | Timer corruption |
| GDP-008 | Reproducibility | CV < 15% across 3 runs | Clock instability |
| GDP-009 | Report actual\_us | LmHead ~595µs not ~1.9µs | Wrong value source (B1) |
| GDP-010 | No hardcoded scores | Not all scores == 100 | compute\_brick\_score not called (B3) |
| GDP-011 | Brick count | 11 entries for Qwen 1.5B | Zip truncation (B6) |
| GDP-012 | Falsification accounting | total == len(brick\_scores) | Hardcoded summary (B10-B12) |
| GDP-013 | Decoded token denominator | LmHead per\_decoded\_tok\_us > 100µs | Element denominator (B2) |
| GDP-014 | No magic constants | rust/tdg/cuda scores == 0.0 | Hardcoded 173.9/98.1/95.2 (B7-B9) |
| GDP-015 | No 137 constant | total\_points != 137 | Hardcoded FalsificationSummary (B10) |

## Cross-Repository Data Flow

```
trueno::BrickProfiler
  ├── BrickStats { name, count, total_ns, avg_us() }
  ├── SyncMode { Immediate, Deferred }
  └── all_brick_stats() → Iterator<&BrickStats>
        │
        ▼
realizar::CudaExecutor
  ├── start_brick_id() / stop_brick_id() — per-kernel gates
  ├── profiler.is_enabled() — C-GDP-001 graph bypass
  └── forward_graphed_decode.rs — eager path when profiling
        │
        ▼
aprender::OwnedQuantizedModelCuda
  ├── profiler() → &BrickProfiler
  ├── enable_profiling() / set_profiler_sync_mode(Immediate)
  └── delegates to CudaExecutor
        │
        ▼
apr-cli::commands
  ├── gguf.rs::brick_scores_from_profiler() — BrickStats → BrickScore
  ├── cbtop_measure_batch.rs::build_and_output_report() — BrickScore → JSON
  └── cbtop_get_cpu_memory.rs — simulated path (same invariants)
```

## Bug Taxonomy (18 Bugs Found)

| Group | File | Bugs | Description |
|-------|------|------|-------------|
| B1-B5 | gguf.rs | 5 | Hardcoded BrickScore fields, wrong denominator |
| B6 | cbtop\_measure\_batch.rs | 1 | 7-weight zip truncation |
| B7-B9 | cbtop\_measure\_batch.rs | 3 | Hardcoded PMAT scores |
| B10-B12 | cbtop\_measure\_batch.rs | 3 | Hardcoded FalsificationSummary |
| B13 | cbtop\_measure\_batch.rs | 1 | Hardcoded target/status |
| B6b | cbtop\_get\_cpu\_memory.rs | 1 | Same zip truncation (simulated) |
| B14-B18 | cbtop\_get\_cpu\_memory.rs | 4 | Same hardcoded values (simulated) |

## Verification Status

All 15 falsification tests: **PASS** (verified 2026-03-06 on RTX 4090)

| PR | Repository | Status |
|----|-----------|--------|
| [realizar#137](https://github.com/paiml/realizar/pull/137) | realizar | C-GDP-001 eager decode |
| [aprender#426](https://github.com/paiml/aprender/pull/426) | aprender | 18 bug fixes |
