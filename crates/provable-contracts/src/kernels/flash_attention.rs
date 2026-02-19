//! Flash Attention: IO-aware tiled attention.
//!
//! Matches `flash-attention-v1.yaml`.
//! Online softmax with running max and running sum per tile.
//!
//! Each function provides one of three backends:
//! - `fn flash_attention_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn flash_attention_avx2(...)` -- AVX2 SIMD implementation
//! - `fn flash_attention_ptx() -> &'static str` -- PTX assembly source string

use super::ops;

// ────────────────────────────────────────────────────────────────────────────
// Naive attention helper (for comparison in tests)
// ────────────────────────────────────────────────────────────────────────────

/// Naive scaled dot-product self-attention (for verification only).
///
/// Q, K, V are all n x d. Output is n x d.
#[cfg(test)]
fn naive_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize, output: &mut [f32]) {
    let mut scores = vec![0.0f32; n * n];
    ops::score_matrix(q, k, n, n, d, &mut scores);
    ops::softmax_rows(&mut scores, n, n);
    ops::matmul_sv(&scores, v, n, n, d, output);
}

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Flash Attention: IO-aware tiled attention with online softmax.
///
/// Q, K, V are all n x d (row-major). Output is n x d.
///
/// For each query row i, processes KV in tiles of `tile_size` rows, maintaining
/// a running max and running sum for numerically stable online softmax.
///
/// # Panics
/// Panics if `q.len() != n*d`, `k.len() != n*d`, `v.len() != n*d`,
/// `output.len() != n*d`, or `tile_size == 0`.
pub fn flash_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    tile_size: usize,
    output: &mut [f32],
) {
    assert_eq!(q.len(), n * d, "Q dimension mismatch: expected {} got {}", n * d, q.len());
    assert_eq!(k.len(), n * d, "K dimension mismatch: expected {} got {}", n * d, k.len());
    assert_eq!(v.len(), n * d, "V dimension mismatch: expected {} got {}", n * d, v.len());
    assert_eq!(
        output.len(),
        n * d,
        "output dimension mismatch: expected {} got {}",
        n * d,
        output.len()
    );
    assert!(tile_size > 0, "tile_size must be > 0");

    let scale = 1.0 / (d as f32).sqrt();

    for i in 0..n {
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; d];

        // Process KV in tiles
        let mut tile_start = 0;
        while tile_start < n {
            let tile_end = (tile_start + tile_size).min(n);
            process_tile(
                q, k, v, i, d, scale, tile_start, tile_end,
                &mut running_max, &mut running_sum, &mut acc,
            );
            tile_start = tile_end;
        }

        // Normalize
        normalize_row(&acc, running_sum, &mut output[i * d..(i + 1) * d]);
    }
}

/// Process a single tile of KV for one query row, updating online softmax state.
#[allow(clippy::too_many_arguments)]
fn process_tile(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    i: usize,
    d: usize,
    scale: f32,
    tile_start: usize,
    tile_end: usize,
    running_max: &mut f32,
    running_sum: &mut f32,
    acc: &mut [f32],
) {
    let tile_len = tile_end - tile_start;

    // Compute scores for this tile: Q[i] . K[j] / sqrt(d)
    let mut tile_scores = vec![0.0f32; tile_len];
    let q_row = &q[i * d..(i + 1) * d];
    for (tj, j) in (tile_start..tile_end).enumerate() {
        tile_scores[tj] = ops::dot(q_row, &k[j * d..(j + 1) * d]) * scale;
    }

    // Find max of this tile
    let tile_max = tile_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let new_max = running_max.max(tile_max);

    // Correction factor: rescale previous accumulation
    let correction = (*running_max - new_max).exp();
    for a in acc.iter_mut() {
        *a *= correction;
    }
    *running_sum *= correction;

    // Accumulate this tile
    for (tj, j) in (tile_start..tile_end).enumerate() {
        let w = (tile_scores[tj] - new_max).exp();
        ops::weighted_accumulate(acc, w, &v[j * d..(j + 1) * d]);
        *running_sum += w;
    }

    *running_max = new_max;
}

/// Normalize accumulated attention output by the softmax denominator.
fn normalize_row(acc: &[f32], running_sum: f32, output: &mut [f32]) {
    if running_sum > 0.0 {
        for (o, a) in output.iter_mut().zip(acc.iter()) {
            *o = a / running_sum;
        }
    } else {
        for o in output.iter_mut() {
            *o = 0.0;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 Flash Attention -- delegates to scalar.
///
/// The tiled online softmax algorithm is inherently sequential along the tile
/// dimension; the scalar implementation is the reference.
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if dimensions do not match or `tile_size == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn flash_attention_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    tile_size: usize,
    output: &mut [f32],
) {
    flash_attention_scalar(q, k, v, n, d, tile_size, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for Flash Attention.
///
/// One block per query row. Tiled loop over KV with shared memory, online
/// softmax maintaining running max and sum across tiles.
pub fn flash_attention_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry flash_attention_kernel(
    .param .u64 Q,
    .param .u64 K,
    .param .u64 V,
    .param .u64 OUT,
    .param .u32 N,
    .param .u32 D,
    .param .u32 TILE_SIZE
) {
    .reg .u32 %tid, %bid, %n, %d, %tile_size;
    .reg .u32 %i, %j, %kk, %dd, %tile_start, %tile_end, %tile_len;
    .reg .u32 %tmp32;
    .reg .u64 %q_ptr, %k_ptr, %v_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %dot, %q_val, %k_val, %v_val, %scale;
    .reg .f32 %score, %tile_max, %new_max, %running_max, %running_sum;
    .reg .f32 %correction, %weight, %acc, %d_f, %rcp_sum;
    .reg .pred %p_tile, %p_j, %p_k, %p_d, %p_end;
    .shared .f32 tile_scores[256];
    .shared .f32 tile_v[4096];

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    // Load params
    ld.param.u32 %n, [N];
    ld.param.u32 %d, [D];
    ld.param.u32 %tile_size, [TILE_SIZE];
    ld.param.u64 %q_ptr, [Q];
    ld.param.u64 %k_ptr, [K];
    ld.param.u64 %v_ptr, [V];
    ld.param.u64 %out_ptr, [OUT];

    // bid = query row index (i)
    // tid = thread within block (used to parallelize over D)

    // Scale = 1/sqrt(d)
    cvt.rn.f32.u32 %d_f, %d;
    sqrt.approx.f32 %scale, %d_f;
    rcp.approx.f32 %scale, %scale;

    // Online softmax state
    mov.f32 %running_max, 0fFF7FFFFF;
    mov.f32 %running_sum, 0f00000000;
    mov.f32 %acc, 0f00000000;

    // Tile loop
    mov.u32 %tile_start, 0;
TILE_LOOP:
    setp.ge.u32 %p_tile, %tile_start, %n;
    @%p_tile bra TILE_DONE;

    // tile_end = min(tile_start + tile_size, n)
    add.u32 %tile_end, %tile_start, %tile_size;
    min.u32 %tile_end, %tile_end, %n;

    bar.sync 0;

    // Phase 1: Compute tile_scores[j-tile_start] = Q[bid] . K[j] / sqrt(d)
    // Thread 0 computes all scores for simplicity (small tiles)
    setp.ne.u32 %p_j, %tid, 0;
    @%p_j bra SCORES_DONE;

    mov.f32 %tile_max, 0fFF7FFFFF;
    mov.u32 %j, %tile_start;
SCORE_LOOP:
    setp.ge.u32 %p_j, %j, %tile_end;
    @%p_j bra SCORE_DONE;

    mov.f32 %dot, 0f00000000;
    mov.u32 %kk, 0;
DOT_LOOP:
    setp.ge.u32 %p_k, %kk, %d;
    @%p_k bra DOT_DONE;

    // Q[bid * d + kk]
    mad.lo.u32 %tmp32, %bid, %d, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %q_ptr, %off64;
    ld.global.f32 %q_val, [%addr];

    // K[j * d + kk]
    mad.lo.u32 %tmp32, %j, %d, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %k_ptr, %off64;
    ld.global.f32 %k_val, [%addr];

    fma.rn.f32 %dot, %q_val, %k_val, %dot;
    add.u32 %kk, %kk, 1;
    bra DOT_LOOP;
DOT_DONE:

    mul.f32 %score, %dot, %scale;

    // Store in shared memory
    sub.u32 %tmp32, %j, %tile_start;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %off64, tile_scores;
    st.shared.f32 [%addr], %score;

    max.f32 %tile_max, %tile_max, %score;

    add.u32 %j, %j, 1;
    bra SCORE_LOOP;
SCORE_DONE:

    // Store tile_max in scores[255] for other threads to read
    mul.wide.u32 %off64, 255, 4;
    add.u64 %addr, %off64, tile_scores;
    st.shared.f32 [%addr], %tile_max;

SCORES_DONE:
    bar.sync 0;

    // All threads read tile_max
    mul.wide.u32 %off64, 255, 4;
    add.u64 %addr, %off64, tile_scores;
    ld.shared.f32 %tile_max, [%addr];

    // new_max = max(running_max, tile_max)
    max.f32 %new_max, %running_max, %tile_max;

    // correction = exp(running_max - new_max)
    sub.f32 %correction, %running_max, %new_max;
    mul.f32 %correction, %correction, 0f3FB8AA3B;
    ex2.approx.f32 %correction, %correction;

    // Rescale accumulator and running_sum
    mul.f32 %acc, %acc, %correction;
    mul.f32 %running_sum, %running_sum, %correction;

    // Accumulate this tile: for each j in tile
    // Each thread handles its own dimension(s) for the V weighted sum
    // For simplicity, thread 0 accumulates running_sum, all threads accumulate acc for their dim
    mov.u32 %j, %tile_start;
ACC_LOOP:
    setp.ge.u32 %p_j, %j, %tile_end;
    @%p_j bra ACC_DONE;

    // Load score for this j
    sub.u32 %tmp32, %j, %tile_start;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %off64, tile_scores;
    ld.shared.f32 %score, [%addr];

    // weight = exp(score - new_max)
    sub.f32 %weight, %score, %new_max;
    mul.f32 %weight, %weight, 0f3FB8AA3B;
    ex2.approx.f32 %weight, %weight;

    // acc += weight * V[j * d + tid] (each thread handles one dimension)
    setp.ge.u32 %p_d, %tid, %d;
    @%p_d bra SKIP_V;

    mad.lo.u32 %tmp32, %j, %d, %tid;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %v_ptr, %off64;
    ld.global.f32 %v_val, [%addr];
    fma.rn.f32 %acc, %weight, %v_val, %acc;

SKIP_V:
    // Thread 0 accumulates running_sum
    setp.ne.u32 %p_d, %tid, 0;
    @%p_d bra SKIP_SUM;
    add.f32 %running_sum, %running_sum, %weight;
SKIP_SUM:

    add.u32 %j, %j, 1;
    bra ACC_LOOP;
ACC_DONE:

    mov.f32 %running_max, %new_max;

    add.u32 %tile_start, %tile_start, %tile_size;
    bra TILE_LOOP;
TILE_DONE:

    bar.sync 0;

    // Normalize: output[bid * d + tid] = acc / running_sum
    setp.ge.u32 %p_d, %tid, %d;
    @%p_d bra EXIT;

    // Broadcast running_sum from thread 0 via shared memory
    setp.ne.u32 %p_d, %tid, 0;
    @%p_d bra LOAD_SUM;
    // Thread 0 stores running_sum
    mov.u64 %addr, tile_scores;
    st.shared.f32 [%addr], %running_sum;
LOAD_SUM:
    bar.sync 0;
    mov.u64 %addr, tile_scores;
    ld.shared.f32 %rcp_sum, [%addr];
    rcp.approx.f32 %rcp_sum, %rcp_sum;

    mul.f32 %acc, %acc, %rcp_sum;

    // Store output
    mad.lo.u32 %tmp32, %bid, %d, %tid;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %acc;

EXIT:
    ret;
}
"#
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ulp::assert_ulp_eq;
    use super::super::ops::{sequential_floats, patterned_floats};
    use proptest::prelude::*;

    // ── Flash attention matches naive attention ─────────────────────────

    #[test]
    fn test_flash_matches_naive_small() {
        let n = 4;
        let d = 3;
        let tile_size = 2;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-5,
                "mismatch at index {i}: flash={f} naive={nv} (diff={})",
                (f - nv).abs()
            );
        }
    }

    #[test]
    fn test_flash_matches_naive_larger() {
        let n = 8;
        let d = 4;
        let tile_size = 3;

        let q = patterned_floats(n * d, 7, 3.0, 0.5);
        let k = patterned_floats(n * d, 5, 2.0, 0.3);
        let v = patterned_floats(n * d, 11, 5.0, 0.2);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-4,
                "mismatch at index {i}: flash={f} naive={nv} (diff={})",
                (f - nv).abs()
            );
        }
    }

    // ── Single tile degrades to standard attention ──────────────────────

    #[test]
    fn test_flash_single_tile() {
        // When tile_size >= n, flash attention processes everything in one tile,
        // which should give the same result as naive attention.
        let n = 4;
        let d = 3;
        let tile_size = n + 10; // larger than n

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-6,
                "mismatch at index {i}: flash={f} naive={nv}"
            );
        }
    }

    // ── Tile size = 1 (extreme tiling) ──────────────────────────────────

    #[test]
    fn test_flash_tile_size_one() {
        let n = 5;
        let d = 2;
        let tile_size = 1;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.2);
        let v = sequential_floats(n * d, 0.15);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-5,
                "mismatch at index {i}: flash={f} naive={nv}"
            );
        }
    }

    // ── Single element ──────────────────────────────────────────────────

    #[test]
    fn test_flash_single_element() {
        let n = 1;
        let d = 3;
        let tile_size = 1;

        let q = vec![1.0, 2.0, 3.0];
        let k = vec![1.0, 0.0, 0.0];
        let v = vec![10.0, 20.0, 30.0];
        let mut output = vec![0.0f32; d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

        // Single query, single key: softmax of single score = 1.0, output = V
        assert_ulp_eq(&output, &v, 0);
    }

    // ── Dimension assertions ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Q dimension mismatch")]
    fn test_flash_bad_q_dim() {
        let mut output = vec![0.0f32; 4];
        flash_attention_scalar(&[1.0], &[1.0; 4], &[1.0; 4], 2, 2, 1, &mut output);
    }

    #[test]
    #[should_panic(expected = "tile_size must be > 0")]
    fn test_flash_zero_tile_size() {
        let mut output = vec![0.0f32; 4];
        flash_attention_scalar(&[1.0; 4], &[1.0; 4], &[1.0; 4], 2, 2, 0, &mut output);
    }

    // ── Property-based tests ────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_flash_matches_naive(
            n in 1usize..6,
            d in 1usize..5,
            tile_size in 1usize..8,
        ) {
            let q = patterned_floats(n*d, 7, 3.0, 0.3);
            let k = patterned_floats(n*d, 5, 2.0, 0.2);
            let v = patterned_floats(n*d, 11, 5.0, 0.15);

            let mut flash_out = vec![0.0f32; n * d];
            let mut naive_out = vec![0.0f32; n * d];

            flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
            naive_attention(&q, &k, &v, n, d, &mut naive_out);

            for idx in 0..n*d {
                let diff = (flash_out[idx] - naive_out[idx]).abs();
                prop_assert!(
                    diff < 1e-4,
                    "mismatch at {idx}: flash={} naive={} (diff={diff})",
                    flash_out[idx], naive_out[idx]
                );
            }
        }

        #[test]
        fn prop_flash_output_row_norms_bounded(
            n in 1usize..5,
            d in 1usize..4,
            tile_size in 1usize..6,
        ) {
            let q = sequential_floats(n*d, 0.1);
            let k = sequential_floats(n*d, 0.1);
            let v = sequential_floats(n*d, 0.1);
            let mut output = vec![0.0f32; n * d];

            flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

            // Output is convex combination of V rows, so each output row norm
            // should be <= max V row norm
            let max_v_norm: f32 = (0..n)
                .map(|r| {
                    (0..d).map(|c| v[r * d + c] * v[r * d + c]).sum::<f32>().sqrt()
                })
                .fold(0.0f32, f32::max);

            for i in 0..n {
                let row_norm: f32 = (0..d)
                    .map(|c| output[i * d + c] * output[i * d + c])
                    .sum::<f32>()
                    .sqrt();
                prop_assert!(
                    row_norm <= max_v_norm + 1e-4,
                    "output row {i} norm {row_norm} exceeds max V row norm {max_v_norm}"
                );
            }
        }
    }

    // ── AVX2 parity test ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_flash_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 6;
        let d = 4;
        let tile_size = 2;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut scalar_out = vec![0.0f32; n * d];
        let mut avx2_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut scalar_out);
        unsafe { flash_attention_avx2(&q, &k, &v, n, d, tile_size, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 8);
    }

    // ── PTX structural tests ────────────────────────────────────────────

    #[test]
    fn test_flash_attention_ptx_structure() {
        let ptx = flash_attention_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry flash_attention_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
        assert!(ptx.contains("bar.sync"), "missing barrier synchronization");
        assert!(ptx.contains("ex2.approx.f32"), "missing exp approximation");
        assert!(ptx.contains("fma.rn.f32"), "missing FMA instruction");
        assert!(ptx.contains("rcp.approx.f32"), "missing reciprocal for normalization");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(open, close, "unbalanced braces: {open} open vs {close} close");
    }

    #[test]
    fn test_flash_attention_ptx_nonempty() {
        assert!(!flash_attention_ptx().is_empty());
    }

    // ── Different tile sizes produce same result ────────────────────────

    #[test]
    fn test_flash_tile_size_invariance() {
        let n = 6;
        let d = 3;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut out_t1 = vec![0.0f32; n * d];
        let mut out_t2 = vec![0.0f32; n * d];
        let mut out_t3 = vec![0.0f32; n * d];
        let mut out_tall = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, 1, &mut out_t1);
        flash_attention_scalar(&q, &k, &v, n, d, 2, &mut out_t2);
        flash_attention_scalar(&q, &k, &v, n, d, 3, &mut out_t3);
        flash_attention_scalar(&q, &k, &v, n, d, n, &mut out_tall);

        for i in 0..n * d {
            assert!(
                (out_t1[i] - out_tall[i]).abs() < 1e-5,
                "tile_size=1 vs full: index {i}: {} vs {}",
                out_t1[i],
                out_tall[i]
            );
            assert!(
                (out_t2[i] - out_tall[i]).abs() < 1e-5,
                "tile_size=2 vs full: index {i}: {} vs {}",
                out_t2[i],
                out_tall[i]
            );
            assert!(
                (out_t3[i] - out_tall[i]).abs() < 1e-5,
                "tile_size=3 vs full: index {i}: {} vs {}",
                out_t3[i],
                out_tall[i]
            );
        }
    }
}
