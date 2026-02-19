//! Phase 5 falsification tests for matrix and attention kernels.
//!
//! Covers Matmul, Attention, GQA, and Flash Attention kernels.
//! 20 tests total: FALSIFY-MM-001..005, FALSIFY-ATT-001..005,
//! FALSIFY-GQ-001..006, FALSIFY-FA-001..004.

mod common;

use proptest::prelude::*;
use provable_contracts::kernels::attention::*;
use provable_contracts::kernels::flash_attention::*;
use provable_contracts::kernels::gqa::*;
use provable_contracts::kernels::matmul::*;

// ============================================================================
// Matmul (FALSIFY-MM-001 through FALSIFY-MM-005)
// ============================================================================

proptest! {
    /// FALSIFY-MM-001
    /// Contract: matmul-kernel-v1.yaml
    /// Prediction: output length always equals m * n
    /// Failure: output dimension mismatch
    #[test]
    fn falsify_mm_001_shape_correctness(
        m in 1usize..=8,
        p in 1usize..=8,
        n in 1usize..=8,
        a_vals in proptest::collection::vec(-5.0f32..5.0, 1..=64),
        b_vals in proptest::collection::vec(-5.0f32..5.0, 1..=64),
    ) {
        let a: Vec<f32> = a_vals.iter().copied().cycle().take(m * p).collect();
        let b: Vec<f32> = b_vals.iter().copied().cycle().take(p * n).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, p, n, &mut c);
        prop_assert_eq!(
            c.len(),
            m * n,
            "FALSIFY-MM-001: output length {} != m*n = {}",
            c.len(),
            m * n
        );
    }

    /// FALSIFY-MM-002
    /// Contract: matmul-kernel-v1.yaml
    /// Prediction: (A*B)*C approximately equals A*(B*C) for small matrices
    /// Failure: floating-point associativity violation beyond tolerance
    #[test]
    fn falsify_mm_002_associativity(
        a_vals in proptest::collection::vec(-2.0f32..2.0, 4..=4),
        b_vals in proptest::collection::vec(-2.0f32..2.0, 4..=4),
        c_vals in proptest::collection::vec(-2.0f32..2.0, 4..=4),
    ) {
        let n = 2usize;
        // (A * B) * C
        let mut ab = vec![0.0f32; n * n];
        matmul_scalar(&a_vals, &b_vals, n, n, n, &mut ab);
        let mut ab_c = vec![0.0f32; n * n];
        matmul_scalar(&ab, &c_vals, n, n, n, &mut ab_c);

        // A * (B * C)
        let mut bc = vec![0.0f32; n * n];
        matmul_scalar(&b_vals, &c_vals, n, n, n, &mut bc);
        let mut a_bc = vec![0.0f32; n * n];
        matmul_scalar(&a_vals, &bc, n, n, n, &mut a_bc);

        let dist = common::l2_distance(&ab_c, &a_bc);
        prop_assert!(
            dist < 1e-3,
            "FALSIFY-MM-002: associativity violated, L2 distance = {dist}"
        );
    }

    /// FALSIFY-MM-003
    /// Contract: matmul-kernel-v1.yaml
    /// Prediction: matmul(A, alpha*B) approximately equals alpha * matmul(A, B)
    /// Failure: linearity property violated
    #[test]
    fn falsify_mm_003_linearity(
        m in 1usize..=4,
        p in 1usize..=4,
        n in 1usize..=4,
        alpha in -3.0f32..3.0,
        a_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        b_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
    ) {
        let a: Vec<f32> = a_vals.iter().copied().cycle().take(m * p).collect();
        let b: Vec<f32> = b_vals.iter().copied().cycle().take(p * n).collect();

        // matmul(A, alpha * B)
        let alpha_b: Vec<f32> = b.iter().map(|x| alpha * x).collect();
        let mut c_alpha_b = vec![0.0f32; m * n];
        matmul_scalar(&a, &alpha_b, m, p, n, &mut c_alpha_b);

        // alpha * matmul(A, B)
        let mut c_ab = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, p, n, &mut c_ab);
        let alpha_c_ab: Vec<f32> = c_ab.iter().map(|x| alpha * x).collect();

        let dist = common::l2_distance(&c_alpha_b, &alpha_c_ab);
        prop_assert!(
            dist < 1e-3,
            "FALSIFY-MM-003: linearity violated, L2 distance = {dist}"
        );
    }

    /// FALSIFY-MM-004
    /// Contract: matmul-kernel-v1.yaml
    /// Prediction: AVX2 output matches scalar within 4 ULP
    /// Failure: SIMD divergence exceeds tolerance
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_mm_004_simd_equivalence(
        m in 1usize..=4,
        p in 1usize..=4,
        n in 1usize..=4,
        a_vals in proptest::collection::vec(-5.0f32..5.0, 1..=16),
        b_vals in proptest::collection::vec(-5.0f32..5.0, 1..=16),
    ) {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return Ok(());
        }
        let a: Vec<f32> = a_vals.iter().copied().cycle().take(m * p).collect();
        let b: Vec<f32> = b_vals.iter().copied().cycle().take(p * n).collect();

        let mut scalar_out = vec![0.0f32; m * n];
        let mut avx2_out = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, p, n, &mut scalar_out);
        unsafe { matmul_avx2(&a, &b, m, p, n, &mut avx2_out) };

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-MM-004: SIMD divergence = {ulp} ULP (max 4)"
        );
    }
}

/// FALSIFY-MM-005
/// Contract: matmul-kernel-v1.yaml
/// Prediction: A * I = A for any matrix A
/// Failure: identity multiplication changes matrix
#[test]
fn falsify_mm_005_identity() {
    let sizes = [1, 2, 3, 4, 5];
    for &n in &sizes {
        let identity = common::identity_matrix(n);
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.7 - 1.5).collect();
        let mut c = vec![0.0f32; n * n];
        matmul_scalar(&a, &identity, n, n, n, &mut c);
        let ulp = common::max_ulp_distance(&a, &c);
        assert!(
            ulp == 0,
            "FALSIFY-MM-005: A * I != A for n={n}, max ULP = {ulp}"
        );
    }
}

// ============================================================================
// Attention (FALSIFY-ATT-001 through FALSIFY-ATT-005)
// ============================================================================

proptest! {
    /// FALSIFY-ATT-001
    /// Contract: attention-kernel-v1.yaml
    /// Prediction: attention output is bounded by min/max of V values (convex combination)
    /// Failure: output outside V value range indicates broken softmax normalization
    #[test]
    fn falsify_att_001_weight_normalization(
        n in 2usize..=4,
        m in 2usize..=4,
        d_k in 2usize..=4,
        d_v in 2usize..=4,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(m * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(m * d_v).collect();
        let mut output = vec![0.0f32; n * d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

        // Each output column should be within the range of the corresponding V column
        for j in 0..d_v {
            let v_col_min = (0..m).map(|r| v[r * d_v + j]).fold(f32::INFINITY, f32::min);
            let v_col_max = (0..m).map(|r| v[r * d_v + j]).fold(f32::NEG_INFINITY, f32::max);
            for i in 0..n {
                let val = output[i * d_v + j];
                prop_assert!(
                    val >= v_col_min - 1e-5 && val <= v_col_max + 1e-5,
                    "FALSIFY-ATT-001: output[{i},{j}] = {val} outside V range [{v_col_min}, {v_col_max}]"
                );
            }
        }
    }

    /// FALSIFY-ATT-002
    /// Contract: attention-kernel-v1.yaml
    /// Prediction: each output element is between min(V) and max(V)
    /// Failure: convexity property violated
    #[test]
    fn falsify_att_002_convexity(
        n in 2usize..=3,
        m in 2usize..=4,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-5.0f32..5.0, 1..=9),
        k_vals in proptest::collection::vec(-5.0f32..5.0, 1..=12),
        v_vals in proptest::collection::vec(-5.0f32..5.0, 1..=12),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(m * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(m * d_v).collect();
        let mut output = vec![0.0f32; n * d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

        let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        for (idx, &val) in output.iter().enumerate() {
            prop_assert!(
                val >= v_min - 1e-5 && val <= v_max + 1e-5,
                "FALSIFY-ATT-002: output[{idx}] = {val} outside global V range [{v_min}, {v_max}]"
            );
        }
    }
}

/// FALSIFY-ATT-003
/// Contract: attention-kernel-v1.yaml
/// Prediction: scaling factor 1/sqrt(d_k) changes output when d_k changes
/// Failure: scaling dimension has no effect on attention output
#[test]
fn falsify_att_003_scaling() {
    let n = 2;
    let m = 2;

    // Use the same raw data but interpret with different d_k
    // d_k=1, d_v=2: Q is 2x1, K is 2x1, V is 2x2
    let q_dk1 = vec![1.0, 2.0]; // 2 queries, each dim 1
    let k_dk1 = vec![1.0, 2.0]; // 2 keys, each dim 1
    let v = vec![1.0, 2.0, 3.0, 4.0]; // 2x2

    let mut out_dk1 = vec![0.0f32; n * 2];
    attention_scalar(&q_dk1, &k_dk1, &v, n, m, 1, 2, &mut out_dk1);

    // d_k=2, d_v=2: Q is 2x2, K is 2x2, V is 2x2
    let q_dk2 = vec![1.0, 0.5, 2.0, 0.5]; // 2 queries, each dim 2
    let k_dk2 = vec![1.0, 0.5, 2.0, 0.5]; // 2 keys, each dim 2
    let mut out_dk2 = vec![0.0f32; n * 2];
    attention_scalar(&q_dk2, &k_dk2, &v, n, m, 2, 2, &mut out_dk2);

    // Outputs should differ because of the 1/sqrt(d_k) scaling
    let diff: f32 = out_dk1
        .iter()
        .zip(out_dk2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "FALSIFY-ATT-003: outputs identical despite different d_k scaling, diff = {diff}"
    );
}

proptest! {
    /// FALSIFY-ATT-004
    /// Contract: attention-kernel-v1.yaml
    /// Prediction: AVX2 output matches scalar within 8 ULP
    /// Failure: SIMD divergence exceeds tolerance
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_att_004_simd_equivalence(
        n in 2usize..=3,
        m in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=9),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=9),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=9),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(m * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(m * d_v).collect();

        let mut scalar_out = vec![0.0f32; n * d_v];
        let mut avx2_out = vec![0.0f32; n * d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut scalar_out);
        unsafe { attention_avx2(&q, &k, &v, n, m, d_k, d_v, &mut avx2_out) };

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-ATT-004: SIMD divergence = {ulp} ULP (max 8)"
        );
    }

    /// FALSIFY-ATT-005
    /// Contract: attention-kernel-v1.yaml
    /// Prediction: all outputs finite and bounded by max(|V|) * n
    /// Failure: output overflow or NaN
    #[test]
    fn falsify_att_005_output_bounds(
        n in 2usize..=4,
        m in 2usize..=4,
        d_k in 2usize..=4,
        d_v in 2usize..=4,
        q_vals in proptest::collection::vec(-5.0f32..5.0, 1..=16),
        k_vals in proptest::collection::vec(-5.0f32..5.0, 1..=16),
        v_vals in proptest::collection::vec(-5.0f32..5.0, 1..=16),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(m * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(m * d_v).collect();
        let mut output = vec![0.0f32; n * d_v];

        attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut output);

        common::assert_all_finite(&output);

        let v_max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let loose_bound = v_max_abs * (n as f32);
        for (idx, &val) in output.iter().enumerate() {
            prop_assert!(
                val.abs() <= loose_bound + 1e-5,
                "FALSIFY-ATT-005: |output[{idx}]| = {} exceeds loose bound {loose_bound}",
                val.abs()
            );
        }
    }
}

// ============================================================================
// GQA (FALSIFY-GQ-001 through FALSIFY-GQ-006)
// ============================================================================

proptest! {
    /// FALSIFY-GQ-001
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: output bounded by V range (convex combination)
    /// Failure: output outside V value range
    #[test]
    fn falsify_gq_001_weight_normalization(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=12),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
    ) {
        let num_heads = 2usize;
        let num_kv_heads = 1usize;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(num_heads * seq_len * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_v).collect();
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

        let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        for (idx, &val) in output.iter().enumerate() {
            prop_assert!(
                val >= v_min - 1e-5 && val <= v_max + 1e-5,
                "FALSIFY-GQ-001: output[{idx}] = {val} outside V range [{v_min}, {v_max}]"
            );
        }
    }

    /// FALSIFY-GQ-002
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: with num_heads=2, num_kv_heads=1, both heads use same KV but different Q
    /// Failure: KV broadcasting broken (heads produce identical output despite different Q)
    #[test]
    fn falsify_gq_002_kv_head_broadcasting(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
    ) {
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_stride = seq_len * d_v;

        // Generate Q with distinctly different values per head
        let q: Vec<f32> = (0..num_heads * seq_len * d_k)
            .map(|i| if i < seq_len * d_k { (i as f32) * 0.3 } else { (i as f32) * 0.7 + 1.0 })
            .collect();
        let k: Vec<f32> = (0..num_kv_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.2)
            .collect();
        let v: Vec<f32> = (0..num_kv_heads * seq_len * d_v)
            .map(|i| (i as f32) * 0.5 - 1.0)
            .collect();
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

        // Both heads should produce finite output
        common::assert_all_finite(&output);

        // Both heads use same KV but different Q, so outputs should generally differ
        let head0 = &output[0..head_stride];
        let head1 = &output[head_stride..2 * head_stride];
        let diff = common::l2_distance(head0, head1);
        prop_assert!(
            diff > 1e-7,
            "FALSIFY-GQ-002: heads produced identical output despite different Q, diff = {diff}"
        );
    }

    /// FALSIFY-GQ-003
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: when num_heads == num_kv_heads, GQA equals standard attention per head
    /// Failure: MHA equivalence broken
    #[test]
    fn falsify_gq_003_mha_equivalence(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
    ) {
        let num_heads = 1usize;
        let num_kv_heads = 1usize;

        let q: Vec<f32> = (0..num_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.15)
            .collect();
        let k: Vec<f32> = (0..num_kv_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.2)
            .collect();
        let v: Vec<f32> = (0..num_kv_heads * seq_len * d_v)
            .map(|i| (i as f32) * 0.25)
            .collect();

        // GQA with num_heads == num_kv_heads == 1
        let mut gqa_out = vec![0.0f32; num_heads * seq_len * d_v];
        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut gqa_out);

        // Standard attention: q is seq_len x d_k, k is seq_len x d_k, v is seq_len x d_v
        let mut att_out = vec![0.0f32; seq_len * d_v];
        attention_scalar(&q, &k, &v, seq_len, seq_len, d_k, d_v, &mut att_out);

        let dist = common::l2_distance(&gqa_out, &att_out);
        prop_assert!(
            dist < 1e-5,
            "FALSIFY-GQ-003: GQA != standard attention, L2 distance = {dist}"
        );
    }

    /// FALSIFY-GQ-004
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: output values within per-column V range (convexity)
    /// Failure: convex combination property violated
    #[test]
    fn falsify_gq_004_convexity(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-4.0f32..4.0, 1..=18),
        k_vals in proptest::collection::vec(-4.0f32..4.0, 1..=6),
        v_vals in proptest::collection::vec(-4.0f32..4.0, 1..=6),
    ) {
        let num_heads = 2usize;
        let num_kv_heads = 2usize;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(num_heads * seq_len * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_v).collect();
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

        // For each head, output is convex combination of V rows from the mapped KV head
        let heads_per_kv = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let v_start = kv_head * seq_len * d_v;
            let v_slice = &v[v_start..v_start + seq_len * d_v];

            for j in 0..d_v {
                let v_col_min = (0..seq_len).map(|r| v_slice[r * d_v + j]).fold(f32::INFINITY, f32::min);
                let v_col_max = (0..seq_len).map(|r| v_slice[r * d_v + j]).fold(f32::NEG_INFINITY, f32::max);
                for i in 0..seq_len {
                    let val = output[h * seq_len * d_v + i * d_v + j];
                    prop_assert!(
                        val >= v_col_min - 1e-5 && val <= v_col_max + 1e-5,
                        "FALSIFY-GQ-004: head {h} output[{i},{j}] = {val} outside V column range [{v_col_min}, {v_col_max}]"
                    );
                }
            }
        }
    }

    /// FALSIFY-GQ-005
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: AVX2 output matches scalar within 8 ULP
    /// Failure: SIMD divergence exceeds tolerance
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_gq_005_simd_equivalence(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=12),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let num_heads = 2usize;
        let num_kv_heads = 1usize;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(num_heads * seq_len * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_v).collect();

        let mut scalar_out = vec![0.0f32; num_heads * seq_len * d_v];
        let mut avx2_out = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut scalar_out);
        unsafe { gqa_avx2(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut avx2_out) };

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-GQ-005: SIMD divergence = {ulp} ULP (max 8)"
        );
    }
}

/// FALSIFY-GQ-006
/// Contract: gqa-kernel-v1.yaml
/// Prediction: num_heads must be divisible by num_kv_heads; valid combos work
/// Failure: valid head divisibility combinations rejected
#[test]
fn falsify_gq_006_head_divisibility() {
    let valid_combos: &[(usize, usize)] = &[(2, 1), (4, 2), (4, 1), (1, 1), (6, 3)];
    let seq_len = 2;
    let d_k = 2;
    let d_v = 2;

    for &(num_heads, num_kv_heads) in valid_combos {
        let q = vec![0.5f32; num_heads * seq_len * d_k];
        let k = vec![0.5f32; num_kv_heads * seq_len * d_k];
        let v = vec![1.0f32; num_kv_heads * seq_len * d_v];
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        // Should not panic
        gqa_scalar(
            &q,
            &k,
            &v,
            seq_len,
            d_k,
            d_v,
            num_heads,
            num_kv_heads,
            &mut output,
        );

        common::assert_all_finite(&output);

        // V is all 1.0 and attention weights sum to 1 per row,
        // so each output element should be 1.0
        for (idx, &val) in output.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "FALSIFY-GQ-006: output[{idx}] = {val} (expected 1.0) for heads={num_heads}, kv_heads={num_kv_heads}"
            );
        }
    }
}

// ============================================================================
// Flash Attention (FALSIFY-FA-001 through FALSIFY-FA-004)
// ============================================================================

proptest! {
    /// FALSIFY-FA-001
    /// Contract: flash-attention-v1.yaml
    /// Prediction: flash_attention matches standard attention for small inputs
    /// Failure: online softmax produces different result than standard softmax
    #[test]
    fn falsify_fa_001_matches_standard(
        n in 2usize..=4,
        d in 2usize..=4,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();

        // Flash attention with tile_size = n (single tile)
        let mut flash_out = vec![0.0f32; n * d];
        flash_attention_scalar(&q, &k, &v, n, d, n, &mut flash_out);

        // Standard attention (self-attention: n queries, n keys)
        let mut expected = vec![0.0f32; n * d];
        attention_scalar(&q, &k, &v, n, n, d, d, &mut expected);

        let dist = common::l2_distance(&flash_out, &expected);
        prop_assert!(
            dist < 1e-4,
            "FALSIFY-FA-001: flash vs standard attention L2 distance = {dist}"
        );
    }

    /// FALSIFY-FA-002
    /// Contract: flash-attention-v1.yaml
    /// Prediction: output is finite for moderate inputs (online softmax does not overflow)
    /// Failure: NaN or Inf in output
    #[test]
    fn falsify_fa_002_online_softmax(
        n in 2usize..=6,
        d in 2usize..=4,
        tile_size in 1usize..=4,
        q_vals in proptest::collection::vec(-5.0f32..5.0, 1..=24),
        k_vals in proptest::collection::vec(-5.0f32..5.0, 1..=24),
        v_vals in proptest::collection::vec(-5.0f32..5.0, 1..=24),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();
        let mut output = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

        for (idx, &val) in output.iter().enumerate() {
            prop_assert!(
                val.is_finite(),
                "FALSIFY-FA-002: output[{idx}] = {val} is not finite (tile_size={tile_size})"
            );
        }
    }

    /// FALSIFY-FA-003
    /// Contract: flash-attention-v1.yaml
    /// Prediction: tile_size < n produces same output as single-tile (tile_size = n)
    /// Failure: tiled computation diverges from single-tile computation
    #[test]
    fn falsify_fa_003_tile_coverage(
        d in 2usize..=4,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=32),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=32),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=32),
    ) {
        let n = 8usize;
        let tile_small = 4usize;
        let tile_full = n;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();

        let mut out_small = vec![0.0f32; n * d];
        let mut out_full = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_small, &mut out_small);
        flash_attention_scalar(&q, &k, &v, n, d, tile_full, &mut out_full);

        let dist = common::l2_distance(&out_small, &out_full);
        prop_assert!(
            dist < 1e-4,
            "FALSIFY-FA-003: tile_size={tile_small} vs tile_size={tile_full} L2 distance = {dist}"
        );
    }

    /// FALSIFY-FA-004
    /// Contract: flash-attention-v1.yaml
    /// Prediction: output bounded by V range (convex combination of V rows)
    /// Failure: output outside V value range
    #[test]
    fn falsify_fa_004_output_conservation(
        n in 4usize..=8,
        d in 2usize..=4,
        tile_size in 2usize..=4,
        q_vals in proptest::collection::vec(-5.0f32..5.0, 1..=32),
        k_vals in proptest::collection::vec(-5.0f32..5.0, 1..=32),
        v_vals in proptest::collection::vec(-5.0f32..5.0, 1..=32),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();
        let mut output = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

        // Per-column convexity: output in each column bounded by V column range
        for j in 0..d {
            let v_col_min = (0..n).map(|r| v[r * d + j]).fold(f32::INFINITY, f32::min);
            let v_col_max = (0..n).map(|r| v[r * d + j]).fold(f32::NEG_INFINITY, f32::max);
            for i in 0..n {
                let val = output[i * d + j];
                prop_assert!(
                    val >= v_col_min - 1e-5 && val <= v_col_max + 1e-5,
                    "FALSIFY-FA-004: output[{i},{j}] = {val} outside V column range [{v_col_min}, {v_col_max}]"
                );
            }
        }
    }
}
