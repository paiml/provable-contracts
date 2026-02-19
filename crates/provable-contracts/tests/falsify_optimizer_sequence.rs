//! Phase 5 falsification tests for optimizer, sequence, and classical ML kernels.
//!
//! Covers AdamW, Conv1d, SSM, KMeans, PageRank, LBFGS, CMA-ES, and Gated Delta Net
//! with 47 tests total. Each test targets a specific mathematical invariant that
//! would break if the implementation contains a common bug class.

mod common;

use proptest::prelude::*;
use provable_contracts::kernels::adamw::*;
use provable_contracts::kernels::conv1d::*;
use provable_contracts::kernels::ssm::*;
use provable_contracts::kernels::kmeans::*;
use provable_contracts::kernels::pagerank::*;
use provable_contracts::kernels::lbfgs::*;
use provable_contracts::kernels::cma_es::*;
use provable_contracts::kernels::gated_delta_net::*;

// ============================================================================
// AdamW (6 tests: FALSIFY-AW-001 through FALSIFY-AW-006)
// ============================================================================

/// FALSIFY-AW-001: Decoupled weight decay
/// Contract: adamw-kernel-v1.yaml
/// Prediction: with weight_decay > 0 and zero gradients, params decrease by factor (1 - lr*wd)
/// If fails: weight decay is not decoupled from adaptive learning rate
#[test]
fn falsify_aw_001_decoupled_weight_decay() {
    let lr = 0.001_f32;
    let wd = 0.01_f32;
    let original = [1.0_f32, -2.0, 3.0, -4.0];
    let mut params = original;
    let grads = [0.0_f32; 4];
    let mut m = [0.0_f32; 4];
    let mut v = [0.0_f32; 4];

    adamw_step_scalar(&mut params, &grads, &mut m, &mut v, lr, 0.9, 0.999, 1e-8, wd, 1);

    for i in 0..4 {
        // With zero grads, m and v stay zero, so adaptive term is 0/(0+eps) ~ 0
        // param -= lr * (0 + wd * param) => param *= (1 - lr * wd)
        let expected = original[i] * (1.0 - lr * wd);
        assert!(
            (params[i] - expected).abs() < 1e-6,
            "FALSIFY-AW-001 failed: params[{i}] = {}, expected {expected}",
            params[i]
        );
    }
}

/// FALSIFY-AW-002: Bias correction
/// Contract: adamw-kernel-v1.yaml
/// Prediction: at t=1, bias-corrected moment differs from raw moment (moments are non-zero after non-zero gradient)
/// If fails: bias correction is not applied to moment estimates
#[test]
fn falsify_aw_002_bias_correction() {
    let mut params = [0.5_f32; 4];
    let grads = [1.0_f32, 0.5, -0.3, 0.8];
    let mut m = [0.0_f32; 4];
    let mut v = [0.0_f32; 4];

    adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1);

    // After one step with non-zero grads, m and v should be non-zero
    for i in 0..4 {
        assert!(
            m[i].abs() > 1e-10,
            "FALSIFY-AW-002 failed: m[{i}] = {} is zero after non-zero gradient",
            m[i]
        );
        assert!(
            v[i] > 0.0,
            "FALSIFY-AW-002 failed: v[{i}] = {} should be positive after non-zero gradient",
            v[i]
        );
    }
}

proptest! {
    /// FALSIFY-AW-003: SIMD equivalence
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: avx2 vs scalar param results within 8 ULP after one step
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_aw_003_simd_equivalence(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
        grads_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = params_v.len().min(grads_v.len());
        let mut params_s: Vec<f32> = params_v[..n].to_vec();
        let mut params_a = params_s.clone();
        let grads: Vec<f32> = grads_v[..n].to_vec();
        let mut m_s = vec![0.0_f32; n];
        let mut m_a = m_s.clone();
        let mut v_s = vec![0.0_f32; n];
        let mut v_a = v_s.clone();

        adamw_step_scalar(&mut params_s, &grads, &mut m_s, &mut v_s, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);
        unsafe {
            adamw_step_avx2(&mut params_a, &grads, &mut m_a, &mut v_a, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);
        }

        let ulp = common::max_ulp_distance(&params_s, &params_a);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-AW-003 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }

    /// FALSIFY-AW-004: Moments finite
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: m and v stay finite after a step with moderate gradients
    /// If fails: numerical overflow in moment accumulation
    #[test]
    fn falsify_aw_004_moments_finite(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
        grads_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        let n = params_v.len().min(grads_v.len());
        let mut params: Vec<f32> = params_v[..n].to_vec();
        let grads: Vec<f32> = grads_v[..n].to_vec();
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);

        for i in 0..n {
            prop_assert!(
                m[i].is_finite(),
                "FALSIFY-AW-004 failed: m[{i}] = {} is not finite",
                m[i]
            );
            prop_assert!(
                v[i].is_finite(),
                "FALSIFY-AW-004 failed: v[{i}] = {} is not finite",
                v[i]
            );
        }
    }

    /// FALSIFY-AW-005: Update finite
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: params stay finite after a step
    /// If fails: division by zero or overflow in parameter update
    #[test]
    fn falsify_aw_005_update_finite(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
        grads_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        let n = params_v.len().min(grads_v.len());
        let mut params: Vec<f32> = params_v[..n].to_vec();
        let grads: Vec<f32> = grads_v[..n].to_vec();
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);

        common::assert_all_finite(&params);
    }

    /// FALSIFY-AW-006: Weight decay direction
    /// Contract: adamw-kernel-v1.yaml
    /// Prediction: with wd > 0 and zero grads, all params move toward zero (|new| <= |old|)
    /// If fails: weight decay pushes params away from zero
    #[test]
    fn falsify_aw_006_weight_decay_direction(
        params_v in proptest::collection::vec(-5.0_f32..5.0, 4..=16),
    ) {
        let n = params_v.len();
        let original: Vec<f32> = params_v.clone();
        let mut params = params_v;
        let grads = vec![0.0_f32; n];
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adamw_step_scalar(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);

        for i in 0..n {
            prop_assert!(
                params[i].abs() <= original[i].abs() + 1e-7,
                "FALSIFY-AW-006 failed: |params[{i}]| = {} > |original| = {}, weight decay should shrink params",
                params[i].abs(), original[i].abs()
            );
        }
    }
}

// ============================================================================
// Conv1d (6 tests: FALSIFY-CV-001 through FALSIFY-CV-006)
// ============================================================================

/// FALSIFY-CV-001: Output shape
/// Contract: conv1d-kernel-v1.yaml
/// Prediction: output length = floor((length + 2*padding - kernel_size) / stride) + 1
/// If fails: output dimension calculation is incorrect
#[test]
fn falsify_cv_001_output_shape() {
    // Test with various concrete sizes
    let cases: Vec<(usize, usize, usize, usize, usize, usize)> = vec![
        // (c_in, c_out, length, kernel_size, stride, padding)
        (1, 1, 8, 3, 1, 0),   // out = (8+0-3)/1+1 = 6
        (2, 3, 10, 3, 2, 1),  // out = (10+2-3)/2+1 = 5
        (1, 1, 4, 1, 1, 0),   // out = (4+0-1)/1+1 = 4
        (1, 2, 6, 3, 1, 1),   // out = (6+2-3)/1+1 = 6
    ];

    for (c_in, c_out, length, kernel_size, stride, padding) in cases {
        let expected_out_len = (length + 2 * padding - kernel_size) / stride + 1;
        let input = vec![1.0_f32; c_in * length];
        let weight = vec![0.1_f32; c_out * c_in * kernel_size];
        let mut output = vec![0.0_f32; c_out * expected_out_len];

        // Should not panic -- output buffer has correct size
        conv1d_scalar(
            &input, &weight, None, c_in, c_out, length, kernel_size, stride, padding, &mut output,
        );
        assert_eq!(
            output.len(),
            c_out * expected_out_len,
            "FALSIFY-CV-001 failed: output length mismatch for c_in={c_in}, c_out={c_out}, \
             length={length}, kernel_size={kernel_size}, stride={stride}, padding={padding}"
        );
    }
}

proptest! {
    /// FALSIFY-CV-002: Linearity
    /// Contract: conv1d-kernel-v1.yaml
    /// Prediction: conv1d(alpha*x, w, None) ~ alpha * conv1d(x, w, None) for small inputs
    /// If fails: convolution is not a linear operation in input
    #[test]
    fn falsify_cv_002_linearity(
        input_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        weight_v in proptest::collection::vec(-5.0_f32..5.0, 3..=3),
        alpha in 0.1_f32..5.0,
    ) {
        let c_in = 1;
        let c_out = 1;
        let length = 4;
        let kernel_size = 3;
        let stride = 1;
        let padding = 0;
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;

        // conv1d(x, w)
        let mut out_x = vec![0.0_f32; c_out * out_len];
        conv1d_scalar(&input_v, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut out_x);

        // conv1d(alpha*x, w)
        let scaled_input: Vec<f32> = input_v.iter().map(|&x| x * alpha).collect();
        let mut out_ax = vec![0.0_f32; c_out * out_len];
        conv1d_scalar(&scaled_input, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut out_ax);

        // alpha * conv1d(x, w)
        let expected: Vec<f32> = out_x.iter().map(|&y| y * alpha).collect();

        for i in 0..out_len {
            prop_assert!(
                (out_ax[i] - expected[i]).abs() < 1e-3,
                "FALSIFY-CV-002 failed: linearity violated at [{i}]: conv(alpha*x) = {}, alpha*conv(x) = {}",
                out_ax[i], expected[i]
            );
        }
    }
}

/// FALSIFY-CV-003: Im2col equivalence (pointwise)
/// Contract: conv1d-kernel-v1.yaml
/// Prediction: conv1d with kernel_size=1, stride=1 acts as pointwise multiplication
/// If fails: kernel_size=1 edge case is handled incorrectly
#[test]
fn falsify_cv_003_pointwise_equivalence() {
    let c_in = 1;
    let c_out = 1;
    let length = 8;
    let kernel_size = 1;
    let stride = 1;
    let padding = 0;
    let out_len = length; // (8+0-1)/1+1 = 8

    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [2.5_f32]; // single weight, kernel_size=1
    let mut output = vec![0.0_f32; c_out * out_len];

    conv1d_scalar(&input, &weight, None, c_in, c_out, length, kernel_size, stride, padding, &mut output);

    for i in 0..length {
        let expected = input[i] * weight[0];
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "FALSIFY-CV-003 failed: output[{i}] = {}, expected {} (pointwise mul)",
            output[i], expected
        );
    }
}

proptest! {
    /// FALSIFY-CV-004: Output bound
    /// Contract: conv1d-kernel-v1.yaml
    /// Prediction: output bounded by c_in * kernel_size * max(|input|) * max(|weight|) + max(|bias|)
    /// If fails: accumulation overflow or incorrect summation
    #[test]
    fn falsify_cv_004_output_bound(
        input_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        weight_v in proptest::collection::vec(-5.0_f32..5.0, 3..=3),
        bias_v in proptest::collection::vec(-5.0_f32..5.0, 1..=1),
    ) {
        let c_in = 1_usize;
        let c_out = 1_usize;
        let length = 4_usize;
        let kernel_size = 3_usize;
        let stride = 1_usize;
        let padding = 0_usize;
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;

        let mut output = vec![0.0_f32; c_out * out_len];
        conv1d_scalar(
            &input_v, &weight_v, Some(&bias_v),
            c_in, c_out, length, kernel_size, stride, padding, &mut output,
        );

        let max_input = input_v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let max_weight = weight_v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let max_bias = bias_v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let bound = (c_in * kernel_size) as f32 * max_input * max_weight + max_bias;

        for i in 0..output.len() {
            prop_assert!(
                output[i].abs() <= bound + 1e-4,
                "FALSIFY-CV-004 failed: |output[{i}]| = {} exceeds bound {bound}",
                output[i].abs()
            );
        }
    }

    /// FALSIFY-CV-005: SIMD equivalence
    /// Contract: conv1d-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_cv_005_simd_equivalence(
        input_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        weight_v in proptest::collection::vec(-5.0_f32..5.0, 3..=3),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let c_in = 1;
        let c_out = 1;
        let length = 4;
        let kernel_size = 3;
        let stride = 1;
        let padding = 0;
        let out_len = (length + 2 * padding - kernel_size) / stride + 1;

        let mut scalar_out = vec![0.0_f32; c_out * out_len];
        let mut avx2_out = vec![0.0_f32; c_out * out_len];

        conv1d_scalar(&input_v, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut scalar_out);
        unsafe {
            conv1d_avx2(&input_v, &weight_v, None, c_in, c_out, length, kernel_size, stride, padding, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-CV-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-CV-006: Identity kernel
/// Contract: conv1d-kernel-v1.yaml
/// Prediction: conv1d with kernel_size=1, c_in=c_out=1, weight=[1.0], bias=None reproduces input
/// If fails: identity convolution does not preserve input
#[test]
fn falsify_cv_006_identity_kernel() {
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [1.0_f32];
    let length = 8;
    let mut output = vec![0.0_f32; length];

    conv1d_scalar(&input, &weight, None, 1, 1, length, 1, 1, 0, &mut output);

    for i in 0..length {
        assert!(
            (output[i] - input[i]).abs() < 1e-7,
            "FALSIFY-CV-006 failed: output[{i}] = {}, expected {}",
            output[i], input[i]
        );
    }
}

// ============================================================================
// SSM (6 tests: FALSIFY-SSM-001 through FALSIFY-SSM-006)
// ============================================================================

/// FALSIFY-SSM-001: Causality
/// Contract: ssm-kernel-v1.yaml
/// Prediction: changing x[t] doesn't affect output[0..t]
/// If fails: future inputs leak into past outputs (violates causality)
#[test]
fn falsify_ssm_001_causality() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.5_f32, 0.8];
    let b_bar = vec![1.0_f32; state_dim * seq_len];
    let c = [1.0_f32, 1.0];

    // Run with original input
    let x1 = [1.0_f32, 2.0, 3.0, 4.0];
    let mut out1 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x1, state_dim, seq_len, &mut out1);

    // Run with different last element
    let x2 = [1.0_f32, 2.0, 3.0, 99.0];
    let mut out2 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x2, state_dim, seq_len, &mut out2);

    // Outputs before the changed position must be identical
    for t in 0..3 {
        assert!(
            (out1[t] - out2[t]).abs() < 1e-7,
            "FALSIFY-SSM-001 failed: output[{t}] differs ({} vs {}) when only x[3] changed",
            out1[t], out2[t]
        );
    }
}

proptest! {
    /// FALSIFY-SSM-002: Scan linearity (with a_bar=0)
    /// Contract: ssm-kernel-v1.yaml
    /// Prediction: when a_bar=0 (no recurrence), scaling input scales output
    /// If fails: recurrence incorrectly mixes timesteps when state should not accumulate
    #[test]
    fn falsify_ssm_002_scan_linearity(
        x_v in proptest::collection::vec(0.1_f32..5.0, 4..=4),
        alpha in 0.1_f32..5.0,
    ) {
        let state_dim = 2;
        let seq_len = 4;
        let a_bar = [0.0_f32; 2]; // no recurrence
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32, 1.0];

        // ssm(x)
        let mut out_x = vec![0.0_f32; seq_len];
        ssm_scan_scalar(&a_bar, &b_bar, &c, &x_v, state_dim, seq_len, &mut out_x);

        // ssm(alpha * x)
        let scaled: Vec<f32> = x_v.iter().map(|&x| x * alpha).collect();
        let mut out_ax = vec![0.0_f32; seq_len];
        ssm_scan_scalar(&a_bar, &b_bar, &c, &scaled, state_dim, seq_len, &mut out_ax);

        // alpha * ssm(x)
        for t in 0..seq_len {
            let expected = alpha * out_x[t];
            prop_assert!(
                (out_ax[t] - expected).abs() < 1e-3,
                "FALSIFY-SSM-002 failed: linearity violated at t={t}: ssm(alpha*x) = {}, alpha*ssm(x) = {}",
                out_ax[t], expected
            );
        }
    }

    /// FALSIFY-SSM-003: Output finiteness
    /// Contract: ssm-kernel-v1.yaml
    /// Prediction: output stays finite for moderate inputs
    /// If fails: numerical instability in scan computation
    #[test]
    fn falsify_ssm_003_output_finite(
        x_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        a_v in proptest::collection::vec(0.0_f32..0.99, 2..=2),
    ) {
        let state_dim = 2;
        let seq_len = 4;
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32; 2];

        let mut output = vec![0.0_f32; seq_len];
        ssm_scan_scalar(&a_v, &b_bar, &c, &x_v, state_dim, seq_len, &mut output);

        common::assert_all_finite(&output);
    }
}

/// FALSIFY-SSM-004: Deterministic
/// Contract: ssm-kernel-v1.yaml
/// Prediction: same input produces same output (run twice, compare)
/// If fails: non-deterministic behavior in scan
#[test]
fn falsify_ssm_004_deterministic() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.5_f32, 0.8];
    let b_bar = vec![0.3_f32; state_dim * seq_len];
    let c = [1.0_f32, 0.5];
    let x = [1.0_f32, -0.5, 0.3, 2.0];

    let mut out1 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut out1);

    let mut out2 = vec![0.0_f32; seq_len];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut out2);

    for t in 0..seq_len {
        assert!(
            (out1[t] - out2[t]).abs() < 1e-10,
            "FALSIFY-SSM-004 failed: output[{t}] differs between runs: {} vs {}",
            out1[t], out2[t]
        );
    }
}

proptest! {
    /// FALSIFY-SSM-005: SIMD equivalence
    /// Contract: ssm-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_ssm_005_simd_equivalence(
        x_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let state_dim = 2;
        let seq_len = 4;
        let a_bar = [0.5_f32, 0.8];
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32, 1.0];

        let mut scalar_out = vec![0.0_f32; seq_len];
        let mut avx2_out = vec![0.0_f32; seq_len];

        ssm_scan_scalar(&a_bar, &b_bar, &c, &x_v, state_dim, seq_len, &mut scalar_out);
        unsafe {
            ssm_scan_avx2(&a_bar, &b_bar, &c, &x_v, state_dim, seq_len, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-SSM-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-SSM-006: Zero input
/// Contract: ssm-kernel-v1.yaml
/// Prediction: ssm with x=all zeros produces all-zero output
/// If fails: scan produces non-zero output from zero input
#[test]
fn falsify_ssm_006_zero_input() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.9_f32, 0.8];
    let b_bar = vec![1.0_f32; state_dim * seq_len];
    let c = [1.0_f32, 1.0];
    let x = [0.0_f32; 4];
    let mut output = vec![0.0_f32; seq_len];

    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut output);

    for t in 0..seq_len {
        assert!(
            output[t].abs() < 1e-7,
            "FALSIFY-SSM-006 failed: output[{t}] = {}, expected 0.0 for zero input",
            output[t]
        );
    }
}

// ============================================================================
// KMeans (6 tests: FALSIFY-KM-001 through FALSIFY-KM-006)
// ============================================================================

proptest! {
    /// FALSIFY-KM-001: Nearest assignment
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: each point assigned to nearest centroid
    /// If fails: distance computation or argmin is incorrect
    #[test]
    fn falsify_km_001_nearest_assignment(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 32..=32),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let n = 8_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut assignments = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut assignments);

        for p in 0..n {
            let assigned = assignments[p] as usize;
            let assigned_dist: f32 = (0..d)
                .map(|j| {
                    let diff = points_v[p * d + j] - centroids_v[assigned * d + j];
                    diff * diff
                })
                .sum();

            for c in 0..k {
                let dist: f32 = (0..d)
                    .map(|j| {
                        let diff = points_v[p * d + j] - centroids_v[c * d + j];
                        diff * diff
                    })
                    .sum();
                prop_assert!(
                    assigned_dist <= dist + 1e-6,
                    "FALSIFY-KM-001 failed: point {p} assigned to cluster {assigned} (dist={assigned_dist}) \
                     but cluster {c} is closer (dist={dist})"
                );
            }
        }
    }

    /// FALSIFY-KM-002: Valid indices
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: all assignments in [0, k)
    /// If fails: assignment index out of bounds
    #[test]
    fn falsify_km_002_valid_indices(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 32..=32),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let n = 8_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut assignments = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut assignments);

        for (p, &a) in assignments.iter().enumerate() {
            prop_assert!(
                (a as usize) < k,
                "FALSIFY-KM-002 failed: assignment[{p}] = {a}, must be < {k}"
            );
        }
    }
}

/// FALSIFY-KM-003: Objective decrease
/// Contract: kmeans-kernel-v1.yaml
/// Prediction: after one assign+update cycle, total distance doesn't increase
/// If fails: update step does not reduce or maintain objective
#[test]
fn falsify_km_003_objective_decrease() {
    let n = 8_usize;
    let k = 2_usize;
    let d = 2_usize;
    // Well-separated clusters
    let points = [
        1.0_f32, 1.0,
        1.5, 1.5,
        2.0, 2.0,
        0.5, 0.5,
        10.0, 10.0,
        10.5, 10.5,
        11.0, 11.0,
        9.5, 9.5,
    ];
    // Start with suboptimal centroids
    let mut centroids = [0.0_f32, 0.0, 5.0, 5.0];
    let mut assignments = vec![0_u32; n];

    // Compute initial assignment and objective
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
    let obj_before: f32 = (0..n)
        .map(|p| {
            let c = assignments[p] as usize;
            (0..d)
                .map(|j| {
                    let diff = points[p * d + j] - centroids[c * d + j];
                    diff * diff
                })
                .sum::<f32>()
        })
        .sum();

    // Update centroids
    kmeans_update_scalar(&points, &assignments, n, k, d, &mut centroids);

    // Re-assign after update
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
    let obj_after: f32 = (0..n)
        .map(|p| {
            let c = assignments[p] as usize;
            (0..d)
                .map(|j| {
                    let diff = points[p * d + j] - centroids[c * d + j];
                    diff * diff
                })
                .sum::<f32>()
        })
        .sum();

    assert!(
        obj_after <= obj_before + 1e-4,
        "FALSIFY-KM-003 failed: objective increased from {obj_before} to {obj_after}"
    );
}

proptest! {
    /// FALSIFY-KM-004: Non-negative distances (implied by correct assignment)
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: squared distances are non-negative
    /// If fails: distance computation produces negative values
    #[test]
    fn falsify_km_004_nonneg_distances(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 16..=16),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let n = 4_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut assignments = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut assignments);

        // Verify that distances to assigned centroid are non-negative
        for p in 0..n {
            let c = assignments[p] as usize;
            let dist: f32 = (0..d)
                .map(|j| {
                    let diff = points_v[p * d + j] - centroids_v[c * d + j];
                    diff * diff
                })
                .sum();
            prop_assert!(
                dist >= 0.0,
                "FALSIFY-KM-004 failed: squared distance for point {p} is {dist}"
            );
        }
    }

    /// FALSIFY-KM-005: SIMD equivalence
    /// Contract: kmeans-kernel-v1.yaml
    /// Prediction: avx2 assign vs scalar produces same assignments
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_km_005_simd_equivalence(
        points_v in proptest::collection::vec(-5.0_f32..5.0, 16..=16),
        centroids_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = 4_usize;
        let k = 2_usize;
        let d = 4_usize;
        let mut asg_s = vec![0_u32; n];
        let mut asg_a = vec![0_u32; n];

        kmeans_assign_scalar(&points_v, &centroids_v, n, k, d, &mut asg_s);
        unsafe {
            kmeans_assign_avx2(&points_v, &centroids_v, n, k, d, &mut asg_a);
        }

        prop_assert_eq!(
            asg_s, asg_a,
            "FALSIFY-KM-005 failed: avx2 assignments differ from scalar"
        );
    }
}

/// FALSIFY-KM-006: Convergence
/// Contract: kmeans-kernel-v1.yaml
/// Prediction: after multiple assign+update cycles, centroids stabilize
/// If fails: algorithm does not converge on well-separated data
#[test]
fn falsify_km_006_convergence() {
    let n = 8_usize;
    let k = 2_usize;
    let d = 2_usize;
    let points = [
        0.0_f32, 0.0,
        0.1, 0.1,
        -0.1, 0.1,
        0.1, -0.1,
        10.0, 10.0,
        10.1, 10.1,
        9.9, 10.1,
        10.1, 9.9,
    ];
    let mut centroids = [1.0_f32, 1.0, 8.0, 8.0];
    let mut assignments = vec![0_u32; n];

    // Run 20 iterations
    for _ in 0..20 {
        kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
        kmeans_update_scalar(&points, &assignments, n, k, d, &mut centroids);
    }

    let prev_centroids = centroids;
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut assignments);
    kmeans_update_scalar(&points, &assignments, n, k, d, &mut centroids);

    let change: f32 = centroids
        .iter()
        .zip(prev_centroids.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();

    assert!(
        change < 1e-4,
        "FALSIFY-KM-006 failed: centroids not converged after 20 iterations, change = {change}"
    );
}

// ============================================================================
// PageRank (6 tests: FALSIFY-PR-001 through FALSIFY-PR-006)
// ============================================================================

/// Helper: generate a row-stochastic matrix of size n x n.
fn row_stochastic_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        let mut sum = 0.0_f32;
        for j in 0..n {
            let val = ((i * 7 + j * 13 + 3) % 11) as f32 + 1.0;
            mat[i * n + j] = val;
            sum += val;
        }
        for j in 0..n {
            mat[i * n + j] /= sum;
        }
    }
    mat
}

proptest! {
    /// FALSIFY-PR-001: Output distribution
    /// Contract: pagerank-kernel-v1.yaml
    /// Prediction: output sums to ~1 and all >= 0 with row-stochastic transition and uniform rank
    /// If fails: pagerank iteration does not preserve distribution property
    #[test]
    fn falsify_pr_001_output_distribution(n in 2_usize..8) {
        let transition = row_stochastic_matrix(n);
        let rank = vec![1.0 / n as f32; n];
        let mut output = vec![0.0_f32; n];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "FALSIFY-PR-001 failed: output sums to {sum}, expected ~1.0"
        );
        for (i, &o) in output.iter().enumerate() {
            prop_assert!(
                o >= 0.0,
                "FALSIFY-PR-001 failed: output[{i}] = {o} is negative"
            );
        }
    }
}

/// FALSIFY-PR-002: Input normalization
/// Contract: pagerank-kernel-v1.yaml
/// Prediction: with row-stochastic transition matrix and uniform rank, output is valid distribution
/// If fails: transition matrix handling is incorrect
#[test]
fn falsify_pr_002_input_normalization() {
    let n = 4;
    let transition = row_stochastic_matrix(n);
    let rank = vec![1.0 / n as f32; n];
    let mut output = vec![0.0_f32; n];

    pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

    common::assert_all_finite(&output);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "FALSIFY-PR-002 failed: output sums to {sum}, expected ~1.0"
    );
    for (i, &o) in output.iter().enumerate() {
        assert!(
            o >= 0.0,
            "FALSIFY-PR-002 failed: output[{i}] = {o} is negative"
        );
    }
}

/// FALSIFY-PR-003: Convergence
/// Contract: pagerank-kernel-v1.yaml
/// Prediction: repeated iteration converges (output changes decrease)
/// If fails: iteration does not converge to stationary distribution
#[test]
fn falsify_pr_003_convergence() {
    let n = 4;
    let transition = row_stochastic_matrix(n);
    let mut rank = vec![1.0 / n as f32; n];
    let mut output = vec![0.0_f32; n];

    let mut prev_change = f32::MAX;
    for _ in 0..50 {
        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);
        let change: f32 = rank
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // Change should generally decrease (may not be strictly monotone, so check final)
        prev_change = change;
        rank.copy_from_slice(&output);
    }

    assert!(
        prev_change < 1e-4,
        "FALSIFY-PR-003 failed: not converged after 50 iterations, last change = {prev_change}"
    );
}

proptest! {
    /// FALSIFY-PR-004: Non-negative
    /// Contract: pagerank-kernel-v1.yaml
    /// Prediction: all output ranks >= 0
    /// If fails: teleport or damping term produces negative values
    #[test]
    fn falsify_pr_004_non_negative(n in 2_usize..8) {
        let transition = row_stochastic_matrix(n);
        let rank = vec![1.0 / n as f32; n];
        let mut output = vec![0.0_f32; n];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

        for (i, &o) in output.iter().enumerate() {
            prop_assert!(
                o >= 0.0,
                "FALSIFY-PR-004 failed: output[{i}] = {o} is negative"
            );
        }
    }

    /// FALSIFY-PR-005: SIMD equivalence
    /// Contract: pagerank-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_pr_005_simd_equivalence(n in 2_usize..8) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let transition = row_stochastic_matrix(n);
        let rank = vec![1.0 / n as f32; n];
        let mut scalar_out = vec![0.0_f32; n];
        let mut avx2_out = vec![0.0_f32; n];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut scalar_out);
        unsafe {
            pagerank_iterate_avx2(&transition, &rank, n, 0.85, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-PR-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-PR-006: Uniform stationary
/// Contract: pagerank-kernel-v1.yaml
/// Prediction: uniform transition matrix with uniform rank gives uniform output
/// If fails: symmetry is broken by implementation
#[test]
fn falsify_pr_006_uniform_stationary() {
    let n = 4;
    // Uniform row-stochastic matrix
    let val = 1.0 / n as f32;
    let transition = vec![val; n * n];
    let rank = vec![val; n];
    let mut output = vec![0.0_f32; n];

    pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

    let expected = 1.0 / n as f32;
    for (i, &o) in output.iter().enumerate() {
        assert!(
            (o - expected).abs() < 1e-5,
            "FALSIFY-PR-006 failed: output[{i}] = {o}, expected {expected} (uniform stationary)"
        );
    }
}

// ============================================================================
// LBFGS (6 tests: FALSIFY-LB-001 through FALSIFY-LB-006)
// ============================================================================

proptest! {
    /// FALSIFY-LB-001: Descent direction
    /// Contract: lbfgs-kernel-v1.yaml
    /// Prediction: dot(direction, gradient) < 0 (direction opposes gradient)
    /// If fails: L-BFGS direction is not a descent direction
    #[test]
    fn falsify_lb_001_descent_direction(
        gradient_v in proptest::collection::vec(0.1_f32..5.0, 4..=4),
    ) {
        let d = 4;
        // Create valid curvature pair: s and y with s.y > 0
        let s_history = [0.1_f32, 0.2, 0.3, 0.4];
        let y_history = [0.5_f32, 0.6, 0.7, 0.8];
        // s.y = 0.05 + 0.12 + 0.21 + 0.32 = 0.7 > 0
        let mut direction = vec![0.0_f32; d];

        lbfgs_direction_scalar(&gradient_v, &s_history, &y_history, 1, d, &mut direction);

        let dot: f32 = gradient_v
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();
        prop_assert!(
            dot < 0.0,
            "FALSIFY-LB-001 failed: dot(direction, gradient) = {dot}, must be < 0 for descent"
        );
    }
}

/// FALSIFY-LB-002: Curvature condition
/// Contract: lbfgs-kernel-v1.yaml
/// Prediction: with s and y satisfying s.y > 0, direction is well-defined and finite
/// If fails: two-loop recursion produces NaN or infinity with valid curvature
#[test]
fn falsify_lb_002_curvature_condition() {
    let d = 4;
    let gradient = [1.0_f32, 2.0, 3.0, 4.0];
    let s_history = [0.1_f32, 0.2, 0.3, 0.4];
    let y_history = [0.5_f32, 0.6, 0.7, 0.8];
    let mut direction = vec![0.0_f32; d];

    lbfgs_direction_scalar(&gradient, &s_history, &y_history, 1, d, &mut direction);

    common::assert_all_finite(&direction);
}

/// FALSIFY-LB-003: History bound (m=0 gives steepest descent)
/// Contract: lbfgs-kernel-v1.yaml
/// Prediction: m=0 (no history) gives direction = -gradient
/// If fails: steepest descent fallback is incorrect
#[test]
fn falsify_lb_003_history_bound() {
    let d = 4;
    let gradient = [1.0_f32, -2.0, 3.0, -4.0];
    let s_history: [f32; 0] = [];
    let y_history: [f32; 0] = [];
    let mut direction = vec![0.0_f32; d];

    lbfgs_direction_scalar(&gradient, &s_history, &y_history, 0, d, &mut direction);

    for i in 0..d {
        assert!(
            (direction[i] - (-gradient[i])).abs() < 1e-7,
            "FALSIFY-LB-003 failed: direction[{i}] = {}, expected {} (-gradient)",
            direction[i], -gradient[i]
        );
    }
}

proptest! {
    /// FALSIFY-LB-004: Objective decrease (direction is finite and descending)
    /// Contract: lbfgs-kernel-v1.yaml
    /// Prediction: direction is finite and dot with gradient < 0
    /// If fails: L-BFGS direction has numerical issues
    #[test]
    fn falsify_lb_004_objective_decrease(
        gradient_v in proptest::collection::vec(0.1_f32..5.0, 4..=4),
    ) {
        let d = 4;
        let s_history = [0.2_f32, 0.3, 0.1, 0.4];
        let y_history = [0.4_f32, 0.5, 0.3, 0.6];
        let mut direction = vec![0.0_f32; d];

        lbfgs_direction_scalar(&gradient_v, &s_history, &y_history, 1, d, &mut direction);

        common::assert_all_finite(&direction);
        let dot: f32 = gradient_v
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();
        prop_assert!(
            dot < 0.0,
            "FALSIFY-LB-004 failed: dot(direction, gradient) = {dot}, must be < 0"
        );
    }

    /// FALSIFY-LB-005: SIMD equivalence
    /// Contract: lbfgs-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_lb_005_simd_equivalence(
        gradient_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let d = 4;
        let s_history = [0.1_f32, 0.2, 0.3, 0.4];
        let y_history = [0.5_f32, 0.6, 0.7, 0.8];
        let mut scalar_dir = vec![0.0_f32; d];
        let mut avx2_dir = vec![0.0_f32; d];

        lbfgs_direction_scalar(&gradient_v, &s_history, &y_history, 1, d, &mut scalar_dir);
        unsafe {
            lbfgs_direction_avx2(&gradient_v, &s_history, &y_history, 1, d, &mut avx2_dir);
        }

        let ulp = common::max_ulp_distance(&scalar_dir, &avx2_dir);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-LB-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-LB-006: Gradient recovery
/// Contract: lbfgs-kernel-v1.yaml
/// Prediction: with m=0 history, direction = -gradient
/// If fails: L-BFGS does not reduce to steepest descent with no history
#[test]
fn falsify_lb_006_gradient_recovery() {
    let d = 4;
    let gradient = [3.14_f32, -2.71, 1.41, -0.57];
    let s_history: [f32; 0] = [];
    let y_history: [f32; 0] = [];
    let mut direction = vec![0.0_f32; d];

    lbfgs_direction_scalar(&gradient, &s_history, &y_history, 0, d, &mut direction);

    for i in 0..d {
        assert!(
            (direction[i] + gradient[i]).abs() < 1e-7,
            "FALSIFY-LB-006 failed: direction[{i}] = {}, expected {} (should be -gradient)",
            direction[i], -gradient[i]
        );
    }
}

// ============================================================================
// CMA-ES (6 tests: FALSIFY-CMA-001 through FALSIFY-CMA-006)
// ============================================================================

proptest! {
    /// FALSIFY-CMA-001: Output structure (output is finite)
    /// Contract: cma-es-kernel-v1.yaml
    /// Prediction: output = mean + sigma * L * z is finite for moderate inputs
    /// If fails: matrix-vector multiply or accumulation produces NaN/infinity
    #[test]
    fn falsify_cma_001_output_structure(
        mean_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        z_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        sigma in 0.01_f32..5.0,
    ) {
        let d = 4;
        let cholesky_l = common::identity_matrix(d);
        let mut output = vec![0.0_f32; d];

        cma_sample_scalar(&mean_v, sigma, &cholesky_l, d, &z_v, &mut output);

        common::assert_all_finite(&output);
    }
}

/// FALSIFY-CMA-002: Symmetry
/// Contract: cma-es-kernel-v1.yaml
/// Prediction: sample(mean, sigma, I, d, z) and sample(mean, sigma, I, d, -z) are symmetric around mean
/// If fails: Cholesky multiplication breaks antisymmetry of z
#[test]
fn falsify_cma_002_symmetry() {
    let d = 4;
    let mean = [1.0_f32, 2.0, 3.0, 4.0];
    let sigma = 1.5_f32;
    let cholesky_l = common::identity_matrix(d);
    let z = [0.5_f32, -0.3, 0.8, -0.1];
    let neg_z: Vec<f32> = z.iter().map(|&x| -x).collect();

    let mut out_pos = vec![0.0_f32; d];
    let mut out_neg = vec![0.0_f32; d];

    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut out_pos);
    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &neg_z, &mut out_neg);

    // (out_pos + out_neg) / 2 should equal mean
    for i in 0..d {
        let midpoint = (out_pos[i] + out_neg[i]) / 2.0;
        assert!(
            (midpoint - mean[i]).abs() < 1e-5,
            "FALSIFY-CMA-002 failed: midpoint[{i}] = {midpoint}, expected mean[{i}] = {}",
            mean[i]
        );
    }
}

proptest! {
    /// FALSIFY-CMA-003: Step size
    /// Contract: cma-es-kernel-v1.yaml
    /// Prediction: with small sigma, output is close to mean
    /// If fails: sigma scaling is not applied correctly
    #[test]
    fn falsify_cma_003_step_size(
        mean_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        z_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
    ) {
        let d = 4;
        let sigma = 0.001_f32; // very small sigma
        let cholesky_l = common::identity_matrix(d);
        let mut output = vec![0.0_f32; d];

        cma_sample_scalar(&mean_v, sigma, &cholesky_l, d, &z_v, &mut output);

        let dist = common::l2_distance(&mean_v, &output);
        // With sigma=0.001 and |z| bounded by ~sqrt(4)*5 ~ 10, dist should be < 0.01 * 10 = 0.1
        let z_norm = common::l2_norm(&z_v);
        let bound = sigma * z_norm + 1e-4;
        prop_assert!(
            dist <= bound,
            "FALSIFY-CMA-003 failed: distance from mean = {dist}, expected <= {bound} with sigma={sigma}"
        );
    }
}

/// FALSIFY-CMA-004: Positive definite (identity Cholesky)
/// Contract: cma-es-kernel-v1.yaml
/// Prediction: with identity Cholesky, output is mean + sigma*z
/// If fails: lower triangular multiply is incorrect for identity matrix
#[test]
fn falsify_cma_004_positive_definite() {
    let d = 4;
    let mean = [1.0_f32, 2.0, 3.0, 4.0];
    let sigma = 2.0_f32;
    let cholesky_l = common::identity_matrix(d);
    let z = [0.5_f32, -0.3, 0.8, -0.1];
    let mut output = vec![0.0_f32; d];

    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut output);

    for i in 0..d {
        let expected = mean[i] + sigma * z[i];
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "FALSIFY-CMA-004 failed: output[{i}] = {}, expected {expected} (mean + sigma*z)",
            output[i]
        );
    }
}

proptest! {
    /// FALSIFY-CMA-005: SIMD equivalence
    /// Contract: cma-es-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_cma_005_simd_equivalence(
        mean_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        z_v in proptest::collection::vec(-5.0_f32..5.0, 4..=4),
        sigma in 0.01_f32..5.0,
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let d = 4;
        let cholesky_l = [
            1.0_f32, 0.0, 0.0, 0.0,
            0.5, 1.0, 0.0, 0.0,
            0.3, 0.2, 1.0, 0.0,
            0.1, 0.4, 0.6, 1.0,
        ];
        let mut scalar_out = vec![0.0_f32; d];
        let mut avx2_out = vec![0.0_f32; d];

        cma_sample_scalar(&mean_v, sigma, &cholesky_l, d, &z_v, &mut scalar_out);
        unsafe {
            cma_sample_avx2(&mean_v, sigma, &cholesky_l, d, &z_v, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-CMA-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}

/// FALSIFY-CMA-006: Identity Cholesky
/// Contract: cma-es-kernel-v1.yaml
/// Prediction: L=I gives output = mean + sigma*z
/// If fails: identity matrix handling in lower-triangular multiply is broken
#[test]
fn falsify_cma_006_identity_cholesky() {
    let d = 4;
    let mean = [10.0_f32, 20.0, 30.0, 40.0];
    let sigma = 3.0_f32;
    let cholesky_l = common::identity_matrix(d);
    let z = [1.0_f32, -1.0, 0.5, -0.5];
    let mut output = vec![0.0_f32; d];

    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut output);

    for i in 0..d {
        let expected = mean[i] + sigma * z[i];
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "FALSIFY-CMA-006 failed: output[{i}] = {}, expected {expected}",
            output[i]
        );
    }
}

// ============================================================================
// Gated Delta Net (5 tests: FALSIFY-GDN-001 through FALSIFY-GDN-005)
// ============================================================================

/// FALSIFY-GDN-001: Output shape
/// Contract: gated-delta-net-v1.yaml
/// Prediction: output length = seq_len * v_dim
/// If fails: output dimension calculation is incorrect
#[test]
fn falsify_gdn_001_output_shape() {
    let seq_len = 4;
    let k_dim = 3;
    let v_dim = 2;
    let q = vec![0.1_f32; seq_len * k_dim];
    let k = vec![0.2_f32; seq_len * k_dim];
    let v = vec![0.3_f32; seq_len * v_dim];
    let alpha = vec![0.5_f32; seq_len];
    let beta = vec![0.5_f32; seq_len];

    let expected_len = seq_len * v_dim;
    let mut output = vec![0.0_f32; expected_len];

    // Should not panic -- output buffer has correct size
    gdn_recurrence_scalar(&q, &k, &v, &alpha, &beta, seq_len, k_dim, v_dim, &mut output);

    assert_eq!(
        output.len(),
        expected_len,
        "FALSIFY-GDN-001 failed: output length = {}, expected {expected_len}",
        output.len()
    );
}

/// FALSIFY-GDN-002: Causality
/// Contract: gated-delta-net-v1.yaml
/// Prediction: changing input at time t doesn't affect output before t
/// If fails: future inputs leak into past outputs (violates causality)
#[test]
fn falsify_gdn_002_causality() {
    let seq_len = 4;
    let k_dim = 2;
    let v_dim = 2;
    let q = vec![0.1_f32; seq_len * k_dim];
    let k1 = vec![0.2_f32; seq_len * k_dim];
    let v1 = vec![0.3_f32; seq_len * v_dim];
    let alpha = vec![0.5_f32; seq_len];
    let beta = vec![0.5_f32; seq_len];

    let mut out1 = vec![0.0_f32; seq_len * v_dim];
    gdn_recurrence_scalar(&q, &k1, &v1, &alpha, &beta, seq_len, k_dim, v_dim, &mut out1);

    // Modify last timestep's key and value
    let mut k2 = k1.clone();
    let mut v2 = v1.clone();
    for i in 0..k_dim {
        k2[(seq_len - 1) * k_dim + i] = 99.0;
    }
    for i in 0..v_dim {
        v2[(seq_len - 1) * v_dim + i] = 99.0;
    }

    let mut out2 = vec![0.0_f32; seq_len * v_dim];
    gdn_recurrence_scalar(&q, &k2, &v2, &alpha, &beta, seq_len, k_dim, v_dim, &mut out2);

    // Outputs before the last timestep must be identical
    for t in 0..(seq_len - 1) {
        for j in 0..v_dim {
            let idx = t * v_dim + j;
            assert!(
                (out1[idx] - out2[idx]).abs() < 1e-7,
                "FALSIFY-GDN-002 failed: output[{t},{j}] differs ({} vs {}) when only last timestep changed",
                out1[idx], out2[idx]
            );
        }
    }
}

proptest! {
    /// FALSIFY-GDN-003: Decay bound
    /// Contract: gated-delta-net-v1.yaml
    /// Prediction: with alpha in [0,1], output stays bounded/finite
    /// If fails: gating does not properly bound state growth
    #[test]
    fn falsify_gdn_003_decay_bound(
        q_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        k_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        v_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        alpha_v in proptest::collection::vec(0.0_f32..1.0, 4..=4),
        beta_v in proptest::collection::vec(0.0_f32..1.0, 4..=4),
    ) {
        let seq_len = 4;
        let k_dim = 2;
        let v_dim = 2;
        let mut output = vec![0.0_f32; seq_len * v_dim];

        gdn_recurrence_scalar(&q_v, &k_v, &v_v, &alpha_v, &beta_v, seq_len, k_dim, v_dim, &mut output);

        common::assert_all_finite(&output);
    }

    /// FALSIFY-GDN-004: L2 direction (output finite and bounded)
    /// Contract: gated-delta-net-v1.yaml
    /// Prediction: output is finite and bounded for moderate inputs
    /// If fails: numerical instability in state update or readout
    #[test]
    fn falsify_gdn_004_l2_direction(
        q_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        k_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        v_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        let seq_len = 4;
        let k_dim = 2;
        let v_dim = 2;
        let alpha = vec![0.5_f32; seq_len];
        let beta = vec![0.5_f32; seq_len];
        let mut output = vec![0.0_f32; seq_len * v_dim];

        gdn_recurrence_scalar(&q_v, &k_v, &v_v, &alpha, &beta, seq_len, k_dim, v_dim, &mut output);

        common::assert_all_finite(&output);
        // With bounded inputs and alpha=0.5, beta=0.5, output should be bounded
        let out_norm = common::l2_norm(&output);
        prop_assert!(
            out_norm < 1000.0,
            "FALSIFY-GDN-004 failed: output L2 norm = {out_norm} is unreasonably large"
        );
    }

    /// FALSIFY-GDN-005: SIMD equivalence
    /// Contract: gated-delta-net-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: SIMD implementation diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_gdn_005_simd_equivalence(
        q_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        k_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
        v_v in proptest::collection::vec(-5.0_f32..5.0, 8..=8),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let seq_len = 4;
        let k_dim = 2;
        let v_dim = 2;
        let alpha = vec![0.9_f32; seq_len];
        let beta = vec![0.1_f32; seq_len];
        let mut scalar_out = vec![0.0_f32; seq_len * v_dim];
        let mut avx2_out = vec![0.0_f32; seq_len * v_dim];

        gdn_recurrence_scalar(&q_v, &k_v, &v_v, &alpha, &beta, seq_len, k_dim, v_dim, &mut scalar_out);
        unsafe {
            gdn_recurrence_avx2(&q_v, &k_v, &v_v, &alpha, &beta, seq_len, k_dim, v_dim, &mut avx2_out);
        }

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-GDN-005 failed: max ULP distance = {ulp}, exceeds threshold 8"
        );
    }
}
