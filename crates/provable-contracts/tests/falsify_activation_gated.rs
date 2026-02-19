//! Phase 5 falsification tests for activation and gated kernels.
//!
//! Covers: Activation (ReLU, GELU, SiLU), SiLU Standalone, SwiGLU,
//! Cross-Entropy, and RoPE kernels.
//!
//! 28 tests total: FALSIFY-ACT-001..006, FALSIFY-SI-001..006,
//! FALSIFY-SG-001..006, FALSIFY-CE-001..006, FALSIFY-RP-001..004.

mod common;

use proptest::prelude::*;

use provable_contracts::kernels::activation::{
    gelu_scalar, relu_scalar, silu_scalar,
};
#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::activation::relu_avx2;

use provable_contracts::kernels::silu_standalone::{
    sigmoid_scalar, silu_standalone_scalar,
};
#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::silu_standalone::silu_standalone_avx2;

use provable_contracts::kernels::swiglu::swiglu_scalar;
#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::swiglu::swiglu_avx2;

use provable_contracts::kernels::cross_entropy::{
    cross_entropy_scalar, log_softmax_scalar,
};
#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::cross_entropy::cross_entropy_avx2;

use provable_contracts::kernels::rope::rope_scalar;
#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::rope::rope_avx2;

// ============================================================================
// Activation (FALSIFY-ACT-001 through FALSIFY-ACT-006)
// ============================================================================

proptest! {
    /// FALSIFY-ACT-001: ReLU non-negative
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: relu(x)_i >= 0 for all i
    /// If fails: ReLU produces negative output, violating its definition
    #[test]
    fn falsify_act_001_relu_non_negative(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y >= 0.0,
                "FALSIFY-ACT-001 failed: relu output[{i}] = {y} < 0"
            );
        }
    }
}

/// FALSIFY-ACT-002: GELU at zero
/// Contract: activation-kernel-v1.yaml
/// Prediction: gelu([0.0]) is approximately 0.0
/// If fails: GELU does not pass through origin
#[test]
fn falsify_act_002_gelu_at_zero() {
    let input = [0.0f32];
    let mut output = [0.0f32; 1];
    gelu_scalar(&input, &mut output);
    assert!(
        output[0].abs() < 1e-7,
        "FALSIFY-ACT-002 failed: gelu(0.0) = {}, expected ~0.0",
        output[0]
    );
}

proptest! {
    /// FALSIFY-ACT-003: GELU approx error
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: |gelu(x)_i - x_i * 0.5 * (1 + tanh(sqrt(2/pi) * (x_i + 0.044715 * x_i^3)))| < 0.005
    /// If fails: GELU implementation deviates from the standard tanh approximation formula
    #[test]
    fn falsify_act_003_gelu_approx_error(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        gelu_scalar(&input, &mut output);
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        for (&x, &y) in input.iter().zip(output.iter()) {
            let expected = x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh());
            let err = (y - expected).abs();
            prop_assert!(
                err < 0.005,
                "FALSIFY-ACT-003 failed: gelu({x}) = {y}, expected {expected}, error = {err}"
            );
        }
    }
}

/// FALSIFY-ACT-004: SiLU zero preservation
/// Contract: activation-kernel-v1.yaml
/// Prediction: silu([0.0]) = [0.0]
/// If fails: SiLU does not preserve zero
#[test]
fn falsify_act_004_silu_zero_preservation() {
    let input = [0.0f32];
    let mut output = [0.0f32; 1];
    silu_scalar(&input, &mut output);
    assert!(
        output[0].abs() < 1e-7,
        "FALSIFY-ACT-004 failed: silu(0.0) = {}, expected 0.0",
        output[0]
    );
}

proptest! {
    /// FALSIFY-ACT-005: ReLU monotonicity
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: for sorted input, relu output is non-decreasing
    /// If fails: ReLU violates monotonicity
    #[test]
    fn falsify_act_005_relu_monotonicity(
        mut input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        input.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut output = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut output);
        for i in 1..output.len() {
            prop_assert!(
                output[i] >= output[i - 1],
                "FALSIFY-ACT-005 failed: relu not monotone: output[{}] = {} < output[{}] = {}",
                i, output[i], i - 1, output[i - 1]
            );
        }
    }

    /// FALSIFY-ACT-006: SIMD equivalence
    /// Contract: activation-kernel-v1.yaml
    /// Prediction: avx2 relu vs scalar relu within 4 ULP
    /// If fails: AVX2 ReLU diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_act_006_simd_equivalence(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut scalar_out);
        unsafe { relu_avx2(&input, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-ACT-006 failed: relu avx2 vs scalar ULP distance = {ulp} > 4"
        );
    }
}

// ============================================================================
// SiLU Standalone (FALSIFY-SI-001 through FALSIFY-SI-006)
// ============================================================================

/// FALSIFY-SI-001: Zero preservation
/// Contract: silu-kernel-v1.yaml
/// Prediction: silu_standalone([0.0]) = [0.0]
/// If fails: standalone SiLU does not preserve zero
#[test]
fn falsify_si_001_zero_preservation() {
    let input = [0.0f32];
    let mut output = [0.0f32; 1];
    silu_standalone_scalar(&input, &mut output);
    assert!(
        output[0].abs() < 1e-7,
        "FALSIFY-SI-001 failed: silu_standalone(0.0) = {}, expected 0.0",
        output[0]
    );
}

proptest! {
    /// FALSIFY-SI-002: Lower bound
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: silu(x)_i >= -0.279 (known minimum of silu approx -0.278 at x approx -1.278)
    /// If fails: SiLU violates known lower bound
    #[test]
    fn falsify_si_002_lower_bound(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y >= -0.279,
                "FALSIFY-SI-002 failed: silu_standalone output[{i}] = {y} < -0.279"
            );
        }
    }

    /// FALSIFY-SI-003: Monotonicity for large x
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: for x > 1, silu is increasing (sorted positive values give non-decreasing output)
    /// If fails: SiLU is not monotone for positive inputs
    #[test]
    fn falsify_si_003_monotonicity_large_x(
        mut input in proptest::collection::vec(1.0f32..10.0, 2..=32),
    ) {
        input.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for i in 1..output.len() {
            prop_assert!(
                output[i] >= output[i - 1],
                "FALSIFY-SI-003 failed: silu not monotone for positive x: output[{}] = {} < output[{}] = {}",
                i, output[i], i - 1, output[i - 1]
            );
        }
    }

    /// FALSIFY-SI-004: Asymptotic
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: for large positive x (> 10), silu(x) approx x (within |x|*0.01)
    /// If fails: SiLU does not converge to identity for large positive x
    #[test]
    fn falsify_si_004_asymptotic(
        input in proptest::collection::vec(10.0f32..100.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut output);
        for (&x, &y) in input.iter().zip(output.iter()) {
            let ratio = y / x;
            prop_assert!(
                (ratio - 1.0).abs() < 0.01,
                "FALSIFY-SI-004 failed: silu({x}) / {x} = {ratio}, expected ~1.0"
            );
        }
    }

    /// FALSIFY-SI-005: SIMD equivalence
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: avx2 silu_standalone vs scalar within 4 ULP
    /// If fails: AVX2 SiLU standalone diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_si_005_simd_equivalence(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];
        silu_standalone_scalar(&input, &mut scalar_out);
        unsafe { silu_standalone_avx2(&input, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-SI-005 failed: silu_standalone avx2 vs scalar ULP distance = {ulp} > 4"
        );
    }

    /// FALSIFY-SI-006: Sigmoid range
    /// Contract: silu-kernel-v1.yaml
    /// Prediction: sigmoid output in [0, 1]
    /// If fails: sigmoid produces values outside the unit interval
    #[test]
    fn falsify_si_006_sigmoid_range(
        input in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; input.len()];
        sigmoid_scalar(&input, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&y),
                "FALSIFY-SI-006 failed: sigmoid output[{i}] = {y} not in [0, 1]"
            );
        }
    }
}

// ============================================================================
// SwiGLU (FALSIFY-SG-001 through FALSIFY-SG-006)
// ============================================================================

/// FALSIFY-SG-001: Zero gate
/// Contract: swiglu-kernel-v1.yaml
/// Prediction: swiglu(gate=[0,...], value=anything) = [0,...] since silu(0)=0
/// If fails: SwiGLU with zero gate does not produce zero output
#[test]
fn falsify_sg_001_zero_gate() {
    let gate = [0.0f32; 8];
    let value = [1.0f32, -2.5, 3.14, 100.0, -0.001, 42.0, 0.0, -99.9];
    let mut output = [0.0f32; 8];
    swiglu_scalar(&gate, &value, &mut output);
    for (i, &y) in output.iter().enumerate() {
        assert!(
            y == 0.0,
            "FALSIFY-SG-001 failed: swiglu(0, value)[{i}] = {y}, expected 0.0"
        );
    }
}

proptest! {
    /// FALSIFY-SG-002: Fused equals unfused
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: swiglu(g, v) approx silu(g) * v element-wise (within tolerance)
    /// If fails: fused SwiGLU disagrees with manual silu(g) * v computation
    #[test]
    fn falsify_sg_002_fused_equals_unfused(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut output = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut output);
        for ((&g, &v), &y) in gate.iter().zip(value.iter()).zip(output.iter()) {
            let sigmoid_g = 1.0 / (1.0 + (-g).exp());
            let expected = g * sigmoid_g * v;
            let err = (y - expected).abs();
            prop_assert!(
                err < 1e-4,
                "FALSIFY-SG-002 failed: swiglu({g}, {v}) = {y}, manual = {expected}, error = {err}"
            );
        }
    }

    /// FALSIFY-SG-003: Output bound
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: swiglu output is finite for moderate inputs
    /// If fails: SwiGLU produces non-finite values for moderate inputs
    #[test]
    fn falsify_sg_003_output_bound(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut output = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut output);
        common::assert_all_finite(&output);
    }

    /// FALSIFY-SG-004: SIMD equivalence
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 8 ULP
    /// If fails: AVX2 SwiGLU diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_sg_004_simd_equivalence(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut scalar_out = vec![0.0f32; n];
        let mut avx2_out = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut scalar_out);
        unsafe { swiglu_avx2(gate, value, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-SG-004 failed: swiglu avx2 vs scalar ULP distance = {ulp} > 8"
        );
    }
}

/// FALSIFY-SG-005: Symmetry check
/// Contract: swiglu-kernel-v1.yaml
/// Prediction: swiglu with value=[1,1,...] equals silu of gate
/// If fails: SwiGLU with unit value does not reduce to SiLU
#[test]
fn falsify_sg_005_symmetry_check() {
    let gate = [-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
    let value = [1.0f32; 8];
    let mut swiglu_out = [0.0f32; 8];
    let mut silu_out = [0.0f32; 8];
    swiglu_scalar(&gate, &value, &mut swiglu_out);
    silu_scalar(&gate, &mut silu_out);
    for (i, (&sg, &si)) in swiglu_out.iter().zip(silu_out.iter()).enumerate() {
        assert!(
            (sg - si).abs() < 1e-6,
            "FALSIFY-SG-005 failed: swiglu(gate, 1)[{i}] = {sg} != silu(gate)[{i}] = {si}"
        );
    }
}

proptest! {
    /// FALSIFY-SG-006: Finiteness
    /// Contract: swiglu-kernel-v1.yaml
    /// Prediction: output is finite for moderate inputs
    /// If fails: SwiGLU produces NaN or infinity for moderate inputs
    #[test]
    fn falsify_sg_006_finiteness(
        gate in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        value in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n = gate.len().min(value.len());
        let gate = &gate[..n];
        let value = &value[..n];
        let mut output = vec![0.0f32; n];
        swiglu_scalar(gate, value, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "FALSIFY-SG-006 failed: swiglu output[{i}] = {y} is not finite"
            );
        }
    }
}

// ============================================================================
// Cross-Entropy (FALSIFY-CE-001 through FALSIFY-CE-006)
// ============================================================================

proptest! {
    /// FALSIFY-CE-001: Non-negative
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: cross_entropy(one_hot, logits) >= 0
    /// If fails: cross-entropy loss is negative, violating information-theoretic lower bound
    #[test]
    fn falsify_ce_001_non_negative(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        class_idx_raw in 0usize..32,
    ) {
        let n = logits.len();
        let class_idx = class_idx_raw % n;
        let mut targets = vec![0.0f32; n];
        targets[class_idx] = 1.0;
        let loss = cross_entropy_scalar(&targets, &logits);
        prop_assert!(
            loss >= -1e-6,
            "FALSIFY-CE-001 failed: cross_entropy = {loss} < 0"
        );
    }

    /// FALSIFY-CE-002: Log-softmax <= 0
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: all elements of log_softmax(x) <= 0
    /// If fails: log-softmax produces positive values, violating log(probability) <= 0
    #[test]
    fn falsify_ce_002_log_softmax_nonpositive(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; logits.len()];
        log_softmax_scalar(&logits, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y <= 1e-7,
                "FALSIFY-CE-002 failed: log_softmax output[{i}] = {y} > 0"
            );
        }
    }

    /// FALSIFY-CE-003: Finiteness
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: log_softmax output is all finite for moderate inputs [-100, 100]
    /// If fails: log-softmax produces NaN or infinity despite numerical stability measures
    #[test]
    fn falsify_ce_003_finiteness(
        logits in proptest::collection::vec(-100.0f32..100.0, 2..=32),
    ) {
        let mut output = vec![0.0f32; logits.len()];
        log_softmax_scalar(&logits, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "FALSIFY-CE-003 failed: log_softmax output[{i}] = {y} is not finite"
            );
        }
    }

    /// FALSIFY-CE-004: NLL equivalence
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: cross_entropy with one-hot target selecting class k equals -log_softmax(x)[k]
    /// If fails: cross-entropy does not reduce to negative log-likelihood for one-hot targets
    #[test]
    fn falsify_ce_004_nll_equivalence(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        class_idx_raw in 0usize..32,
    ) {
        let n = logits.len();
        let k = class_idx_raw % n;
        let mut targets = vec![0.0f32; n];
        targets[k] = 1.0;
        let loss = cross_entropy_scalar(&targets, &logits);
        let mut log_sm = vec![0.0f32; n];
        log_softmax_scalar(&logits, &mut log_sm);
        let expected = -log_sm[k];
        let err = (loss - expected).abs();
        prop_assert!(
            err < 1e-5,
            "FALSIFY-CE-004 failed: cross_entropy = {loss}, -log_softmax[{k}] = {expected}, error = {err}"
        );
    }

    /// FALSIFY-CE-005: SIMD equivalence
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: avx2 cross_entropy vs scalar within relative tolerance 1e-5
    /// If fails: AVX2 cross-entropy diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_ce_005_simd_equivalence(
        logits in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        class_idx_raw in 0usize..32,
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let n = logits.len();
        let k = class_idx_raw % n;
        let mut targets = vec![0.0f32; n];
        targets[k] = 1.0;
        let scalar_loss = cross_entropy_scalar(&targets, &logits);
        let avx2_loss = unsafe { cross_entropy_avx2(&targets, &logits) };
        let err = (scalar_loss - avx2_loss).abs();
        let denom = scalar_loss.abs().max(1e-8);
        let rel_err = err / denom;
        prop_assert!(
            rel_err < 1e-5,
            "FALSIFY-CE-005 failed: scalar = {scalar_loss}, avx2 = {avx2_loss}, rel_err = {rel_err}"
        );
    }

    /// FALSIFY-CE-006: Log-softmax normalization
    /// Contract: cross-entropy-kernel-v1.yaml
    /// Prediction: exp(log_softmax(x)) sums to approximately 1.0
    /// If fails: softmax probabilities derived from log-softmax do not sum to 1
    #[test]
    fn falsify_ce_006_log_softmax_normalization(
        logits in proptest::collection::vec(-50.0f32..50.0, 2..=32),
    ) {
        let mut log_sm = vec![0.0f32; logits.len()];
        log_softmax_scalar(&logits, &mut log_sm);
        let sum: f32 = log_sm.iter().map(|&y| y.exp()).sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "FALSIFY-CE-006 failed: exp(log_softmax) sums to {sum}, expected ~1.0"
        );
    }
}

// ============================================================================
// RoPE (FALSIFY-RP-001 through FALSIFY-RP-004)
// ============================================================================

proptest! {
    /// FALSIFY-RP-001: Norm preservation
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: ||rope(x)|| approx ||x|| (rotation preserves norm)
    /// If fails: RoPE rotation does not preserve vector norm
    #[test]
    fn falsify_rp_001_norm_preservation(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        position in 0u32..1000,
    ) {
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(0.0)).take(dim).collect();
        let mut output = vec![0.0f32; dim];
        rope_scalar(&x, position, dim, 10000.0, &mut output);
        let input_norm = common::l2_norm(&x);
        let output_norm = common::l2_norm(&output);
        prop_assert!(
            (input_norm - output_norm).abs() < 1e-3,
            "FALSIFY-RP-001 failed: input norm = {input_norm}, output norm = {output_norm}"
        );
    }

    /// FALSIFY-RP-002: Relative position
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: rope at position p1 != rope at position p2 for same non-zero input
    /// If fails: RoPE fails to encode different positions differently
    #[test]
    fn falsify_rp_002_relative_position(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(1.0f32..10.0, 2..=32),
        p1 in 1u32..500,
        p2 in 500u32..1000,
    ) {
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(1.0)).take(dim).collect();
        let mut out1 = vec![0.0f32; dim];
        let mut out2 = vec![0.0f32; dim];
        rope_scalar(&x, p1, dim, 10000.0, &mut out1);
        rope_scalar(&x, p2, dim, 10000.0, &mut out2);
        let differs = out1.iter().zip(out2.iter()).any(|(&a, &b)| (a - b).abs() > 1e-7);
        prop_assert!(
            differs,
            "FALSIFY-RP-002 failed: rope at position {p1} and {p2} produced identical output"
        );
    }

    /// FALSIFY-RP-003: Output bounds
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: all outputs finite and bounded
    /// If fails: RoPE produces non-finite output values
    #[test]
    fn falsify_rp_003_output_bounds(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        position in 0u32..10000,
    ) {
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(0.0)).take(dim).collect();
        let mut output = vec![0.0f32; dim];
        rope_scalar(&x, position, dim, 10000.0, &mut output);
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "FALSIFY-RP-003 failed: rope output[{i}] = {y} is not finite"
            );
        }
    }

    /// FALSIFY-RP-004: SIMD equivalence
    /// Contract: rope-kernel-v1.yaml
    /// Prediction: avx2 vs scalar within 4 ULP
    /// If fails: AVX2 RoPE diverges from scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_rp_004_simd_equivalence(
        half_len in 1usize..=16,
        raw_values in proptest::collection::vec(-10.0f32..10.0, 2..=32),
        position in 0u32..1000,
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let dim = half_len * 2;
        let x: Vec<f32> = raw_values.into_iter().take(dim).chain(std::iter::repeat(0.0)).take(dim).collect();
        let mut scalar_out = vec![0.0f32; dim];
        let mut avx2_out = vec![0.0f32; dim];
        rope_scalar(&x, position, dim, 10000.0, &mut scalar_out);
        unsafe { rope_avx2(&x, position, dim, 10000.0, &mut avx2_out) };
        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 4,
            "FALSIFY-RP-004 failed: rope avx2 vs scalar ULP distance = {ulp} > 4"
        );
    }
}
