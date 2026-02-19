//! Activation kernels: `ReLU`, `GELU`, `SiLU`.
//!
//! Matches `activation-kernel-v1.yaml`.
//!
//! Each function provides one of three backends:
//! - `fn {name}_scalar(...)` — Pure Rust scalar reference (ground truth)
//! - `unsafe fn {name}_avx2(...)` — AVX2 SIMD implementation
//! - `fn {name}_ptx() -> &'static str` — PTX assembly source string

use std::f32::consts::PI;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementations
// ────────────────────────────────────────────────────────────────────────────

/// `ReLU`: max(0, x)
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn relu_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = x.max(0.0);
    }
}

/// `GELU`: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn gelu_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let sqrt_2_over_pi = (2.0f32 / PI).sqrt();
    for (x, y) in input.iter().zip(output.iter_mut()) {
        let inner = sqrt_2_over_pi * (x + 0.044_715 * x * x * x);
        *y = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// `SiLU` (Swish): x / (1 + exp(-x))
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn silu_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = x / (1.0 + (-x).exp());
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementations
// ────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

/// AVX2 `ReLU`: `_mm256_max_ps(x, zero)` with scalar tail.
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let n = input.len();
    // SAFETY: caller guarantees AVX2 is available; target_feature gate enforces it.
    unsafe {
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            let r = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), r);
            i += 8;
        }
        // Scalar tail for remaining elements
        for j in i..n {
            output[j] = input[j].max(0.0);
        }
    }
}

/// AVX2 `GELU` — delegates to scalar (no hardware `tanh` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gelu_avx2(input: &[f32], output: &mut [f32]) {
    gelu_scalar(input, output);
}

/// AVX2 `SiLU` — delegates to scalar (no hardware `exp` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn silu_avx2(input: &[f32], output: &mut [f32]) {
    silu_scalar(input, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementations
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for the `ReLU` kernel (elementwise, 1 thread per element).
pub fn relu_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry relu_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %zero, %y;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];
    mov.f32 %zero, 0f00000000;
    max.f32 %y, %x, %zero;
    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

/// PTX assembly for the `GELU` kernel.
///
/// Uses the approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// where tanh(a) is computed as 1 - 2/(1 + exp(2a)) and
/// exp(a) = 2^(a / ln2) via `ex2.approx.f32`.
pub fn gelu_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry gelu_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %y, %x3, %inner, %two_inner, %scaled;
    .reg .f32 %exp_val, %denom, %tanh_val, %one_plus_tanh, %half_x;
    .reg .f32 %k_sqrt2pi, %k_coeff, %k_half, %k_one, %k_two, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];

    // Constants
    mov.f32 %k_sqrt2pi, 0f3F4C422A;   // sqrt(2/pi) ~ 0.7978845608
    mov.f32 %k_coeff, 0f3D372713;     // 0.044715
    mov.f32 %k_half, 0f3F000000;      // 0.5
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_two, 0f40000000;       // 2.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // x^3
    mul.f32 %x3, %x, %x;
    mul.f32 %x3, %x3, %x;

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fma.rn.f32 %inner, %k_coeff, %x3, %x;
    mul.f32 %inner, %k_sqrt2pi, %inner;

    // tanh(inner) = 1 - 2/(1 + exp(2*inner))
    // exp(2*inner) via ex2: exp(a) = 2^(a/ln2)
    mul.f32 %two_inner, %k_two, %inner;
    mul.f32 %scaled, %two_inner, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;
    add.f32 %denom, %k_one, %exp_val;
    rcp.approx.f32 %denom, %denom;
    // tanh = 1 - 2*rcp(1+exp(2*inner))
    mul.f32 %tanh_val, %k_two, %denom;
    sub.f32 %tanh_val, %k_one, %tanh_val;

    // 0.5 * x * (1 + tanh)
    add.f32 %one_plus_tanh, %k_one, %tanh_val;
    mul.f32 %half_x, %k_half, %x;
    mul.f32 %y, %half_x, %one_plus_tanh;

    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

/// PTX assembly for the `SiLU` (Swish) kernel.
///
/// `SiLU`(x) = x / (1 + exp(-x)), where exp(-x) = 2^(-x / ln2) via `ex2.approx.f32`.
pub fn silu_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry silu_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %y, %neg_x, %scaled, %exp_val, %denom, %rcp_denom;
    .reg .f32 %k_one, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];

    // Constants
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // exp(-x) = 2^(-x / ln2) = 2^(-x * (1/ln2))
    neg.f32 %neg_x, %x;
    mul.f32 %scaled, %neg_x, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;

    // silu = x / (1 + exp(-x))
    add.f32 %denom, %k_one, %exp_val;
    rcp.approx.f32 %rcp_denom, %denom;
    mul.f32 %y, %x, %rcp_denom;

    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ulp::assert_ulp_eq;
    use proptest::prelude::*;

    // ── ReLU known-answer tests ──────────────────────────────────────────

    #[test]
    fn test_relu_negative_to_zero() {
        let input = [-3.0f32, -1.0, -0.5, -1e-6];
        let mut output = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut output);
        for &y in &output {
            assert_eq!(y, 0.0);
        }
    }

    #[test]
    fn test_relu_positive_identity() {
        let input = [0.5f32, 1.0, 3.0, 100.0];
        let mut output = vec![0.0f32; input.len()];
        relu_scalar(&input, &mut output);
        for (x, y) in input.iter().zip(output.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_relu_zero() {
        let input = [0.0f32];
        let mut output = vec![0.0f32; 1];
        relu_scalar(&input, &mut output);
        assert_eq!(output[0], 0.0);
    }

    // ── GELU known-answer tests ──────────────────────────────────────────

    #[test]
    fn test_gelu_zero() {
        let input = [0.0f32];
        let mut output = vec![0.0f32; 1];
        gelu_scalar(&input, &mut output);
        assert!((output[0]).abs() < 1e-7, "GELU(0) should be 0, got {}", output[0]);
    }

    #[test]
    fn test_gelu_large_positive() {
        let input = [10.0f32];
        let mut output = vec![0.0f32; 1];
        gelu_scalar(&input, &mut output);
        // For large positive x, GELU(x) ~ x
        assert!(
            (output[0] - 10.0).abs() < 1e-4,
            "GELU(10) should be ~10, got {}",
            output[0]
        );
    }

    #[test]
    fn test_gelu_large_negative() {
        let input = [-10.0f32];
        let mut output = vec![0.0f32; 1];
        gelu_scalar(&input, &mut output);
        // For large negative x, GELU(x) ~ 0
        assert!(
            output[0].abs() < 1e-4,
            "GELU(-10) should be ~0, got {}",
            output[0]
        );
    }

    // ── SiLU known-answer tests ──────────────────────────────────────────

    #[test]
    fn test_silu_zero() {
        let input = [0.0f32];
        let mut output = vec![0.0f32; 1];
        silu_scalar(&input, &mut output);
        assert!((output[0]).abs() < 1e-7, "SiLU(0) should be 0, got {}", output[0]);
    }

    #[test]
    fn test_silu_positive() {
        let input = [1.0f32];
        let mut output = vec![0.0f32; 1];
        silu_scalar(&input, &mut output);
        // SiLU(1) = 1 / (1 + exp(-1)) ~ 0.7310586
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "SiLU(1) should be ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn test_silu_negative() {
        let input = [-1.0f32];
        let mut output = vec![0.0f32; 1];
        silu_scalar(&input, &mut output);
        // SiLU(-1) = -1 / (1 + exp(1)) ~ -0.2689414
        let expected = -1.0 / (1.0 + 1.0f32.exp());
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "SiLU(-1) should be ~{expected}, got {}",
            output[0]
        );
    }

    // ── Property-based tests ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_relu_nonnegative(x in proptest::num::f32::NORMAL) {
            let input = [x];
            let mut output = [0.0f32];
            relu_scalar(&input, &mut output);
            prop_assert!(output[0] >= 0.0, "ReLU output must be >= 0, got {}", output[0]);
        }

        #[test]
        fn prop_gelu_zero_at_zero(scale in -1e-10f32..1e-10f32) {
            // GELU near zero should be near zero
            let input = [scale];
            let mut output = [0.0f32];
            gelu_scalar(&input, &mut output);
            prop_assert!(
                output[0].abs() < 1e-6,
                "GELU({scale}) should be ~0, got {}",
                output[0]
            );
        }

        #[test]
        fn prop_silu_sign_preserving(x in proptest::num::f32::NORMAL) {
            // SiLU(x) has the same sign as x (or is zero)
            let input = [x];
            let mut output = [0.0f32];
            silu_scalar(&input, &mut output);
            if x > 0.0 {
                prop_assert!(output[0] >= 0.0, "SiLU({x}) should be >= 0, got {}", output[0]);
            } else if x < 0.0 {
                prop_assert!(output[0] <= 0.0, "SiLU({x}) should be <= 0, got {}", output[0]);
            }
        }
    }

    // ── AVX2 parity tests ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_relu_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];

        relu_scalar(&input, &mut scalar_out);
        unsafe { relu_avx2(&input, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gelu_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.25).collect();
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];

        gelu_scalar(&input, &mut scalar_out);
        unsafe { gelu_avx2(&input, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_silu_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.3).collect();
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];

        silu_scalar(&input, &mut scalar_out);
        unsafe { silu_avx2(&input, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_relu_avx2_non_aligned_length() {
        // Test with length not divisible by 8 to exercise the scalar tail
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (-5..6).map(|i| i as f32).collect(); // 11 elements
        let mut scalar_out = vec![0.0f32; input.len()];
        let mut avx2_out = vec![0.0f32; input.len()];

        relu_scalar(&input, &mut scalar_out);
        unsafe { relu_avx2(&input, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }

    // ── PTX structural tests ─────────────────────────────────────────────

    #[test]
    fn test_relu_ptx_structure() {
        let ptx = relu_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry relu_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(open, close, "unbalanced braces: {open} open vs {close} close");
    }

    #[test]
    fn test_gelu_ptx_structure() {
        let ptx = gelu_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry gelu_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx for exp");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(open, close, "unbalanced braces: {open} open vs {close} close");
    }

    #[test]
    fn test_silu_ptx_structure() {
        let ptx = silu_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry silu_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx for exp");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(open, close, "unbalanced braces: {open} open vs {close} close");
    }

    #[test]
    fn test_ptx_kernels_are_nonempty() {
        assert!(!relu_ptx().is_empty());
        assert!(!gelu_ptx().is_empty());
        assert!(!silu_ptx().is_empty());
    }
}
