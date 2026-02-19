//! Matrix multiplication kernel.
//!
//! Matches `matmul-kernel-v1.yaml`.
//! C\[i,j\] = sum_k A\[i,k\] * B\[k,j\]
//!
//! Each function provides one of three backends:
//! - `fn matmul_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn matmul_avx2(...)` -- AVX2 SIMD implementation
//! - `fn matmul_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Matrix multiplication: C = A * B (row-major).
///
/// A is m x p, B is p x n, C is m x n.
///
/// # Panics
/// Panics if `a.len() != m*p`, `b.len() != p*n`, or `c.len() != m*n`.
pub fn matmul_scalar(a: &[f32], b: &[f32], m: usize, p: usize, n: usize, c: &mut [f32]) {
    assert_eq!(a.len(), m * p, "A dimension mismatch: expected {} got {}", m * p, a.len());
    assert_eq!(b.len(), p * n, "B dimension mismatch: expected {} got {}", p * n, b.len());
    assert_eq!(c.len(), m * n, "C dimension mismatch: expected {} got {}", m * n, c.len());
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..p {
                sum += a[i * p + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 matrix multiplication: 8-wide broadcast-multiply along rows of B.
///
/// For each row i of A and each tile of 8 columns in B, broadcasts each A\[i,k\]
/// and multiplies by 8 consecutive B\[k,j..j+8\], accumulating the result. Scalar
/// tail handles remaining columns.
///
/// # Safety
/// Requires AVX2 and FMA support. Caller must verify with
/// `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`.
///
/// # Panics
/// Panics if `a.len() != m*p`, `b.len() != p*n`, or `c.len() != m*n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_avx2(a: &[f32], b: &[f32], m: usize, p: usize, n: usize, c: &mut [f32]) {
    assert_eq!(a.len(), m * p, "A dimension mismatch: expected {} got {}", m * p, a.len());
    assert_eq!(b.len(), p * n, "B dimension mismatch: expected {} got {}", p * n, b.len());
    assert_eq!(c.len(), m * n, "C dimension mismatch: expected {} got {}", m * n, c.len());

    let simd_width = 8;
    let n_simd = n - (n % simd_width);

    for i in 0..m {
        // Zero-initialize C row
        for j in 0..n {
            *c.get_unchecked_mut(i * n + j) = 0.0;
        }

        for k in 0..p {
            let a_ik = *a.get_unchecked(i * p + k);
            let a_broadcast = _mm256_set1_ps(a_ik);

            // SIMD portion: process 8 columns at a time
            let mut j = 0usize;
            while j < n_simd {
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(k * n + j));
                let c_ptr = c.as_mut_ptr().add(i * n + j);
                let c_vec = _mm256_loadu_ps(c_ptr);
                let result = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                _mm256_storeu_ps(c_ptr, result);
                j += simd_width;
            }

            // Scalar tail for remaining columns
            for j in n_simd..n {
                *c.get_unchecked_mut(i * n + j) += a_ik * *b.get_unchecked(k * n + j);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for the tiled matrix multiplication kernel.
///
/// Uses 16x16 thread blocks with shared memory tiles. Each block computes a
/// 16x16 sub-matrix of C by iterating over tiles of the K dimension.
pub fn matmul_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry matmul_kernel(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 M,
    .param .u32 P,
    .param .u32 N
) {
    .reg .u32 %tx, %ty, %bx, %by, %row, %col;
    .reg .u32 %m, %p, %n, %tile, %kk;
    .reg .u32 %a_col, %b_row;
    .reg .u32 %as_idx, %bs_idx, %tmp32;
    .reg .u64 %a_ptr, %b_ptr, %c_ptr, %addr, %off64;
    .reg .f32 %acc, %a_val, %b_val;
    .reg .pred %p_tile, %p_a, %p_b;
    .shared .f32 As[256];
    .shared .f32 Bs[256];

    // Thread and block indices
    mov.u32 %tx, %tid.x;
    mov.u32 %ty, %tid.y;
    mov.u32 %bx, %ctaid.x;
    mov.u32 %by, %ctaid.y;

    // Load dimensions
    ld.param.u32 %m, [M];
    ld.param.u32 %p, [P];
    ld.param.u32 %n, [N];
    ld.param.u64 %a_ptr, [A];
    ld.param.u64 %b_ptr, [B];
    ld.param.u64 %c_ptr, [C];

    // row = by * 16 + ty, col = bx * 16 + tx
    shl.b32 %row, %by, 4;
    add.u32 %row, %row, %ty;
    shl.b32 %col, %bx, 4;
    add.u32 %col, %col, %tx;

    // Initialize accumulator
    mov.f32 %acc, 0f00000000;

    // Tile loop: tile = 0, 16, 32, ...
    mov.u32 %tile, 0;
TILE_LOOP:
    setp.ge.u32 %p_tile, %tile, %p;
    @%p_tile bra TILE_DONE;

    // Load A[row][tile + tx] into shared memory As[ty][tx]
    add.u32 %a_col, %tile, %tx;
    setp.lt.u32 %p_a, %row, %m;
    setp.lt.u32 %p_b, %a_col, %p;
    and.pred %p_a, %p_a, %p_b;
    // As index = ty * 16 + tx
    shl.b32 %as_idx, %ty, 4;
    add.u32 %as_idx, %as_idx, %tx;
    @!%p_a mov.f32 %a_val, 0f00000000;
    @%p_a mad.lo.u32 %tmp32, %row, %p, %a_col;
    @%p_a mul.wide.u32 %off64, %tmp32, 4;
    @%p_a add.u64 %addr, %a_ptr, %off64;
    @%p_a ld.global.f32 %a_val, [%addr];
    mul.wide.u32 %off64, %as_idx, 4;
    add.u64 %addr, %off64, As;
    st.shared.f32 [%addr], %a_val;

    // Load B[tile + ty][col] into shared memory Bs[ty][tx]
    add.u32 %b_row, %tile, %ty;
    setp.lt.u32 %p_a, %b_row, %p;
    setp.lt.u32 %p_b, %col, %n;
    and.pred %p_a, %p_a, %p_b;
    shl.b32 %bs_idx, %ty, 4;
    add.u32 %bs_idx, %bs_idx, %tx;
    @!%p_a mov.f32 %b_val, 0f00000000;
    @%p_a mad.lo.u32 %tmp32, %b_row, %n, %col;
    @%p_a mul.wide.u32 %off64, %tmp32, 4;
    @%p_a add.u64 %addr, %b_ptr, %off64;
    @%p_a ld.global.f32 %b_val, [%addr];
    mul.wide.u32 %off64, %bs_idx, 4;
    add.u64 %addr, %off64, Bs;
    st.shared.f32 [%addr], %b_val;

    bar.sync 0;

    // Accumulate: acc += As[ty][kk] * Bs[kk][tx] for kk in 0..16
    mov.u32 %kk, 0;
INNER_LOOP:
    setp.ge.u32 %p_tile, %kk, 16;
    @%p_tile bra INNER_DONE;

    // Load As[ty][kk]
    shl.b32 %as_idx, %ty, 4;
    add.u32 %as_idx, %as_idx, %kk;
    mul.wide.u32 %off64, %as_idx, 4;
    add.u64 %addr, %off64, As;
    ld.shared.f32 %a_val, [%addr];

    // Load Bs[kk][tx]
    shl.b32 %bs_idx, %kk, 4;
    add.u32 %bs_idx, %bs_idx, %tx;
    mul.wide.u32 %off64, %bs_idx, 4;
    add.u64 %addr, %off64, Bs;
    ld.shared.f32 %b_val, [%addr];

    fma.rn.f32 %acc, %a_val, %b_val, %acc;

    add.u32 %kk, %kk, 1;
    bra INNER_LOOP;
INNER_DONE:

    bar.sync 0;

    add.u32 %tile, %tile, 16;
    bra TILE_LOOP;
TILE_DONE:

    // Store C[row][col] = acc (if in bounds)
    setp.lt.u32 %p_a, %row, %m;
    setp.lt.u32 %p_b, %col, %n;
    and.pred %p_a, %p_a, %p_b;
    @!%p_a bra EXIT;

    mad.lo.u32 %tmp32, %row, %n, %col;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %c_ptr, %off64;
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
    use proptest::prelude::*;

    // ── Identity matrix test ────────────────────────────────────────────

    #[test]
    fn test_matmul_identity_left() {
        // I * X = X
        let n = 3;
        #[rustfmt::skip]
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let x = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let mut c = vec![0.0f32; n * n];
        matmul_scalar(&identity, &x, n, n, n, &mut c);
        assert_ulp_eq(&c, &x, 0);
    }

    #[test]
    fn test_matmul_identity_right() {
        // X * I = X
        let n = 3;
        #[rustfmt::skip]
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let x = vec![
            2.0, 3.0, 4.0,
            5.0, 6.0, 7.0,
            8.0, 9.0, 10.0,
        ];
        let mut c = vec![0.0f32; n * n];
        matmul_scalar(&x, &identity, n, n, n, &mut c);
        assert_ulp_eq(&c, &x, 0);
    }

    // ── Known 2x2 result ────────────────────────────────────────────────

    #[test]
    fn test_matmul_2x2_known() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        matmul_scalar(&a, &b, 2, 2, 2, &mut c);
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert_ulp_eq(&c, &expected, 0);
    }

    // ── Non-square matmul ───────────────────────────────────────────────

    #[test]
    fn test_matmul_non_square() {
        // A is 2x3, B is 3x2 -> C is 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        matmul_scalar(&a, &b, 2, 3, 2, &mut c);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_ulp_eq(&c, &expected, 0);
    }

    // ── Zero matrix test ────────────────────────────────────────────────

    #[test]
    fn test_matmul_zero() {
        let a = vec![0.0f32; 9];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0f32; 9];
        matmul_scalar(&a, &b, 3, 3, 3, &mut c);
        let expected = vec![0.0f32; 9];
        assert_ulp_eq(&c, &expected, 0);
    }

    // ── Property-based tests ────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_matmul_zero_matrix(m in 1usize..8, p in 1usize..8, n in 1usize..8) {
            let a = vec![0.0f32; m * p];
            let b: Vec<f32> = (0..p*n).map(|i| i as f32).collect();
            let mut c = vec![0.0f32; m * n];
            matmul_scalar(&a, &b, m, p, n, &mut c);
            for &val in &c {
                prop_assert!((val).abs() < 1e-10, "0 * B should be 0, got {}", val);
            }
        }

        #[test]
        fn prop_matmul_dimensions_correct(m in 1usize..6, p in 1usize..6, n in 1usize..6) {
            let a: Vec<f32> = (0..m*p).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..p*n).map(|i| (i as f32) * 0.1).collect();
            let mut c = vec![0.0f32; m * n];
            // Should not panic
            matmul_scalar(&a, &b, m, p, n, &mut c);
            prop_assert_eq!(c.len(), m * n);
        }
    }

    // ── Assertion failure tests ─────────────────────────────────────────

    #[test]
    #[should_panic(expected = "A dimension mismatch")]
    fn test_matmul_bad_a_dim() {
        let a = vec![1.0f32; 5]; // wrong size
        let b = vec![1.0f32; 6];
        let mut c = vec![0.0f32; 4];
        matmul_scalar(&a, &b, 2, 3, 2, &mut c);
    }

    #[test]
    #[should_panic(expected = "B dimension mismatch")]
    fn test_matmul_bad_b_dim() {
        let a = vec![1.0f32; 6];
        let b = vec![1.0f32; 5]; // wrong size
        let mut c = vec![0.0f32; 4];
        matmul_scalar(&a, &b, 2, 3, 2, &mut c);
    }

    #[test]
    #[should_panic(expected = "C dimension mismatch")]
    fn test_matmul_bad_c_dim() {
        let a = vec![1.0f32; 6];
        let b = vec![1.0f32; 6];
        let mut c = vec![0.0f32; 5]; // wrong size
        matmul_scalar(&a, &b, 2, 3, 2, &mut c);
    }

    // ── AVX2 parity tests ───────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_matmul_avx2_parity_square() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let n = 16;
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((n * n - i) as f32) * 0.01).collect();
        let mut scalar_out = vec![0.0f32; n * n];
        let mut avx2_out = vec![0.0f32; n * n];

        matmul_scalar(&a, &b, n, n, n, &mut scalar_out);
        unsafe { matmul_avx2(&a, &b, n, n, n, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_matmul_avx2_parity_non_aligned() {
        // n = 13 (not divisible by 8) to exercise scalar tail
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let m = 5;
        let p = 7;
        let n = 13;
        let a: Vec<f32> = (0..m * p).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..p * n).map(|i| (i as f32) * 0.1).collect();
        let mut scalar_out = vec![0.0f32; m * n];
        let mut avx2_out = vec![0.0f32; m * n];

        matmul_scalar(&a, &b, m, p, n, &mut scalar_out);
        unsafe { matmul_avx2(&a, &b, m, p, n, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_matmul_avx2_parity_small() {
        // Very small matrix: n < 8
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut scalar_out = vec![0.0f32; 4];
        let mut avx2_out = vec![0.0f32; 4];

        matmul_scalar(&a, &b, 2, 2, 2, &mut scalar_out);
        unsafe { matmul_avx2(&a, &b, 2, 2, 2, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    // ── PTX structural tests ────────────────────────────────────────────

    #[test]
    fn test_matmul_ptx_structure() {
        let ptx = matmul_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(ptx.contains(".entry matmul_kernel"), "missing entry point");
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
        assert!(ptx.contains("bar.sync"), "missing barrier synchronization");
        assert!(ptx.contains("fma.rn.f32"), "missing FMA instruction");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(open, close, "unbalanced braces: {open} open vs {close} close");
    }

    #[test]
    fn test_matmul_ptx_nonempty() {
        assert!(!matmul_ptx().is_empty());
    }
}
