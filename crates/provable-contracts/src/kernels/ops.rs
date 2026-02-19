//! Shared kernel primitives: dot product, softmax row, score matrix.
//!
//! These building blocks are used across attention, GQA, and flash attention kernels.
//! Centralizing them eliminates duplicated DataTransformation patterns.

/// Dot product of two slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// In-place softmax over a contiguous row.
///
/// Uses the numerically stable formulation: subtract max, exponentiate, normalize.
pub fn softmax_row(row: &mut [f32]) {
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Compute scaled dot-product score matrix: `scores[i,j] = Q[i] . K[j] / sqrt(d)`.
///
/// Q is `m x d`, K is `n x d`, scores is `m x n` (row-major).
pub fn score_matrix(q: &[f32], k: &[f32], m: usize, n: usize, d: usize, scores: &mut [f32]) {
    debug_assert_eq!(q.len(), m * d);
    debug_assert_eq!(k.len(), n * d);
    debug_assert_eq!(scores.len(), m * n);
    let scale = 1.0 / (d as f32).sqrt();

    for i in 0..m {
        for j in 0..n {
            scores[i * n + j] = dot(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]) * scale;
        }
    }
}

/// Weighted sum: `output[i] += weight * v_row[i]` for accumulation in attention.
#[inline]
pub fn weighted_accumulate(output: &mut [f32], weight: f32, v_row: &[f32]) {
    debug_assert_eq!(output.len(), v_row.len());
    for (o, v) in output.iter_mut().zip(v_row.iter()) {
        *o += weight * v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_zero() {
        assert_eq!(dot(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn softmax_row_uniform() {
        let mut row = vec![1.0; 4];
        softmax_row(&mut row);
        for v in &row {
            assert!((*v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_matrix_basic() {
        // 1x2 Q, 1x2 K => 1x1 scores
        let q = [1.0, 0.0];
        let k = [1.0, 0.0];
        let mut scores = [0.0f32; 1];
        score_matrix(&q, &k, 1, 1, 2, &mut scores);
        // dot = 1.0, scale = 1/sqrt(2) ≈ 0.707
        assert!((scores[0] - 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn weighted_accumulate_basic() {
        let mut out = [1.0, 2.0];
        weighted_accumulate(&mut out, 0.5, &[4.0, 6.0]);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }
}
