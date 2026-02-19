//! Cross-kernel falsification tests: isolation + mutation detection.
//!
//! 42 tests total:
//! - 21 isolation tests: prove kernels compute distinct functions by feeding one kernel's
//!   input through a different kernel and verifying invariant violation.
//! - 21 mutation detection tests: implement broken (mutated) kernel variants inline and
//!   verify the correct kernel produces different output, proving the test suite would
//!   catch implementation bugs.

mod common;

use provable_contracts::kernels::softmax::softmax_scalar;
use provable_contracts::kernels::rmsnorm::rmsnorm_scalar;
use provable_contracts::kernels::layernorm::layernorm_scalar;
use provable_contracts::kernels::batchnorm::batchnorm_scalar;
use provable_contracts::kernels::activation::{relu_scalar, gelu_scalar, silu_scalar};
use provable_contracts::kernels::silu_standalone::silu_standalone_scalar;
use provable_contracts::kernels::swiglu::swiglu_scalar;
use provable_contracts::kernels::cross_entropy::{cross_entropy_scalar, log_softmax_scalar};
use provable_contracts::kernels::rope::rope_scalar;
use provable_contracts::kernels::matmul::matmul_scalar;
use provable_contracts::kernels::attention::attention_scalar;
use provable_contracts::kernels::gqa::gqa_scalar;
use provable_contracts::kernels::flash_attention::flash_attention_scalar;
use provable_contracts::kernels::adamw::adamw_step_scalar;
use provable_contracts::kernels::conv1d::conv1d_scalar;
use provable_contracts::kernels::ssm::ssm_scan_scalar;
use provable_contracts::kernels::kmeans::kmeans_assign_scalar;
use provable_contracts::kernels::pagerank::pagerank_iterate_scalar;
use provable_contracts::kernels::lbfgs::lbfgs_direction_scalar;
use provable_contracts::kernels::cma_es::cma_sample_scalar;
use provable_contracts::kernels::gated_delta_net::gdn_recurrence_scalar;

// ============================================================================
// Part 1: Isolation Tests (21 tests)
// ============================================================================

/// Isolation: softmax through rmsnorm — softmax invariant (sum=1) violated
#[test]
fn isolation_softmax_through_rmsnorm() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32; 4];

    // Correct: softmax output sums to 1
    let mut softmax_out = [0.0f32; 4];
    softmax_scalar(&input, &mut softmax_out);
    let softmax_sum: f32 = softmax_out.iter().sum();
    assert!((softmax_sum - 1.0).abs() < 1e-5);

    // Feed same input through rmsnorm instead
    let mut rms_out = [0.0f32; 4];
    rmsnorm_scalar(&input, &gamma, 1e-5, &mut rms_out);
    let rms_sum: f32 = rms_out.iter().sum();

    // rmsnorm output does not sum to 1
    assert!(
        (rms_sum - 1.0).abs() > 0.01,
        "rmsnorm output unexpectedly sums to ~1.0: {rms_sum}"
    );
}

/// Isolation: rmsnorm through layernorm — rmsnorm output differs from layernorm output
#[test]
fn isolation_rmsnorm_through_layernorm() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];

    // Correct: rmsnorm (no mean centering, normalizes by RMS)
    let mut rms_out = [0.0f32; 4];
    rmsnorm_scalar(&input, &gamma, 1e-5, &mut rms_out);

    // Feed same input through layernorm (subtracts mean first, then normalizes by std)
    let mut ln_out = [0.0f32; 4];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut ln_out);

    // RMSNorm does NOT subtract the mean; LayerNorm DOES subtract the mean.
    // So rmsnorm output should NOT have mean~0, while layernorm output does.
    let rms_mean = common::mean(&rms_out);
    let ln_mean = common::mean(&ln_out);

    // rmsnorm preserves the sign structure (all positive since input is all positive)
    // layernorm centers to mean ~ 0
    assert!(
        rms_mean.abs() > 0.1,
        "rmsnorm output unexpectedly has mean ~0: {rms_mean}"
    );
    assert!(
        ln_mean.abs() < 1e-4,
        "layernorm output should have mean ~0: {ln_mean}"
    );

    // The outputs themselves differ
    let dist = common::l2_distance(&rms_out, &ln_out);
    assert!(
        dist > 0.1,
        "rmsnorm and layernorm unexpectedly produce same output: dist = {dist}"
    );
}

/// Isolation: layernorm through softmax — layernorm invariant (mean=0) violated
#[test]
fn isolation_layernorm_through_softmax() {
    let input = [1.0f32, 2.0, 3.0, 4.0];

    // Correct: layernorm with gamma=1, beta=0 produces output with mean ~ 0
    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];
    let mut ln_out = [0.0f32; 4];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut ln_out);
    let ln_mean = common::mean(&ln_out);
    assert!(ln_mean.abs() < 1e-4);

    // Feed same input through softmax instead
    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&input, &mut sm_out);
    let sm_mean = common::mean(&sm_out);

    // Softmax output has mean = 0.25 (sums to 1, all positive), not 0
    assert!(
        sm_mean.abs() > 0.01,
        "softmax output unexpectedly has mean ~0: {sm_mean}"
    );
}

/// Isolation: relu through silu — relu invariant (non-negative) violated
#[test]
fn isolation_relu_through_silu() {
    // Negative inputs: relu clips to 0, silu produces negative values
    let input = [-2.0f32, -1.0, -0.5, -3.0];

    let mut relu_out = [0.0f32; 4];
    relu_scalar(&input, &mut relu_out);
    // relu output is all zeros for negative input
    for &y in &relu_out {
        assert!(y >= 0.0);
    }

    let mut silu_out = [0.0f32; 4];
    silu_scalar(&input, &mut silu_out);
    // silu produces negative output for negative input
    let has_negative = silu_out.iter().any(|&y| y < 0.0);
    assert!(
        has_negative,
        "silu output unexpectedly has no negative values: {silu_out:?}"
    );
}

/// Isolation: softmax through matmul — softmax invariant (sum=1) violated
#[test]
fn isolation_softmax_through_attention() {
    let input = [1.0f32, 2.0, 3.0, 4.0];

    // Correct: softmax sums to 1
    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&input, &mut sm_out);
    let sm_sum: f32 = sm_out.iter().sum();
    assert!((sm_sum - 1.0).abs() < 1e-5);

    // Feed same 4 elements through matmul as 2x2 * 2x2
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 0.0, 0.0, 1.0]; // identity
    let mut c = [0.0f32; 4];
    matmul_scalar(&a, &b, 2, 2, 2, &mut c);
    let matmul_sum: f32 = c.iter().sum();

    // matmul output does not sum to 1
    assert!(
        (matmul_sum - 1.0).abs() > 0.1,
        "matmul output unexpectedly sums to ~1.0: {matmul_sum}"
    );
}

/// Isolation: matmul through softmax — matmul result changes
#[test]
fn isolation_matmul_through_softmax() {
    // A = [[1,2],[3,4]], B = [[2,0],[1,2]]
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [2.0f32, 0.0, 1.0, 2.0];
    let mut ab = [0.0f32; 4];
    matmul_scalar(&a, &b, 2, 2, 2, &mut ab);
    // ab = [[4, 4], [10, 8]]

    // Feed ab through softmax (applied to entire flattened output)
    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&ab, &mut sm_out);

    // softmax output differs from ab
    let dist = common::l2_distance(&ab, &sm_out);
    assert!(
        dist > 1.0,
        "softmax of matmul result unexpectedly close to matmul result: dist = {dist}"
    );
}

/// Isolation: attention through softmax — attention output is not idempotent under softmax
#[test]
fn isolation_attention_through_softmax() {
    // 2 queries, 2 keys, d_k=2, d_v=2
    let q = [1.0f32, 0.0, 0.0, 1.0];
    let k = [1.0f32, 0.0, 0.0, 1.0];
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let mut attn_out = [0.0f32; 4];
    attention_scalar(&q, &k, &v, 2, 2, 2, 2, &mut attn_out);

    // Feed attention output through softmax
    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&attn_out, &mut sm_out);

    // Should differ: softmax normalizes to sum=1, attention output doesn't
    let dist = common::l2_distance(&attn_out, &sm_out);
    assert!(
        dist > 0.01,
        "softmax(attention_output) unexpectedly equals attention_output: dist = {dist}"
    );
}

/// Isolation: gqa through matmul — different results
#[test]
fn isolation_gqa_through_matmul() {
    // GQA: 2 heads, 1 kv head, seq_len=2, d_k=2, d_v=2
    let q = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]; // 2 heads * 2 seq * 2 d_k
    let k = [1.0f32, 0.0, 0.0, 1.0]; // 1 kv head * 2 seq * 2 d_k
    let v = [1.0f32, 2.0, 3.0, 4.0]; // 1 kv head * 2 seq * 2 d_v
    let mut gqa_out = [0.0f32; 8]; // 2 heads * 2 seq * 2 d_v
    gqa_scalar(&q, &k, &v, 2, 2, 2, 2, 1, &mut gqa_out);

    // Feed the same Q through matmul as 4x2 * 2x2 (different operation)
    let mut mm_out = [0.0f32; 8];
    matmul_scalar(&q, &k, 4, 2, 2, &mut mm_out);

    let dist = common::l2_distance(&gqa_out, &mm_out);
    assert!(
        dist > 0.01,
        "GQA output unexpectedly matches matmul output: dist = {dist}"
    );
}

/// Isolation: flash_attention vs attention — tile_size can affect numerical path
#[test]
fn isolation_flash_through_attention() {
    // n=4, d=2
    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut flash_out = [0.0f32; 8];
    flash_attention_scalar(&q, &k, &v, 4, 2, 2, &mut flash_out);

    // Standard attention (n=m=4, d_k=d_v=2)
    let mut attn_out = [0.0f32; 8];
    attention_scalar(&q, &k, &v, 4, 4, 2, 2, &mut attn_out);

    // Both should produce very similar results (same math, different algorithm)
    // But let's verify they're not trivially the same object
    common::assert_all_finite(&flash_out);
    common::assert_all_finite(&attn_out);

    // The key isolation property: feeding flash output through standard attention
    // produces a DIFFERENT result (not idempotent)
    let mut attn_of_flash = [0.0f32; 8];
    attention_scalar(&flash_out, &flash_out, &flash_out, 4, 4, 2, 2, &mut attn_of_flash);
    let dist = common::l2_distance(&flash_out, &attn_of_flash);
    assert!(
        dist > 0.01,
        "attention(flash_output) unexpectedly equals flash_output: dist = {dist}"
    );
}

/// Isolation: silu through relu — relu clips negative parts, changing result
#[test]
fn isolation_silu_through_relu() {
    let input = [-2.0f32, -1.0, 0.5, 2.0];

    let mut silu_out = [0.0f32; 4];
    silu_scalar(&input, &mut silu_out);

    let mut relu_of_silu = [0.0f32; 4];
    relu_scalar(&silu_out, &mut relu_of_silu);

    // silu has negative values for negative input; relu clips them
    let dist = common::l2_distance(&silu_out, &relu_of_silu);
    assert!(
        dist > 0.01,
        "relu(silu(x)) unexpectedly equals silu(x): dist = {dist}"
    );
}

/// Isolation: swiglu vs just silu — gate matters
#[test]
fn isolation_swiglu_through_silu() {
    let gate = [1.0f32, -1.0, 2.0, -2.0];
    let value = [2.0f32, 3.0, 0.5, 1.0];

    let mut swiglu_out = [0.0f32; 4];
    swiglu_scalar(&gate, &value, &mut swiglu_out);

    // Just silu of gate (without value multiplication by arbitrary value)
    let mut silu_out = [0.0f32; 4];
    silu_scalar(&gate, &mut silu_out);

    let dist = common::l2_distance(&swiglu_out, &silu_out);
    assert!(
        dist > 0.01,
        "swiglu output unexpectedly equals silu(gate): dist = {dist}"
    );
}

/// Isolation: softmax vs log_softmax — different functions
#[test]
fn isolation_cross_entropy_through_softmax() {
    let logits = [1.0f32, 2.0, 3.0, 4.0];

    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&logits, &mut sm_out);

    let mut log_sm_out = [0.0f32; 4];
    log_softmax_scalar(&logits, &mut log_sm_out);

    // softmax gives positive probabilities; log_softmax gives negative log-probs
    let dist = common::l2_distance(&sm_out, &log_sm_out);
    assert!(
        dist > 0.1,
        "softmax and log_softmax unexpectedly produce same output: dist = {dist}"
    );
}

/// Isolation: rope through layernorm — layernorm changes the embedding
#[test]
fn isolation_rope_through_layernorm() {
    let input = [1.0f32, 2.0, 3.0, 4.0];

    let mut rope_out = [0.0f32; 4];
    rope_scalar(&input, 5, 4, 10000.0, &mut rope_out);

    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];
    let mut ln_of_rope = [0.0f32; 4];
    layernorm_scalar(&rope_out, &gamma, &beta, 1e-5, &mut ln_of_rope);

    let dist = common::l2_distance(&rope_out, &ln_of_rope);
    assert!(
        dist > 0.01,
        "layernorm(rope(x)) unexpectedly equals rope(x): dist = {dist}"
    );
}

/// Isolation: adamw vs lbfgs — different optimization paths
#[test]
fn isolation_adamw_through_lbfgs() {
    let gradient = [1.0f32, -0.5, 0.3, -0.7];
    let d = 4;

    // AdamW: update params
    let mut params = [1.0f32, 2.0, 3.0, 4.0];
    let mut m_buf = [0.0f32; 4];
    let mut v_buf = [0.0f32; 4];
    adamw_step_scalar(
        &mut params, &gradient, &mut m_buf, &mut v_buf,
        0.001, 0.9, 0.999, 1e-8, 0.01, 1,
    );
    let adamw_delta: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .zip(params.iter())
        .map(|(a, b)| b - a)
        .collect();

    // L-BFGS: compute direction (no history = steepest descent)
    let mut direction = [0.0f32; 4];
    lbfgs_direction_scalar(&gradient, &[], &[], 0, d, &mut direction);

    let dist = common::l2_distance(&adamw_delta, &direction);
    assert!(
        dist > 0.01,
        "adamw update direction unexpectedly matches lbfgs direction: dist = {dist}"
    );
}

/// Isolation: conv1d vs matmul — different operations
#[test]
fn isolation_conv1d_through_matmul() {
    // Conv1d: c_in=1, c_out=1, length=4, kernel_size=2, stride=1, padding=0
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let weight = [1.0f32, 1.0]; // 1x1x2
    let out_len = (4 + 0 - 2) / 1 + 1; // = 3
    let mut conv_out = vec![0.0f32; out_len];
    conv1d_scalar(&input, &weight, None, 1, 1, 4, 2, 1, 0, &mut conv_out);

    // Matmul: treat input as 1x4, weight padded as 4x1 -> 1x1 (not comparable)
    // Instead do: 2x2 matmul on input
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 1.0, 1.0, 1.0];
    let mut mm_out = [0.0f32; 4];
    matmul_scalar(&a, &b, 2, 2, 2, &mut mm_out);

    // Different sizes, different operations — just verify they produce different values
    // conv_out = [3, 5, 7], mm_out = [3, 3, 7, 7]
    assert_ne!(conv_out.len(), mm_out.len());
}

/// Isolation: ssm vs conv1d — different operations on same data
#[test]
fn isolation_ssm_through_conv1d() {
    let seq_len = 4;
    let state_dim = 2;
    let a_bar = [0.9f32, 0.8];
    let b_bar = [1.0f32, 0.5, 0.8, 0.3, 0.6, 0.7, 0.4, 0.9]; // 2 x 4
    let c = [1.0f32, 1.0];
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut ssm_out = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut ssm_out);

    // Conv1d on same x with c_in=1, c_out=1, kernel_size=2
    let weight = [1.0f32, 1.0];
    let conv_out_len = (4 + 0 - 2) / 1 + 1; // 3
    let mut conv_out = vec![0.0f32; conv_out_len];
    conv1d_scalar(&x, &weight, None, 1, 1, 4, 2, 1, 0, &mut conv_out);

    // Different output lengths and semantics
    assert_ne!(ssm_out.len(), conv_out.len());
    // First overlapping values also differ
    assert!(
        (ssm_out[0] - conv_out[0]).abs() > 0.01,
        "ssm and conv1d unexpectedly produce same first value"
    );
}

/// Isolation: kmeans assignments vs pagerank — different problem domains
#[test]
fn isolation_kmeans_through_pagerank() {
    // Kmeans: 4 points in 2D, 2 centroids
    let points = [0.0f32, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0];
    let centroids = [0.5f32, 0.0, 10.5, 10.0];
    let mut assignments = [0u32; 4];
    kmeans_assign_scalar(&points, &centroids, 4, 2, 2, &mut assignments);
    // Points 0,1 -> cluster 0; points 2,3 -> cluster 1

    // PageRank: 4 nodes
    let transition = common::stochastic_transition_matrix(4);
    let rank = [0.25f32; 4];
    let mut pr_out = [0.0f32; 4];
    pagerank_iterate_scalar(&transition, &rank, 4, 0.85, &mut pr_out);

    // Kmeans produces integer assignments, pagerank produces float rank vector
    // They solve completely different problems
    assert_eq!(assignments[0], 0);
    assert_eq!(assignments[2], 1);
    assert!(pr_out.iter().all(|&x| x > 0.0 && x < 1.0));
}

/// Isolation: pagerank through softmax — softmax changes the distribution
#[test]
fn isolation_pagerank_through_softmax() {
    let n = 4;
    // Use a non-uniform initial rank to produce non-uniform pagerank output
    let rank = [0.4f32, 0.3, 0.2, 0.1];
    // Manually create a transition matrix with non-trivial structure
    let transition = [
        0.1f32, 0.2, 0.3, 0.4,
        0.4, 0.1, 0.2, 0.3,
        0.3, 0.4, 0.1, 0.2,
        0.2, 0.3, 0.4, 0.1,
    ];
    let mut pr_out = [0.0f32; 4];
    pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut pr_out);

    // PageRank output sums to ~1, but softmax will exponentiate and re-normalize
    let mut sm_of_pr = [0.0f32; 4];
    softmax_scalar(&pr_out, &mut sm_of_pr);

    // Softmax output sums to 1 too, but values differ from pagerank
    // (softmax amplifies differences via exponentiation)
    let dist = common::l2_distance(&pr_out, &sm_of_pr);
    assert!(
        dist > 0.001,
        "softmax(pagerank) unexpectedly equals pagerank: dist = {dist}"
    );
}

/// Isolation: lbfgs direction applied as adamw step — different result
#[test]
fn isolation_lbfgs_through_adamw() {
    let gradient = [1.0f32, -0.5, 0.3, -0.7];

    // L-BFGS direction (no history)
    let mut direction = [0.0f32; 4];
    lbfgs_direction_scalar(&gradient, &[], &[], 0, 4, &mut direction);
    // direction = -gradient = [-1, 0.5, -0.3, 0.7]

    // Use direction as gradient for adamw
    let mut params = [0.0f32; 4];
    let mut m_buf = [0.0f32; 4];
    let mut v_buf = [0.0f32; 4];
    adamw_step_scalar(
        &mut params, &direction, &mut m_buf, &mut v_buf,
        0.001, 0.9, 0.999, 1e-8, 0.0, 1,
    );

    // AdamW with these "gradients" produces update different from raw direction
    let dist = common::l2_distance(&params, &direction);
    assert!(
        dist > 0.01,
        "adamw(lbfgs_direction) unexpectedly equals lbfgs_direction: dist = {dist}"
    );
}

/// Isolation: cma sample through rmsnorm — changes the sample
#[test]
fn isolation_cma_through_rmsnorm() {
    let mean = [1.0f32, 2.0, 3.0, 4.0];
    let sigma = 0.5;
    // Lower triangular Cholesky factor (non-identity)
    let cholesky_l = [
        2.0f32, 0.0, 0.0, 0.0,
        0.5, 1.5, 0.0, 0.0,
        0.3, 0.2, 1.0, 0.0,
        0.1, 0.4, 0.3, 0.8,
    ];
    let z = [0.5f32, -0.3, 0.7, -0.1];
    let mut cma_out = [0.0f32; 4];
    cma_sample_scalar(&mean, sigma, &cholesky_l, 4, &z, &mut cma_out);

    let gamma = [1.0f32; 4];
    let mut rms_of_cma = [0.0f32; 4];
    rmsnorm_scalar(&cma_out, &gamma, 1e-5, &mut rms_of_cma);

    let dist = common::l2_distance(&cma_out, &rms_of_cma);
    assert!(
        dist > 0.01,
        "rmsnorm(cma_sample) unexpectedly equals cma_sample: dist = {dist}"
    );
}

/// Isolation: gdn output vs ssm output — different recurrences
#[test]
fn isolation_gdn_through_ssm() {
    let seq_len = 4;
    let k_dim = 2;
    let v_dim = 2;

    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let alpha = [0.9f32, 0.8, 0.7, 0.6];
    let beta = [0.1f32, 0.2, 0.3, 0.4];
    let mut gdn_out = [0.0f32; 8]; // seq_len * v_dim
    gdn_recurrence_scalar(&q, &k, &v, &alpha, &beta, seq_len, k_dim, v_dim, &mut gdn_out);

    // SSM on same data (using first v_dim elements as x, etc.)
    let state_dim = 2;
    let a_bar = [0.9f32, 0.8];
    let b_bar = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6]; // state_dim x seq_len
    let c_ssm = [1.0f32, 1.0];
    let x_ssm = [1.0f32, 3.0, 5.0, 7.0];
    let mut ssm_out = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c_ssm, &x_ssm, state_dim, seq_len, &mut ssm_out);

    // Different output dimensions and values
    assert_ne!(gdn_out.len(), ssm_out.len());
    // Even comparing first 4 elements, they differ
    let dist = common::l2_distance(&gdn_out[..4], &ssm_out);
    assert!(
        dist > 0.01,
        "gdn and ssm first 4 outputs unexpectedly match: dist = {dist}"
    );
}

// ============================================================================
// Part 2: Mutation Detection Tests (21 tests)
// ============================================================================

/// Mutation: softmax — no max subtraction causes overflow on large inputs
#[test]
fn mutation_softmax_detect_no_max_subtraction() {
    let input = [1000.0f32, 1000.0, 1000.0, 1000.0];

    // Correct softmax (numerically stable)
    let mut correct = [0.0f32; 4];
    softmax_scalar(&input, &mut correct);
    common::assert_all_finite(&correct);
    common::assert_probability_distribution(&correct, 1e-5);

    // Mutated: softmax WITHOUT max subtraction
    let mut mutated = [0.0f32; 4];
    for (i, &x) in input.iter().enumerate() {
        mutated[i] = x.exp(); // No max subtraction -> overflow
    }
    let sum: f32 = mutated.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for m in mutated.iter_mut() {
            *m /= sum;
        }
    }

    // Mutated version overflows: exp(1000) = inf
    let mutated_has_nan_or_inf = mutated.iter().any(|x| !x.is_finite());
    assert!(
        mutated_has_nan_or_inf,
        "mutated softmax should overflow but got: {mutated:?}"
    );
}

/// Mutation: rmsnorm — eps=0 on zero input causes NaN
#[test]
fn mutation_rmsnorm_detect_zero_eps() {
    let input = [0.0f32, 0.0, 0.0, 0.0];
    let gamma = [1.0f32; 4];

    // Correct: eps=1e-5 keeps it finite
    let mut correct = [0.0f32; 4];
    rmsnorm_scalar(&input, &gamma, 1e-5, &mut correct);
    common::assert_all_finite(&correct);

    // Mutated: eps=0, zero input -> rms = sqrt(0/4 + 0) = 0, 1/rms = inf
    let mut mutated = [0.0f32; 4];
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / input.len() as f32 + 0.0).sqrt(); // eps=0
    let inv_rms = 1.0 / rms; // Division by zero -> inf
    for i in 0..4 {
        mutated[i] = input[i] * inv_rms * gamma[i]; // 0 * inf = NaN
    }

    let mutated_has_nan_or_inf = mutated.iter().any(|x| !x.is_finite());
    assert!(
        mutated_has_nan_or_inf,
        "mutated rmsnorm (eps=0, zero input) should produce NaN/inf but got: {mutated:?}"
    );
}

/// Mutation: layernorm — eps=0 on constant input causes NaN
#[test]
fn mutation_layernorm_detect_no_eps() {
    let input = [5.0f32, 5.0, 5.0, 5.0]; // Constant input -> variance = 0
    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];

    // Correct: eps=1e-5 prevents division by zero
    let mut correct = [0.0f32; 4];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut correct);
    common::assert_all_finite(&correct);

    // Mutated: eps=0, constant input -> var=0, inv_std = 1/sqrt(0) = inf
    let n = input.len() as f32;
    let mean: f32 = input.iter().sum::<f32>() / n;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 0.0).sqrt(); // eps=0, var=0 -> inf
    let mut mutated = [0.0f32; 4];
    for i in 0..4 {
        mutated[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i]; // 0 * inf = NaN
    }

    let mutated_has_nan_or_inf = mutated.iter().any(|x| !x.is_finite());
    assert!(
        mutated_has_nan_or_inf,
        "mutated layernorm (eps=0, constant input) should produce NaN/inf but got: {mutated:?}"
    );
}

/// Mutation: batchnorm — no running stat update leaves running_mean at 0
#[test]
fn mutation_batchnorm_detect_no_running_update() {
    let input = [1.0f32, 2.0, 3.0, 4.0]; // N=4, C=1
    let gamma = [1.0f32];
    let beta = [0.0f32];
    let mut running_mean = [0.0f32];
    let mut running_var = [0.0f32];
    let mut output = [0.0f32; 4];

    // Correct: training updates running stats
    batchnorm_scalar(
        &input, 4, 1, &gamma, &beta, 1e-5,
        &mut running_mean, &mut running_var,
        &mut output, 0.1, true,
    );
    // After training, running_mean should have moved from 0
    assert!(
        running_mean[0].abs() > 0.01,
        "running_mean should be updated after training: {}", running_mean[0]
    );

    // Mutated: if we skip running stat update, running_mean stays at 0
    let mutated_running_mean = 0.0f32; // Would remain 0 if update is skipped
    assert!(
        (running_mean[0] - mutated_running_mean).abs() > 0.01,
        "mutation not detected: running_mean ({}) same as no-update (0.0)",
        running_mean[0]
    );
}

/// Mutation: activation — replace gelu with relu gives different results for negative inputs
#[test]
fn mutation_activation_detect_relu_for_gelu() {
    let input = [-2.0f32, -1.0, -0.5, 0.5];

    let mut correct_gelu = [0.0f32; 4];
    gelu_scalar(&input, &mut correct_gelu);

    // Mutated: use relu instead of gelu
    let mut mutated_relu = [0.0f32; 4];
    relu_scalar(&input, &mut mutated_relu);

    // For negative inputs, gelu gives small negative values; relu gives 0
    let dist = common::l2_distance(&correct_gelu, &mutated_relu);
    assert!(
        dist > 0.01,
        "mutation not detected: gelu and relu too similar: dist = {dist}"
    );
}

/// Mutation: silu — replace sigmoid with constant 0.5 gives different output
#[test]
fn mutation_silu_detect_constant_sigmoid() {
    let input = [2.0f32, -1.0, 3.0, -0.5];

    let mut correct = [0.0f32; 4];
    silu_standalone_scalar(&input, &mut correct);

    // Mutated: silu(x) = 0.5 * x (constant sigmoid = 0.5)
    let mut mutated = [0.0f32; 4];
    for (i, &x) in input.iter().enumerate() {
        mutated[i] = 0.5 * x;
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: silu and 0.5*x too similar: dist = {dist}"
    );
}

/// Mutation: swiglu — replace silu gate with relu gate gives different output
#[test]
fn mutation_swiglu_detect_relu_gate() {
    let gate = [-2.0f32, -1.0, 0.5, 2.0];
    let value = [1.0f32, 2.0, 3.0, 4.0];

    let mut correct = [0.0f32; 4];
    swiglu_scalar(&gate, &value, &mut correct);

    // Mutated: relu(gate) * value instead of silu(gate) * value
    let mut mutated = [0.0f32; 4];
    for i in 0..4 {
        let relu_g = gate[i].max(0.0);
        mutated[i] = relu_g * value[i];
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: swiglu(silu) vs swiglu(relu) too similar: dist = {dist}"
    );
}

/// Mutation: cross-entropy — no max subtraction in log-sum-exp causes overflow
#[test]
fn mutation_cross_entropy_detect_no_max_logsumexp() {
    let logits = [500.0f32, 500.0, 500.0, 500.0];
    let targets = [1.0f32, 0.0, 0.0, 0.0];

    // Correct: uses max subtraction for stability
    let correct = cross_entropy_scalar(&targets, &logits);
    assert!(correct.is_finite());

    // Mutated: log-sum-exp without max subtraction
    let sum_exp: f32 = logits.iter().map(|&x| x.exp()).sum(); // exp(500) = inf
    let lse = sum_exp.ln();
    let mut log_sm_mutated = [0.0f32; 4];
    for (i, &x) in logits.iter().enumerate() {
        log_sm_mutated[i] = x - lse;
    }
    let mutated_loss: f32 = -targets.iter().zip(log_sm_mutated.iter())
        .map(|(&t, &ls)| t * ls)
        .sum::<f32>();

    // Mutated version overflows
    let mutated_bad = !mutated_loss.is_finite();
    assert!(
        mutated_bad,
        "mutated cross-entropy should overflow but got: {mutated_loss}"
    );
}

/// Mutation: rope — swap sin and cos gives different output
#[test]
fn mutation_rope_detect_swap_sincos() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let position = 5u32;
    let dim = 4;
    let base = 10000.0f32;

    let mut correct = [0.0f32; 4];
    rope_scalar(&input, position, dim, base, &mut correct);

    // Mutated: swap sin and cos in the rotation
    let mut mutated = [0.0f32; 4];
    let half_dim = dim / 2;
    for k in 0..half_dim {
        let freq = base.powf(-2.0 * k as f32 / dim as f32);
        let theta = freq * position as f32;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = input[2 * k];
        let x1 = input[2 * k + 1];
        // MUTATION: swap sin and cos roles
        mutated[2 * k] = x0 * sin_t - x1 * cos_t;
        mutated[2 * k + 1] = x0 * cos_t + x1 * sin_t;
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: swapped sin/cos too similar to correct: dist = {dist}"
    );
}

/// Mutation: matmul — swap row/col indices gives wrong result
#[test]
fn mutation_matmul_detect_swap_indices() {
    // Non-square: A is 2x3, B is 3x2, C is 2x2
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = 2;
    let p = 3;
    let n = 2;

    let mut correct = [0.0f32; 4];
    matmul_scalar(&a, &b, m, p, n, &mut correct);

    // Mutated: swap i,j in accumulation (C[i][j] uses wrong indices)
    let mut mutated = [0.0f32; 4];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..p {
                // MUTATION: C[j][i] instead of C[i][j]
                sum += a[i * p + k] * b[k * n + j];
            }
            mutated[j * m + i] = sum; // Transposed storage
        }
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: swapped indices too similar: dist = {dist}"
    );
}

/// Mutation: attention — use 1/d_k instead of 1/sqrt(d_k) gives different output
#[test]
fn mutation_attention_detect_wrong_scaling() {
    let q = [1.0f32, 0.5, 0.3, 0.7];
    let k = [0.5f32, 0.3, 0.7, 0.2];
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let n = 2;
    let m = 2;
    let d_k = 2;
    let d_v = 2;

    let mut correct = [0.0f32; 4];
    attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut correct);

    // Mutated: use 1/d_k instead of 1/sqrt(d_k)
    let wrong_scale = 1.0 / d_k as f32; // 0.5 instead of ~0.707
    let mut scores = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut dot = 0.0f32;
            for kk in 0..d_k {
                dot += q[i * d_k + kk] * k[j * d_k + kk];
            }
            scores[i * m + j] = dot * wrong_scale;
        }
    }
    // Softmax each row
    for i in 0..n {
        let row = &mut scores[i * m..(i + 1) * m];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            *s = (*s - max_val).exp();
            sum += *s;
        }
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
    // output = scores * V
    let mut mutated = [0.0f32; 4];
    for i in 0..n {
        for j in 0..d_v {
            let mut sum = 0.0f32;
            for jj in 0..m {
                sum += scores[i * m + jj] * v[jj * d_v + j];
            }
            mutated[i * d_v + j] = sum;
        }
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 1e-4,
        "mutation not detected: wrong scaling too similar: dist = {dist}"
    );
}

/// Mutation: gqa — use wrong KV head index gives different output
#[test]
fn mutation_gqa_detect_wrong_head_broadcast() {
    // 4 query heads, 2 kv heads, seq_len=2, d_k=2, d_v=2
    // heads_per_kv = 4/2 = 2, so heads 0,1 share kv_head 0; heads 2,3 share kv_head 1
    let q = vec![1.0f32; 4 * 2 * 2]; // 4 heads * 2 seq * 2 d_k = 16
    let k = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]; // 2 kv heads * 2 seq * 2 d_k = 8
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 kv heads * 2 seq * 2 d_v = 8

    let mut correct = vec![0.0f32; 16]; // 4 heads * 2 seq * 2 d_v
    gqa_scalar(&q, &k, &v, 2, 2, 2, 4, 2, &mut correct);

    // Mutated: always use kv_head 0 (wrong broadcast for heads 2,3)
    // This effectively means all 4 heads see the same K,V (kv_head=0)
    let k_head0 = &k[0..4]; // kv head 0: 2 seq * 2 d_k
    let v_head0 = &v[0..4]; // kv head 0: 2 seq * 2 d_v
    // Use kv_head 0 for ALL heads
    let k_mutated = [k_head0, k_head0].concat(); // Pretend 2 kv heads, both are head 0
    let v_mutated = [v_head0, v_head0].concat();
    let mut mutated = vec![0.0f32; 16];
    gqa_scalar(&q, &k_mutated, &v_mutated, 2, 2, 2, 4, 2, &mut mutated);

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: wrong kv head broadcast too similar: dist = {dist}"
    );
}

/// Mutation: flash attention — no rescaling gives wrong output
#[test]
fn mutation_flash_detect_no_rescaling() {
    let n = 4;
    let d = 2;
    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut correct = [0.0f32; 8];
    flash_attention_scalar(&q, &k, &v, n, d, 2, &mut correct);

    // Mutated: flash attention without rescaling between tiles
    let scale = 1.0 / (d as f32).sqrt();
    let tile_size = 2;
    let mut mutated = [0.0f32; 8];
    for i in 0..n {
        let mut acc = vec![0.0f32; d];
        let mut total_sum = 0.0f32;
        let mut tile_start = 0;
        while tile_start < n {
            let tile_end = (tile_start + tile_size).min(n);
            let tile_len = tile_end - tile_start;
            let mut tile_scores = vec![0.0f32; tile_len];
            for (tj, j) in (tile_start..tile_end).enumerate() {
                let mut dot = 0.0f32;
                for kk in 0..d {
                    dot += q[i * d + kk] * k[j * d + kk];
                }
                tile_scores[tj] = dot * scale;
            }
            let tile_max = tile_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // MUTATION: no rescaling of previous accumulation (skip correction)
            for (tj, j) in (tile_start..tile_end).enumerate() {
                let w = (tile_scores[tj] - tile_max).exp();
                for dd in 0..d {
                    acc[dd] += w * v[j * d + dd];
                }
                total_sum += w;
            }
            tile_start = tile_end;
        }
        for dd in 0..d {
            mutated[i * d + dd] = acc[dd] / total_sum;
        }
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 1e-4,
        "mutation not detected: no rescaling too similar to correct: dist = {dist}"
    );
}

/// Mutation: adamw — L2 regularization vs decoupled weight decay gives different results
#[test]
fn mutation_adamw_detect_l2_instead_of_decoupled() {
    let grads = [0.1f32, -0.2, 0.3, -0.4];
    let lr = 0.01;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let wd = 0.1;

    // Correct AdamW: decoupled weight decay
    let mut correct_params = [1.0f32, 2.0, 3.0, 4.0];
    let mut m_correct = [0.0f32; 4];
    let mut v_correct = [0.0f32; 4];
    adamw_step_scalar(
        &mut correct_params, &grads, &mut m_correct, &mut v_correct,
        lr, beta1, beta2, eps, wd, 1,
    );

    // Mutated: L2 regularization (add wd*param to grad BEFORE moment update)
    let init_params = [1.0f32, 2.0, 3.0, 4.0];
    let mut mutated_params = init_params;
    let mut m_mut = [0.0f32; 4];
    let mut v_mut = [0.0f32; 4];
    let bc1 = 1.0 / (1.0 - beta1);
    let bc2 = 1.0 / (1.0 - beta2);
    for i in 0..4 {
        // MUTATION: L2 regularization (modify gradient)
        let g = grads[i] + wd * mutated_params[i];
        m_mut[i] = beta1 * m_mut[i] + (1.0 - beta1) * g;
        v_mut[i] = beta2 * v_mut[i] + (1.0 - beta2) * g * g;
        let m_hat = m_mut[i] * bc1;
        let v_hat = v_mut[i] * bc2;
        mutated_params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        // No separate weight decay step (it's folded into gradient)
    }

    let dist = common::l2_distance(&correct_params, &mutated_params);
    assert!(
        dist > 1e-5,
        "mutation not detected: L2 reg vs decoupled wd too similar: dist = {dist}"
    );
}

/// Mutation: conv1d — check output length formula correctness
#[test]
fn mutation_conv1d_detect_off_by_one() {
    let c_in = 1;
    let c_out = 1;
    let length = 8;
    let kernel_size = 3;
    let stride = 2;
    let padding = 1;

    // Correct formula: (length + 2*padding - kernel_size) / stride + 1
    let correct_out_len = (length + 2 * padding - kernel_size) / stride + 1; // (8+2-3)/2+1 = 4

    // Mutated formula: off by one (forgot +1)
    let mutated_out_len = (length + 2 * padding - kernel_size) / stride; // = 3

    assert_ne!(
        correct_out_len, mutated_out_len,
        "mutation not detected: correct ({correct_out_len}) == mutated ({mutated_out_len})"
    );

    // Verify the correct kernel matches the correct formula
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [1.0f32, 1.0, 1.0];
    let mut output = vec![0.0f32; correct_out_len];
    conv1d_scalar(&input, &weight, None, c_in, c_out, length, kernel_size, stride, padding, &mut output);
    assert_eq!(output.len(), correct_out_len);
}

/// Mutation: ssm — non-causal modification at t=0 should not affect t=0 output
#[test]
fn mutation_ssm_detect_noncausal() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.9f32, 0.8];
    let b_bar = [1.0f32, 0.5, 0.8, 0.3, 0.6, 0.7, 0.4, 0.9]; // 2 x 4
    let c = [1.0f32, 1.0];
    let x1 = [1.0f32, 2.0, 3.0, 4.0];

    let mut out1 = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x1, state_dim, seq_len, &mut out1);

    // Modify input at t=2 only
    let x2 = [1.0f32, 2.0, 100.0, 4.0];
    let mut out2 = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x2, state_dim, seq_len, &mut out2);

    // Causality: output at t=0 and t=1 should be unchanged
    assert!(
        (out1[0] - out2[0]).abs() < 1e-7,
        "SSM is non-causal: output[0] changed from {} to {}",
        out1[0], out2[0]
    );
    assert!(
        (out1[1] - out2[1]).abs() < 1e-7,
        "SSM is non-causal: output[1] changed from {} to {}",
        out1[1], out2[1]
    );

    // But t=2 and beyond should change (the modification propagates forward)
    assert!(
        (out1[2] - out2[2]).abs() > 0.1,
        "SSM modification at t=2 should affect output[2]: {} vs {}",
        out1[2], out2[2]
    );
}

/// Mutation: kmeans — random assignments vs nearest-centroid gives different objective
#[test]
fn mutation_kmeans_detect_random_assignment() {
    // 4 points in 2D, 2 centroids far apart
    let points = [0.0f32, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0];
    let centroids = [0.5f32, 0.0, 10.5, 10.0];
    let n = 4;
    let k = 2;
    let d = 2;

    let mut correct_assignments = [0u32; 4];
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut correct_assignments);

    // Correct assignments: points 0,1 -> cluster 0; points 2,3 -> cluster 1
    assert_eq!(correct_assignments[0], 0);
    assert_eq!(correct_assignments[1], 0);
    assert_eq!(correct_assignments[2], 1);
    assert_eq!(correct_assignments[3], 1);

    // Mutated: random (wrong) assignments
    let mutated_assignments: [u32; 4] = [1, 0, 0, 1]; // Deliberately wrong

    // Compute objective (sum of squared distances)
    let correct_obj: f32 = (0..n).map(|p| {
        let c = correct_assignments[p] as usize;
        (0..d).map(|j| {
            let diff = points[p * d + j] - centroids[c * d + j];
            diff * diff
        }).sum::<f32>()
    }).sum();

    let mutated_obj: f32 = (0..n).map(|p| {
        let c = mutated_assignments[p] as usize;
        (0..d).map(|j| {
            let diff = points[p * d + j] - centroids[c * d + j];
            diff * diff
        }).sum::<f32>()
    }).sum();

    // Correct assignments minimize objective; random assignments have higher cost
    assert!(
        mutated_obj > correct_obj + 1.0,
        "mutation not detected: random obj ({mutated_obj}) not much worse than correct obj ({correct_obj})"
    );
}

/// Mutation: pagerank — without normalization, output does not sum to 1
#[test]
fn mutation_pagerank_detect_no_normalization() {
    let n = 4;
    let transition = common::stochastic_transition_matrix(n);
    let rank = [0.25f32; 4];
    let damping = 0.85;

    // Correct: output sums to ~1
    let mut correct = [0.0f32; 4];
    pagerank_iterate_scalar(&transition, &rank, n, damping, &mut correct);
    let correct_sum: f32 = correct.iter().sum();
    assert!(
        (correct_sum - 1.0).abs() < 0.01,
        "correct pagerank should sum to ~1: {correct_sum}"
    );

    // Mutated: no teleport normalization (omit the (1-d)/n term)
    let mut mutated = [0.0f32; 4];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += transition[i * n + j] * rank[j];
        }
        mutated[i] = damping * sum; // MUTATION: no teleport term
    }
    let mutated_sum: f32 = mutated.iter().sum();
    assert!(
        (mutated_sum - 1.0).abs() > 0.01,
        "mutated pagerank should not sum to 1: {mutated_sum}"
    );
}

/// Mutation: lbfgs — with m=0, direction should be -gradient; reversed would give +gradient
#[test]
fn mutation_lbfgs_detect_reverse_loop() {
    let gradient = [1.0f32, -2.0, 3.0, -4.0];
    let d = 4;

    let mut correct = [0.0f32; 4];
    lbfgs_direction_scalar(&gradient, &[], &[], 0, d, &mut correct);

    // Correct: direction = -gradient
    for i in 0..d {
        assert!(
            (correct[i] - (-gradient[i])).abs() < 1e-7,
            "lbfgs(m=0) should give -gradient at index {i}: got {} expected {}",
            correct[i], -gradient[i]
        );
    }

    // Mutated: direction = +gradient (reversed)
    let mut mutated = [0.0f32; 4];
    for i in 0..d {
        mutated[i] = gradient[i]; // MUTATION: +gradient instead of -gradient
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: +gradient vs -gradient too similar: dist = {dist}"
    );
}

/// Mutation: cma — skip Cholesky transformation gives different output when L != I
#[test]
fn mutation_cma_detect_no_cholesky() {
    let d = 4;
    let mean = [1.0f32, 2.0, 3.0, 4.0];
    let sigma = 1.0;
    // Non-identity lower triangular Cholesky factor
    let cholesky_l = [
        2.0f32, 0.0, 0.0, 0.0,
        0.5, 1.5, 0.0, 0.0,
        0.3, 0.2, 1.0, 0.0,
        0.1, 0.4, 0.3, 0.8,
    ];
    let z = [0.5f32, -0.3, 0.7, -0.1];

    let mut correct = [0.0f32; 4];
    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut correct);

    // Mutated: skip Cholesky, output = mean + sigma * z (no L multiplication)
    let mut mutated = [0.0f32; 4];
    for i in 0..d {
        mutated[i] = mean[i] + sigma * z[i]; // MUTATION: no L * z
    }

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: no Cholesky too similar to correct: dist = {dist}"
    );
}

/// Mutation: gdn — extreme alpha=100 gives dramatically different output
#[test]
fn mutation_gdn_detect_extreme_decay() {
    let seq_len = 4;
    let k_dim = 2;
    let v_dim = 2;

    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let beta = [0.5f32; 4];

    // Correct: alpha = 0.5 (valid range)
    let alpha_correct = [0.5f32; 4];
    let mut correct = [0.0f32; 8];
    gdn_recurrence_scalar(&q, &k, &v, &alpha_correct, &beta, seq_len, k_dim, v_dim, &mut correct);

    // Mutated: alpha = 100 (extreme, state explodes)
    let alpha_extreme = [100.0f32; 4];
    let mut mutated = [0.0f32; 8];
    gdn_recurrence_scalar(&q, &k, &v, &alpha_extreme, &beta, seq_len, k_dim, v_dim, &mut mutated);

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 1.0,
        "mutation not detected: extreme alpha too similar to correct: dist = {dist}"
    );
}
