//! Batch normalization kernel.
//!
//! Matches `batchnorm-kernel-v1.yaml`.
//!
//! Training mode: computes per-channel mean/variance from the batch, normalizes,
//! applies affine transform, and updates running statistics via EMA.
//!
//! Inference mode: uses running mean/variance directly for normalization.
//!
//! Input layout: N*C flattened, where N = batch size, C = channels.
//! Element (n, c) is at index `n * c_count + c`.

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Scalar reference implementation of BatchNorm.
///
/// # Arguments
///
/// * `input`        - Flattened `[N, C]` tensor (row-major).
/// * `n`            - Batch size (N).
/// * `c`            - Number of channels (C).
/// * `gamma`        - Per-channel scale, length `C`.
/// * `beta`         - Per-channel bias, length `C`.
/// * `eps`          - Small constant for numerical stability.
/// * `running_mean` - Running mean per channel, length `C`. Updated in training.
/// * `running_var`  - Running variance per channel, length `C`. Updated in training.
/// * `output`       - Output buffer, same shape as `input`.
/// * `momentum`     - EMA momentum for running stats update.
/// * `training`     - If true, compute batch stats and update running stats.
///                     If false, use running stats for normalization.
///
/// # Panics
///
/// Panics if buffer sizes are inconsistent with `n * c`.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm_scalar(
    input: &[f32],
    n: usize,
    c: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    running_mean: &mut [f32],
    running_var: &mut [f32],
    output: &mut [f32],
    momentum: f32,
    training: bool,
) {
    assert_eq!(input.len(), n * c, "input length must be n * c");
    assert_eq!(output.len(), n * c, "output length must be n * c");
    assert_eq!(gamma.len(), c, "gamma length must be c");
    assert_eq!(beta.len(), c, "beta length must be c");
    assert_eq!(running_mean.len(), c, "running_mean length must be c");
    assert_eq!(running_var.len(), c, "running_var length must be c");
    assert!(n > 0 && c > 0, "batchnorm requires n > 0 and c > 0");

    if training {
        // Compute per-channel batch statistics and normalize
        for ch in 0..c {
            // Compute batch mean for this channel
            let mut sum = 0.0_f32;
            for sample in 0..n {
                sum += input[sample * c + ch];
            }
            let batch_mean = sum / n as f32;

            // Compute batch variance for this channel
            let mut var_sum = 0.0_f32;
            for sample in 0..n {
                let diff = input[sample * c + ch] - batch_mean;
                var_sum += diff * diff;
            }
            let batch_var = var_sum / n as f32;

            // Normalize, scale, and shift
            let inv_std = 1.0 / (batch_var + eps).sqrt();
            for sample in 0..n {
                let idx = sample * c + ch;
                output[idx] = gamma[ch] * (input[idx] - batch_mean) * inv_std + beta[ch];
            }

            // Update running statistics (EMA)
            running_mean[ch] = (1.0 - momentum) * running_mean[ch] + momentum * batch_mean;
            running_var[ch] = (1.0 - momentum) * running_var[ch] + momentum * batch_var;
        }
    } else {
        // Inference: use running stats
        for ch in 0..c {
            let inv_std = 1.0 / (running_var[ch] + eps).sqrt();
            for sample in 0..n {
                let idx = sample * c + ch;
                output[idx] = gamma[ch] * (input[idx] - running_mean[ch]) * inv_std + beta[ch];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 BatchNorm — delegates to scalar.
///
/// Batch dimension reduction is irregular for SIMD (strided access across
/// samples for each channel), so this delegates to the scalar implementation.
///
/// # Safety
///
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
///
/// Panics if buffer sizes are inconsistent with `n * c`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn batchnorm_avx2(
    input: &[f32],
    n: usize,
    c: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    running_mean: &mut [f32],
    running_var: &mut [f32],
    output: &mut [f32],
    momentum: f32,
    training: bool,
) {
    batchnorm_scalar(
        input,
        n,
        c,
        gamma,
        beta,
        eps,
        running_mean,
        running_var,
        output,
        momentum,
        training,
    );
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for BatchNorm kernel (training mode).
///
/// One block per channel. Each block reduces across the batch dimension
/// to compute per-channel mean and variance, then normalizes.
pub fn batchnorm_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// BatchNorm kernel (training): 1 block per channel, 256 threads per block.
// Each block computes mean/var for its channel, normalizes, and updates running stats.
// Input layout: [N, C] row-major, element (n, ch) = input[n * C + ch].
.visible .entry batchnorm_kernel(
    .param .u64 input_ptr,
    .param .u64 gamma_ptr,
    .param .u64 beta_ptr,
    .param .u64 output_ptr,
    .param .u64 running_mean_ptr,
    .param .u64 running_var_ptr,
    .param .u32 batch_size,
    .param .u32 channels,
    .param .f32 eps,
    .param .f32 momentum
)
{
    .reg .u32 %tid, %ch, %n_batch, %n_ch, %i, %idx, %stride;
    .reg .u32 %lane, %warp_id, %mask;
    .reg .u64 %in_base, %g_base, %b_base, %out_base;
    .reg .u64 %rm_base, %rv_base, %addr;
    .reg .f32 %val, %diff, %sq;
    .reg .f32 %sum_local, %sum_warp, %batch_mean;
    .reg .f32 %var_local, %var_warp, %batch_var;
    .reg .f32 %inv_std, %eps, %momentum, %nf;
    .reg .f32 %gamma_val, %beta_val, %normed, %result;
    .reg .f32 %old_rm, %old_rv, %new_rm, %new_rv, %one_minus_m;
    .reg .pred %p;
    .shared .f32 smem[32];

    ld.param.u64 %in_base, [input_ptr];
    ld.param.u64 %g_base, [gamma_ptr];
    ld.param.u64 %b_base, [beta_ptr];
    ld.param.u64 %out_base, [output_ptr];
    ld.param.u64 %rm_base, [running_mean_ptr];
    ld.param.u64 %rv_base, [running_var_ptr];
    ld.param.u32 %n_batch, [batch_size];
    ld.param.u32 %n_ch, [channels];
    ld.param.f32 %eps, [eps];
    ld.param.f32 %momentum, [momentum];

    mov.u32 %tid, %tid.x;
    mov.u32 %ch, %ctaid.x;  // 1 block per channel
    mov.u32 %mask, 0xFFFFFFFF;

    // --- Pass 1: compute sum for mean ---
    mov.f32 %sum_local, 0f00000000;
    mov.u32 %i, %tid;
mean_loop:
    setp.ge.u32 %p, %i, %n_batch;
    @%p bra mean_done;
    // idx = i * channels + ch
    mad.lo.u32 %idx, %i, %n_ch, %ch;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    add.f32 %sum_local, %sum_local, %val;
    add.u32 %i, %i, 256;
    bra mean_loop;
mean_done:

    // Warp-level sum reduction
    shfl.sync.down.b32 %sum_warp, %sum_local, 16, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 8, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    and.b32 %lane, %tid, 31;
    shr.b32 %warp_id, %tid, 5;
    setp.eq.u32 %p, %lane, 0;
    @%p st.shared.f32 [smem + %warp_id * 4], %sum_local;
    bar.sync 0;

    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %sum_local, [smem + %tid * 4];
    @!%p mov.f32 %sum_local, 0f00000000;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    // mean = sum / N
    setp.eq.u32 %p, %tid, 0;
    cvt.rn.f32.u32 %nf, %n_batch;
    div.approx.f32 %batch_mean, %sum_local, %nf;
    @%p st.shared.f32 [smem], %batch_mean;
    bar.sync 0;
    ld.shared.f32 %batch_mean, [smem];

    // --- Pass 2: compute variance ---
    mov.f32 %var_local, 0f00000000;
    mov.u32 %i, %tid;
var_loop:
    setp.ge.u32 %p, %i, %n_batch;
    @%p bra var_done;
    mad.lo.u32 %idx, %i, %n_ch, %ch;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    sub.f32 %diff, %val, %batch_mean;
    mul.f32 %sq, %diff, %diff;
    add.f32 %var_local, %var_local, %sq;
    add.u32 %i, %i, 256;
    bra var_loop;
var_done:

    // Warp-level variance reduction
    shfl.sync.down.b32 %var_warp, %var_local, 16, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 8, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 4, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 2, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 1, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;

    and.b32 %lane, %tid, 31;
    shr.b32 %warp_id, %tid, 5;
    setp.eq.u32 %p, %lane, 0;
    @%p st.shared.f32 [smem + %warp_id * 4], %var_local;
    bar.sync 0;

    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %var_local, [smem + %tid * 4];
    @!%p mov.f32 %var_local, 0f00000000;
    shfl.sync.down.b32 %var_warp, %var_local, 4, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 2, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 1, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;

    // var = var_sum / N, inv_std = rsqrt(var + eps)
    setp.eq.u32 %p, %tid, 0;
    div.approx.f32 %batch_var, %var_local, %nf;
    add.f32 %batch_var, %batch_var, %eps;
    rsqrt.approx.f32 %inv_std, %batch_var;
    @%p st.shared.f32 [smem], %inv_std;

    // Also update running stats (thread 0 only)
    @%p {
        // running_mean = (1-m)*running_mean + m*batch_mean
        cvt.u64.u32 %addr, %ch;
        shl.b64 %addr, %addr, 2;
        add.u64 %addr, %rm_base, %addr;
        ld.global.f32 %old_rm, [%addr];
        mov.f32 %one_minus_m, 0f3F800000;
        sub.f32 %one_minus_m, %one_minus_m, %momentum;
        mul.f32 %new_rm, %one_minus_m, %old_rm;
        fma.rn.f32 %new_rm, %momentum, %batch_mean, %new_rm;
        st.global.f32 [%addr], %new_rm;

        // running_var = (1-m)*running_var + m*batch_var (before eps was added)
        // Recompute batch_var without eps
        sub.f32 %batch_var, %batch_var, %eps;
        cvt.u64.u32 %addr, %ch;
        shl.b64 %addr, %addr, 2;
        add.u64 %addr, %rv_base, %addr;
        ld.global.f32 %old_rv, [%addr];
        mul.f32 %new_rv, %one_minus_m, %old_rv;
        fma.rn.f32 %new_rv, %momentum, %batch_var, %new_rv;
        st.global.f32 [%addr], %new_rv;
    }

    bar.sync 0;
    ld.shared.f32 %inv_std, [smem];

    // Load gamma and beta for this channel
    cvt.u64.u32 %addr, %ch;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %g_base, %addr;
    ld.global.f32 %gamma_val, [%addr];
    cvt.u64.u32 %addr, %ch;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %b_base, %addr;
    ld.global.f32 %beta_val, [%addr];

    // --- Pass 3: normalize + affine ---
    mov.u32 %i, %tid;
norm_loop:
    setp.ge.u32 %p, %i, %n_batch;
    @%p bra norm_done;
    mad.lo.u32 %idx, %i, %n_ch, %ch;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    sub.f32 %diff, %val, %batch_mean;
    mul.f32 %normed, %diff, %inv_std;
    fma.rn.f32 %result, %gamma_val, %normed, %beta_val;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out_base, %addr;
    st.global.f32 [%addr], %result;
    add.u32 %i, %i, 256;
    bra norm_loop;
norm_done:

    bar.sync 0;
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

    // ── Scalar known-answer tests ────────────────────────────────────────

    #[test]
    fn test_batchnorm_constant_input_training() {
        // All inputs constant for one channel -> output = beta (when gamma=1)
        // N=4, C=1, all values = 5.0
        let input = [5.0_f32, 5.0, 5.0, 5.0];
        let gamma = [1.0_f32];
        let beta = [3.0_f32];
        let mut running_mean = [0.0_f32];
        let mut running_var = [0.0_f32];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input, 4, 1, &gamma, &beta, 1e-5,
            &mut running_mean, &mut running_var,
            &mut output, 0.1, true,
        );

        // With constant input: mean=5.0, var=0.0, (x-mean)=0
        // output = gamma * 0 + beta = beta = 3.0
        for (i, &o) in output.iter().enumerate() {
            assert!(
                (o - 3.0).abs() < 1e-3,
                "output[{i}] = {o}, expected ~3.0"
            );
        }
    }

    #[test]
    fn test_batchnorm_training_updates_running_stats() {
        let input = [1.0_f32, 2.0, 3.0, 4.0]; // N=4, C=1
        let gamma = [1.0_f32];
        let beta = [0.0_f32];
        let mut running_mean = [0.0_f32];
        let mut running_var = [0.0_f32];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input, 4, 1, &gamma, &beta, 1e-5,
            &mut running_mean, &mut running_var,
            &mut output, 0.1, true,
        );

        // batch_mean = 2.5, batch_var = 1.25
        // running_mean = 0.9*0 + 0.1*2.5 = 0.25
        // running_var = 0.9*0 + 0.1*1.25 = 0.125
        assert!(
            (running_mean[0] - 0.25).abs() < 1e-5,
            "running_mean = {}, expected 0.25",
            running_mean[0]
        );
        assert!(
            (running_var[0] - 0.125).abs() < 1e-5,
            "running_var = {}, expected 0.125",
            running_var[0]
        );
    }

    #[test]
    fn test_batchnorm_inference_uses_running_stats() {
        let input = [1.0_f32, 2.0, 3.0, 4.0]; // N=4, C=1
        let gamma = [1.0_f32];
        let beta = [0.0_f32];
        let mut running_mean = [10.0_f32]; // Intentionally different from batch stats
        let mut running_var = [4.0_f32];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input, 4, 1, &gamma, &beta, 0.0,
            &mut running_mean, &mut running_var,
            &mut output, 0.1, false,
        );

        // Inference uses running stats: inv_std = 1/sqrt(4) = 0.5
        // output[i] = (input[i] - 10) * 0.5
        let inv_std = 1.0 / 4.0_f32.sqrt();
        for (i, &x) in input.iter().enumerate() {
            let expected = (x - 10.0) * inv_std;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "output[{i}] = {}, expected {expected}",
                output[i]
            );
        }

        // Running stats should NOT change during inference
        assert!((running_mean[0] - 10.0).abs() < 1e-10);
        assert!((running_var[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_batchnorm_inference_differs_from_training() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0_f32];
        let beta = [0.0_f32];

        // Training output
        let mut rm_train = [5.0_f32];
        let mut rv_train = [2.0_f32];
        let mut train_out = [0.0_f32; 4];
        batchnorm_scalar(
            &input, 4, 1, &gamma, &beta, 1e-5,
            &mut rm_train, &mut rv_train,
            &mut train_out, 0.1, true,
        );

        // Inference output (with different running stats)
        let mut rm_eval = [5.0_f32];
        let mut rv_eval = [2.0_f32];
        let mut eval_out = [0.0_f32; 4];
        batchnorm_scalar(
            &input, 4, 1, &gamma, &beta, 1e-5,
            &mut rm_eval, &mut rv_eval,
            &mut eval_out, 0.1, false,
        );

        // They should differ since batch stats != running stats
        let mut all_equal = true;
        for i in 0..4 {
            if (train_out[i] - eval_out[i]).abs() > 1e-6 {
                all_equal = false;
            }
        }
        assert!(
            !all_equal,
            "training and inference outputs should differ when running stats != batch stats"
        );
    }

    #[test]
    fn test_batchnorm_multi_channel() {
        // N=2, C=2
        // Channel 0: [1.0, 3.0], mean=2.0, var=1.0
        // Channel 1: [2.0, 4.0], mean=3.0, var=1.0
        let input = [1.0_f32, 2.0, 3.0, 4.0]; // [sample0_ch0, sample0_ch1, sample1_ch0, sample1_ch1]
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let mut running_mean = [0.0_f32, 0.0];
        let mut running_var = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input, 2, 2, &gamma, &beta, 1e-8,
            &mut running_mean, &mut running_var,
            &mut output, 0.1, true,
        );

        // Channel 0: mean=2, var=1, inv_std=1/sqrt(1+eps)~1
        // (1-2)*1 = -1, (3-2)*1 = 1
        assert!((output[0] - (-1.0)).abs() < 1e-3);
        assert!((output[2] - 1.0).abs() < 1e-3);
        // Channel 1: mean=3, var=1
        // (2-3)*1 = -1, (4-3)*1 = 1
        assert!((output[1] - (-1.0)).abs() < 1e-3);
        assert!((output[3] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_batchnorm_single_sample() {
        // N=1 (batch size 1) -> var=0, output=beta when gamma=1
        let input = [7.0_f32, 3.0]; // N=1, C=2
        let gamma = [1.0_f32, 1.0];
        let beta = [5.0_f32, -5.0];
        let mut running_mean = [0.0_f32, 0.0];
        let mut running_var = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 2];

        batchnorm_scalar(
            &input, 1, 2, &gamma, &beta, 1e-5,
            &mut running_mean, &mut running_var,
            &mut output, 0.1, true,
        );

        // With N=1: mean=input, var=0, (x-mean)=0, output=beta
        assert!((output[0] - 5.0).abs() < 1e-3, "output[0] = {}", output[0]);
        assert!((output[1] - (-5.0)).abs() < 1e-3, "output[1] = {}", output[1]);
    }

    #[test]
    #[should_panic(expected = "input length must be n * c")]
    fn test_batchnorm_input_length_mismatch() {
        let input = [1.0_f32, 2.0];
        let gamma = [1.0_f32];
        let beta = [0.0_f32];
        let mut rm = [0.0_f32];
        let mut rv = [0.0_f32];
        let mut output = [0.0_f32; 2];
        batchnorm_scalar(&input, 3, 1, &gamma, &beta, 1e-5, &mut rm, &mut rv, &mut output, 0.1, true);
    }

    #[test]
    #[should_panic(expected = "batchnorm requires n > 0 and c > 0")]
    fn test_batchnorm_zero_batch() {
        let input: [f32; 0] = [];
        let gamma: [f32; 0] = [];
        let beta: [f32; 0] = [];
        let mut rm: [f32; 0] = [];
        let mut rv: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        batchnorm_scalar(&input, 0, 0, &gamma, &beta, 1e-5, &mut rm, &mut rv, &mut output, 0.1, true);
    }

    // ── Property-based tests ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_batchnorm_training_finite(
            n in 2_usize..8,
            c in 1_usize..4,
        ) {
            let total = n * c;
            let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 1.0).collect();
            let gamma = vec![1.0_f32; c];
            let beta = vec![0.0_f32; c];
            let mut running_mean = vec![0.0_f32; c];
            let mut running_var = vec![1.0_f32; c];
            let mut output = vec![0.0_f32; total];

            batchnorm_scalar(
                &input, n, c, &gamma, &beta, 1e-5,
                &mut running_mean, &mut running_var,
                &mut output, 0.1, true,
            );

            for (i, &o) in output.iter().enumerate() {
                prop_assert!(o.is_finite(), "output[{i}] = {o} is not finite");
            }
            for (i, &rv) in running_var.iter().enumerate() {
                prop_assert!(rv >= 0.0, "running_var[{i}] = {rv} is negative");
            }
        }

        #[test]
        fn prop_batchnorm_running_var_nonneg(
            n in 2_usize..8,
            c in 1_usize..4,
            iters in 1_usize..20,
        ) {
            let total = n * c;
            let mut running_mean = vec![0.0_f32; c];
            let mut running_var = vec![1.0_f32; c];

            for step in 0..iters {
                let input: Vec<f32> = (0..total)
                    .map(|i| ((i + step) as f32) * 0.3 - 2.0)
                    .collect();
                let gamma = vec![1.0_f32; c];
                let beta = vec![0.0_f32; c];
                let mut output = vec![0.0_f32; total];

                batchnorm_scalar(
                    &input, n, c, &gamma, &beta, 1e-5,
                    &mut running_mean, &mut running_var,
                    &mut output, 0.1, true,
                );
            }

            for (i, &rv) in running_var.iter().enumerate() {
                prop_assert!(
                    rv >= 0.0,
                    "running_var[{i}] = {rv} after {iters} iterations"
                );
            }
        }
    }

    // ── AVX2 parity tests ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_batchnorm_avx2_parity_training() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (0..16).map(|x| x as f32 * 0.5).collect();
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];

        let mut rm_scalar = vec![0.0_f32; 4];
        let mut rv_scalar = vec![1.0_f32; 4];
        let mut scalar_out = vec![0.0_f32; 16];
        batchnorm_scalar(
            &input, 4, 4, &gamma, &beta, 1e-5,
            &mut rm_scalar, &mut rv_scalar,
            &mut scalar_out, 0.1, true,
        );

        let mut rm_avx2 = vec![0.0_f32; 4];
        let mut rv_avx2 = vec![1.0_f32; 4];
        let mut avx2_out = vec![0.0_f32; 16];
        unsafe {
            batchnorm_avx2(
                &input, 4, 4, &gamma, &beta, 1e-5,
                &mut rm_avx2, &mut rv_avx2,
                &mut avx2_out, 0.1, true,
            );
        }

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
        assert_ulp_eq(&rm_scalar, &rm_avx2, 4);
        assert_ulp_eq(&rv_scalar, &rv_avx2, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_batchnorm_avx2_parity_inference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let gamma = vec![1.0_f32; 3];
        let beta = vec![0.0_f32; 3];

        let mut rm_scalar = vec![2.0_f32; 3];
        let mut rv_scalar = vec![1.0_f32; 3];
        let mut scalar_out = vec![0.0_f32; 12];
        batchnorm_scalar(
            &input, 4, 3, &gamma, &beta, 1e-5,
            &mut rm_scalar, &mut rv_scalar,
            &mut scalar_out, 0.1, false,
        );

        let mut rm_avx2 = vec![2.0_f32; 3];
        let mut rv_avx2 = vec![1.0_f32; 3];
        let mut avx2_out = vec![0.0_f32; 12];
        unsafe {
            batchnorm_avx2(
                &input, 4, 3, &gamma, &beta, 1e-5,
                &mut rm_avx2, &mut rv_avx2,
                &mut avx2_out, 0.1, false,
            );
        }

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    // ── PTX structural tests ─────────────────────────────────────────────

    #[test]
    fn test_batchnorm_ptx_version() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
    }

    #[test]
    fn test_batchnorm_ptx_target() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
    }

    #[test]
    fn test_batchnorm_ptx_entry() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".entry batchnorm_kernel"), "missing entry point");
    }

    #[test]
    fn test_batchnorm_ptx_ret() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains("ret;"), "missing ret instruction");
    }

    #[test]
    fn test_batchnorm_ptx_shared_memory() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
    }

    #[test]
    fn test_batchnorm_ptx_warp_shuffle() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains("shfl.sync"), "missing warp shuffle instructions");
    }

    #[test]
    fn test_batchnorm_ptx_bar_sync() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains("bar.sync"), "missing bar.sync for block synchronization");
    }

    #[test]
    fn test_batchnorm_ptx_balanced_braces() {
        let ptx = batchnorm_ptx();
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(open, close, "unbalanced braces: {open} open vs {close} close");
    }
}
