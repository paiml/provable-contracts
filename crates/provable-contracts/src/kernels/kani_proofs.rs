//! Kani bounded proof harnesses for kernel contracts.
//!
//! This module contains 45 `#[kani::proof]` harnesses that promote key
//! properties from Level 3 (statistically tested via proptest) to Level 4
//! (bounded mathematical proof for ALL inputs up to size N).
//!
//! All code here is behind `#[cfg(kani)]` and invisible to normal builds.

use super::activation;
use super::adamw;
use super::attention;
use super::batchnorm;
use super::cma_es;
use super::conv1d;
use super::cross_entropy;
use super::flash_attention;
use super::gated_delta_net;
use super::gqa;
use super::kmeans;
use super::layernorm;
use super::lbfgs;
use super::matmul;
use super::pagerank;
use super::rmsnorm;
use super::rope;
use super::silu_standalone;
use super::softmax;
use super::ssm;
use super::swiglu;

// ════════════════════════════════════════════════════════════════════════════
// Float transcendental stubs
// ════════════════════════════════════════════════════════════════════════════

/// Stub for `f32::exp` — returns an arbitrary positive finite value.
fn stub_exp(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r > 0.0 && r.is_finite());
    r
}

/// Stub for `f32::sqrt` — returns an arbitrary non-negative finite value.
fn stub_sqrt(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r >= 0.0 && r.is_finite());
    r
}

/// Stub for `f32::ln` — returns an arbitrary finite value.
fn stub_ln(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

/// Stub for `f32::sin` — returns a value in [-1, 1].
fn stub_sin(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r >= -1.0 && r <= 1.0);
    r
}

/// Stub for `f32::cos` — returns a value in [-1, 1].
fn stub_cos(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r >= -1.0 && r <= 1.0);
    r
}

/// Stub for `f32::tanh` — returns a value in (-1, 1).
fn stub_tanh(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r > -1.0 && r < 1.0 && r.is_finite());
    r
}

/// Stub for `f32::powf` — returns an arbitrary positive finite value.
fn stub_powf(_base: f32, _exp: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r > 0.0 && r.is_finite());
    r
}

/// Stub for `f32::powi` — returns an arbitrary finite value.
fn stub_powi(_base: f32, _exp: i32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

// Group A (Activation) + Group B (Normalization): 14 harnesses
include!("kani_proofs_ab.rs");

// Group C (Gated + Positional + Loss) + Group D (Matrix + Attention): 14 harnesses
include!("kani_proofs_cd.rs");

// Group E1 (Optimizer + Sequence): 7 harnesses
include!("kani_proofs_e1.rs");

// Group E2 (Classical ML): 10 harnesses
include!("kani_proofs_e2.rs");
