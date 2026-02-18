//! # provable-contracts
//!
//! Papers to Math to Contracts in Code.
//!
//! A Rust library for converting peer-reviewed research papers into
//! mathematically provable kernel implementations via YAML contract
//! intermediaries with Kani bounded model checking verification.
//!
//! ## Modules
//!
//! - [`schema`] — Parse and validate YAML kernel contracts
//! - [`scaffold`] — Generate Rust trait stubs + failing tests from contracts
//! - [`kani`] — Generate `#[kani::proof]` harnesses from contracts
//! - [`probar`] — Generate probar property-based tests from contracts
//! - [`audit`] — Trace paper→equation→contract→test→proof chain
//! - [`binding`] — Map contract equations to implementation functions

pub mod schema;
pub mod scaffold;
pub mod kani_gen;
pub mod probar_gen;
pub mod audit;
pub mod binding;
pub mod error;
