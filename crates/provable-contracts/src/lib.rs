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
//! - [`diff`] — Detect drift between contract versions
//! - [`coverage`] — Cross-contract obligation coverage report
//! - [`generate`] — End-to-end codegen to disk
//! - [`graph`] — Contract dependency graph and cycle detection
//! - [`latex`] — LaTeX conversion for contract math notation
//! - [`book_gen`] — mdBook page generation for contracts

pub mod audit;
pub mod binding;
pub mod book_gen;
pub mod coverage;
pub mod diff;
pub mod error;
pub mod generate;
pub mod graph;
pub mod kani_gen;
pub mod latex;
pub mod probar_gen;
pub mod scaffold;
pub mod schema;
