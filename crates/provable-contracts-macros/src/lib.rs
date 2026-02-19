//! # provable-contracts-macros
//!
//! Proc macros for compile-time contract enforcement.
//!
//! ## `#[contract]` Attribute
//!
//! Annotates a function with a provable-contracts YAML contract reference.
//! At compile time, verifies the contract exists (via build.rs env vars)
//! and registers the binding for audit.
//!
//! ```rust,ignore
//! use provable_contracts_macros::contract;
//!
//! #[contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
//! pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
//!     // ...
//! }
//! ```
//!
//! ## How It Works
//!
//! 1. **build.rs** in the consuming crate reads `binding.yaml` and sets
//!    `CONTRACT_<NAME>_<EQ>=bound` env vars for each implemented binding.
//!
//! 2. `#[contract("name", equation = "eq")]` expands to a `const` that reads
//!    the corresponding env var via `env!()`. Missing env var = compile error.
//!
//! 3. A static string in a dedicated link section registers the binding for
//!    runtime audit (when `contract-audit` feature is enabled).

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, ItemFn, Lit, Meta, Token, parse_macro_input};

/// Arguments to `#[contract("contract-name", equation = "equation-name")]`
struct ContractArgs {
    contract_name: String,
    equation_name: String,
}

impl Parse for ContractArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse the contract name (first positional string literal)
        let contract_lit: Lit = input.parse()?;
        let contract_name = match &contract_lit {
            Lit::Str(s) => s.value(),
            _ => {
                return Err(syn::Error::new_spanned(
                    contract_lit,
                    "expected string literal for contract name",
                ));
            }
        };

        // Parse comma
        input.parse::<Token![,]>()?;

        // Parse `equation = "name"`
        let meta: Meta = input.parse()?;
        let equation_name = match &meta {
            Meta::NameValue(nv) if nv.path.is_ident("equation") => match &nv.value {
                Expr::Lit(expr_lit) => match &expr_lit.lit {
                    Lit::Str(s) => s.value(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "expected string literal for equation name",
                        ));
                    }
                },
                _ => {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "expected string literal for equation name",
                    ));
                }
            },
            _ => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "expected `equation = \"...\"`",
                ));
            }
        };

        Ok(ContractArgs {
            contract_name,
            equation_name,
        })
    }
}

/// Compile-time contract enforcement attribute.
///
/// Annotates a function with a provable-contracts YAML contract reference.
/// The macro generates:
///
/// 1. A `const` assertion that reads a `CONTRACT_<NAME>_<EQ>` env var
///    (set by build.rs). If the env var is missing, compilation fails.
///
/// 2. A static binding registration string (in a dedicated link section
///    when `contract-audit` is enabled) for runtime traceability.
///
/// # Arguments
///
/// - First argument: contract YAML name (e.g., `"rmsnorm-kernel-v1"`)
/// - `equation`: equation name within the contract (e.g., `"rmsnorm"`)
///
/// # Example
///
/// ```rust,ignore
/// #[contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
/// pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
///     // Implementation must satisfy proof obligations from the YAML
///     todo!()
/// }
/// ```
///
/// # Compile-Time Behavior
///
/// If build.rs has NOT set `CONTRACT_RMSNORM_KERNEL_V1_RMSNORM=bound`,
/// compilation fails with:
///
/// ```text
/// error: environment variable `CONTRACT_RMSNORM_KERNEL_V1_RMSNORM` not defined
/// ```
#[proc_macro_attribute]
pub fn contract(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ContractArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let env_key = make_env_key(&args.contract_name, &args.equation_name);
    let const_name = format_ident!(
        "_CONTRACT_CHECK_{}_{}",
        args.contract_name.to_uppercase().replace(['-', '.'], "_"),
        args.equation_name.to_uppercase().replace(['-', '.'], "_")
    );

    let contract_name = &args.contract_name;
    let equation_name = &args.equation_name;
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();

    let binding_const_name = format_ident!(
        "_CONTRACT_BINDING_{}_{}",
        args.contract_name.to_uppercase().replace(['-', '.'], "_"),
        args.equation_name.to_uppercase().replace(['-', '.'], "_")
    );

    // Place const assertions inside the function body so the macro works
    // in all contexts: free functions, inherent methods, AND trait impl methods.
    // Trait impls forbid associated consts not declared by the trait, but
    // consts inside a function body are always legal.
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_stmts = &input_fn.block.stmts;

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            // 1. Compile-time contract existence check.
            //    build.rs must set this env var. Missing = compile error.
            const #const_name: &str = env!(#env_key);

            // 2. Binding registration for audit/traceability.
            //    Encodes contract, equation, module, and function name.
            #[allow(dead_code)]
            const #binding_const_name: &str = concat!(
                "contract=", #contract_name,
                ",equation=", #equation_name,
                ",module=", module_path!(),
                ",function=", #fn_name_str,
            );

            // 3. The original function body.
            #(#fn_stmts)*
        }
    };

    TokenStream::from(expanded)
}

/// Generate the env var key from contract name and equation name.
///
/// Convention: `CONTRACT_<CONTRACT_UPPER>_<EQUATION_UPPER>`
/// where hyphens and dots are replaced with underscores.
///
/// Example: `("rmsnorm-kernel-v1", "rmsnorm")` → `"CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"`
fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_env_key() {
        assert_eq!(
            make_env_key("rmsnorm-kernel-v1", "rmsnorm"),
            "CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"
        );
        assert_eq!(
            make_env_key("attention-kernel-v1", "scaled_dot_product"),
            "CONTRACT_ATTENTION_KERNEL_V1_SCALED_DOT_PRODUCT"
        );
        assert_eq!(
            make_env_key("gated-delta-net-v1", "decay"),
            "CONTRACT_GATED_DELTA_NET_V1_DECAY"
        );
    }

    #[test]
    fn test_make_env_key_with_dots() {
        assert_eq!(make_env_key("v1.0", "eq.1"), "CONTRACT_V1_0_EQ_1");
    }
}
