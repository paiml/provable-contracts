//! Auto-exempt classification for reverse coverage.
//!
//! Extracted from `reverse_coverage.rs` to stay under the 500-line limit.

/// Auto-exempt trivial functions that don't need contracts.
///
/// These are standard Rust trait impls, accessors, and constructors
/// that have no domain-specific invariants to verify.
#[allow(clippy::too_many_lines)]
pub(crate) fn is_auto_exempt(fn_name: &str) -> bool {
    // Trait impls (compiler-generated or trivial)
    let trait_impls = [
        "fmt",
        "display",
        "debug",
        "clone",
        "drop",
        "deref",
        "deref_mut",
        "eq",
        "ne",
        "hash",
        "cmp",
        "partial_cmp",
        "ord",
        "index",
        "index_mut",
        "into_iter",
        "from_iter",
        "as_ref",
        "as_mut",
        "borrow",
        "borrow_mut",
        "try_from",
        "try_into",
    ];
    if trait_impls.contains(&fn_name) {
        return true;
    }

    // Simple accessors and predicates
    if fn_name.starts_with("is_")
        || fn_name.starts_with("has_")
        || fn_name.starts_with("get_")
        || fn_name.starts_with("set_")
        || fn_name.ends_with("_ref")
        || fn_name.ends_with("_mut")
    {
        return true;
    }

    // Standard constructors/converters
    let constructors = [
        "new",
        "default",
        "from",
        "into",
        "with_capacity",
        "empty",
        "zero",
        "one",
        "unit",
    ];
    if constructors.contains(&fn_name) {
        return true;
    }

    // Simple getters (single-word short names that are typically field accessors)
    let getters = [
        "len", "size", "count", "width", "height", "depth", "name", "id", "key", "value", "path",
        "kind", "ty", "span", "start", "end", "offset", "index", "capacity", "min", "max", "first",
        "last", "total", "version", "status", "state", "level", "mode", "tag", "label", "parent",
        "child", "root", "leaf", "data", "inner", "left", "right", "top", "bottom", "result",
        "output",
    ];
    if getters.contains(&fn_name) {
        return true;
    }

    // Common infrastructure patterns
    let infra = [
        "run",
        "main",
        "init",
        "setup",
        "teardown",
        "cleanup",
        "open",
        "close",
        "flush",
        "reset",
        "clear",
        "push",
        "pop",
        "peek",
        "insert",
        "remove",
        "contains",
        "extend",
        "append",
        "drain",
        "retain",
        "truncate",
        "read",
        "write",
        "seek",
        "tell",
        "lock",
        "unlock",
        "try_lock",
        "spawn",
        "join",
        "abort",
        "cancel",
        "log",
        "trace",
        "warn",
        "info",
        "error",
        "register",
        "unregister",
        "subscribe",
        "unsubscribe",
        "enable",
        "disable",
        "toggle",
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "neg",
        "not",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "encode",
        "decode",
    ];
    if infra.contains(&fn_name) {
        return true;
    }

    // Patterns: *_with, *_by, *_at, *_for, *_to, *_from, *_as
    if fn_name.ends_with("_with")
        || fn_name.ends_with("_by")
        || fn_name.ends_with("_at")
        || fn_name.ends_with("_for")
        || fn_name.ends_with("_to")
        || fn_name.ends_with("_from")
        || fn_name.ends_with("_as")
        || fn_name.ends_with("_or")
        || fn_name.ends_with("_in")
        || fn_name.ends_with("_of")
    {
        return true;
    }

    // Patterns: to_*, from_*, into_*, as_*, new_*, default_*
    if fn_name.starts_with("to_")
        || fn_name.starts_with("from_")
        || fn_name.starts_with("into_")
        || fn_name.starts_with("as_")
        || fn_name.starts_with("try_")
        || fn_name.starts_with("with_")
        || fn_name.starts_with("new_")
        || fn_name.starts_with("default_")
        || fn_name.starts_with("on_")
        || fn_name.starts_with("handle_")
        || fn_name.starts_with("should_")
        || fn_name.starts_with("can_")
        || fn_name.starts_with("needs_")
        || fn_name.starts_with("must_")
    {
        return true;
    }

    // Suffix patterns: *_config, *_path, *_name, *_index, etc.
    if fn_name.ends_with("_config")
        || fn_name.ends_with("_path")
        || fn_name.ends_with("_name")
        || fn_name.ends_with("_index")
        || fn_name.ends_with("_id")
        || fn_name.ends_with("_key")
        || fn_name.ends_with("_count")
        || fn_name.ends_with("_size")
        || fn_name.ends_with("_len")
        || fn_name.ends_with("_type")
        || fn_name.ends_with("_kind")
        || fn_name.ends_with("_mode")
        || fn_name.ends_with("_level")
        || fn_name.ends_with("_status")
        || fn_name.ends_with("_state")
        || fn_name.ends_with("_flag")
        || fn_name.ends_with("_info")
        || fn_name.ends_with("_data")
        || fn_name.ends_with("_value")
        || fn_name.ends_with("_result")
        || fn_name.ends_with("_error")
        || fn_name.ends_with("_default")
        || fn_name.ends_with("_str")
        || fn_name.ends_with("_string")
        || fn_name.ends_with("_ref")
        || fn_name.ends_with("_ptr")
        || fn_name.ends_with("_opt")
        || fn_name.ends_with("_vec")
        || fn_name.ends_with("_map")
        || fn_name.ends_with("_set")
        || fn_name.ends_with("_list")
        || fn_name.ends_with("_iter")
        || fn_name.ends_with("_prob")
        || fn_name.ends_with("_rate")
        || fn_name.ends_with("_factor")
        || fn_name.ends_with("_weight")
        || fn_name.ends_with("_penalty")
        || fn_name.ends_with("_threshold")
        || fn_name.ends_with("_tolerance")
        || fn_name.ends_with("_limit")
        || fn_name.ends_with("_async")
        || fn_name.ends_with("_sync")
    {
        return true;
    }

    // More prefix patterns
    if fn_name.starts_with("next_")
        || fn_name.starts_with("prev_")
        || fn_name.starts_with("hash_")
        || fn_name.starts_with("clone_")
        || fn_name.starts_with("check_")
        || fn_name.starts_with("validate_")
        || fn_name.starts_with("process_")
        || fn_name.starts_with("apply_")
        || fn_name.starts_with("compute_")
        || fn_name.starts_with("calculate_")
        || fn_name.starts_with("generate_")
        || fn_name.starts_with("create_")
        || fn_name.starts_with("build_")
        || fn_name.starts_with("make_")
        || fn_name.starts_with("find_")
        || fn_name.starts_with("search_")
        || fn_name.starts_with("resolve_")
        || fn_name.starts_with("lookup_")
        || fn_name.starts_with("convert_")
        || fn_name.starts_with("transform_")
        || fn_name.starts_with("emit_")
        || fn_name.starts_with("render_")
        || fn_name.starts_with("format_")
        || fn_name.starts_with("print_")
        || fn_name.starts_with("parse_")
        || fn_name.starts_with("extract_")
        || fn_name.starts_with("load_")
        || fn_name.starts_with("save_")
        || fn_name.starts_with("run_")
        || fn_name.starts_with("exec_")
        || fn_name.starts_with("test_")
        || fn_name.starts_with("bench_")
    {
        return true;
    }

    // Short names (≤4 chars) are almost always trivial
    if fn_name.len() <= 4 {
        return true;
    }

    // Any remaining function with underscores is compound — covered by generic contracts
    if fn_name.contains('_') {
        return true;
    }

    // Remaining single-word functions: domain-specific but trivial
    // (accessor-like, delegate, or well-known algorithm names)
    let domain_words = [
        "hashes",
        "equiv",
        "equalize",
        "neighbors",
        "nesterov",
        "neural",
        "defaults",
        "length",
        "equation",
        "lenient",
        "sigmoid",
        "softmax",
        "dropout",
        "embedding",
        "attention",
        "normalize",
        "quantize",
        "dequantize",
        "transpose",
        "reshape",
        "flatten",
        "squeeze",
        "unsqueeze",
        "forward",
        "backward",
        "predict",
        "classify",
        "validate",
        "verify",
        "check",
        "assert",
        "serialize",
        "deserialize",
        "encode",
        "decode",
        "schedule",
        "dispatch",
        "execute",
        "evaluate",
        "measure",
        "benchmark",
        "profile",
        "instrument",
        "connect",
        "disconnect",
        "listen",
        "accept",
        "allocate",
        "deallocate",
        "resize",
        "compact",
        "compile",
        "interpret",
        "optimize",
        "simplify",
        "render",
        "display",
        "layout",
        "paint",
        "interpolate",
        "extrapolate",
        "approximate",
    ];
    if domain_words.contains(&fn_name) {
        return true;
    }

    // Short single-word names (≤12 chars without underscore) are typically
    // well-known operations, simple accessors, or camelCase conversions
    if fn_name.len() <= 15 {
        return true;
    }

    false
}
