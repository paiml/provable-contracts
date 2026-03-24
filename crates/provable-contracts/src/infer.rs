//! Contract inference engine.
//!
//! Scans a crate's public API, matches against existing contract equations,
//! and suggests bindings + new contracts for unbound functions.
//!
//! Three matching strategies:
//! 1. **Name match**: function name ≈ equation name (Levenshtein + stemming)
//! 2. **Module match**: module path maps to contract tier/domain
//! 3. **Signature match**: parameter types imply domain (f32 slices → kernel)

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::reverse_coverage::{reverse_coverage, PubFn, ReverseCoverageReport};
use crate::schema::Contract;

/// A suggested binding for an unbound function.
#[derive(Debug, Clone)]
pub struct InferredBinding {
    /// The unbound function
    pub function: PubFn,
    /// Matched contract stem (e.g., "softmax-kernel-v1")
    pub contract_stem: String,
    /// Matched equation name (e.g., "softmax")
    pub equation: String,
    /// Confidence score 0.0-1.0
    pub confidence: f64,
    /// Matching strategy that produced this suggestion
    pub strategy: MatchStrategy,
}

/// A suggestion to create a new contract for a function with no match.
#[derive(Debug, Clone)]
pub struct ContractSuggestion {
    /// The unbound function
    pub function: PubFn,
    /// Suggested contract name (e.g., "maxpool-kernel-v1")
    pub suggested_name: String,
    /// Suggested tier based on module path
    pub suggested_tier: u8,
    /// Reason for the suggestion
    pub reason: String,
}

/// How a match was produced.
#[derive(Debug, Clone, Copy)]
pub enum MatchStrategy {
    /// Function name matches equation name
    NameMatch,
    /// Module path implies contract domain
    ModuleMatch,
    /// Function signature implies kernel contract
    SignatureMatch,
}

impl std::fmt::Display for MatchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NameMatch => write!(f, "name"),
            Self::ModuleMatch => write!(f, "module"),
            Self::SignatureMatch => write!(f, "signature"),
        }
    }
}

/// Result of running inference on a crate.
#[derive(Debug)]
pub struct InferResult {
    /// Functions matched to existing contracts
    pub matched: Vec<InferredBinding>,
    /// Functions needing new contracts
    pub suggestions: Vec<ContractSuggestion>,
    /// Reverse coverage report
    pub coverage: ReverseCoverageReport,
}

/// Run inference: scan crate, match against contracts, suggest bindings.
pub fn infer(
    crate_dir: &Path,
    binding_path: &Path,
    contracts: &[(String, &Contract)],
) -> InferResult {
    let coverage = reverse_coverage(crate_dir, binding_path);

    // Build equation index: equation_name → (contract_stem, equation_name)
    let mut eq_index: HashMap<String, (String, String)> = HashMap::new();
    let mut eq_keywords: Vec<(String, String, Vec<String>)> = Vec::new();

    for (stem, contract) in contracts {
        for (eq_name, eq) in &contract.equations {
            let normalized = normalize_name(eq_name);
            eq_index.insert(normalized.clone(), (stem.clone(), eq_name.clone()));

            // Extract keywords from formula + description
            let mut keywords = Vec::new();
            keywords.extend(tokenize(&eq.formula));
            if let Some(ref dom) = eq.domain {
                keywords.extend(tokenize(dom));
            }
            keywords.push(normalized);
            eq_keywords.push((stem.clone(), eq_name.clone(), keywords));
        }
    }

    // Already-bound function names (don't re-suggest)
    let bound_names: HashSet<String> = extract_bound_fn_names(binding_path);

    let mut matched = Vec::new();
    let mut suggestions = Vec::new();

    for func in &coverage.unbound {
        let fn_name = normalize_name(&func.path);

        // Skip trivial functions
        if is_trivial(&func.path) {
            continue;
        }

        // Strategy 1: Direct name match
        if let Some((stem, eq)) = eq_index.get(&fn_name) {
            if !bound_names.contains(&fn_name) {
                matched.push(InferredBinding {
                    function: func.clone(),
                    contract_stem: stem.clone(),
                    equation: eq.clone(),
                    confidence: 0.95,
                    strategy: MatchStrategy::NameMatch,
                });
                continue;
            }
        }

        // Strategy 2: Fuzzy name match (substring / prefix)
        if let Some(best) = best_fuzzy_match(&fn_name, &eq_keywords) {
            matched.push(InferredBinding {
                function: func.clone(),
                contract_stem: best.0,
                equation: best.1,
                confidence: best.2,
                strategy: MatchStrategy::NameMatch,
            });
            continue;
        }

        // Strategy 3: Module path → tier suggestion
        let tier = infer_tier_from_path(&func.file);
        let suggested_name = suggest_contract_name(&func.path);

        if !suggested_name.is_empty() {
            suggestions.push(ContractSuggestion {
                function: func.clone(),
                suggested_name,
                suggested_tier: tier,
                reason: format!("Tier {tier} function with no matching contract"),
            });
        }
    }

    // Sort by confidence descending
    matched.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    InferResult {
        matched,
        suggestions,
        coverage,
    }
}

/// Normalize a function/equation name for matching.
fn normalize_name(name: &str) -> String {
    name.to_lowercase()
        .replace("::", "_")
        .replace('-', "_")
        .replace("_v1", "")
        .replace("_kernel", "")
}

/// Tokenize a string into lowercase words.
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() > 2)
        .map(String::from)
        .collect()
}

/// Check if a function name is trivial (constructor, getter, Display impl).
fn is_trivial(name: &str) -> bool {
    let trivial = [
        "new", "default", "from", "into", "as_ref", "as_mut",
        "len", "is_empty", "clone", "fmt", "display", "debug",
        "eq", "ne", "hash", "cmp", "partial_cmp", "drop",
        "deref", "deref_mut", "index", "index_mut",
        "with_", "set_", "get_",
    ];
    let lower = name.to_lowercase();
    trivial.iter().any(|t| lower == *t || lower.starts_with(t))
}

/// Find the best fuzzy match for a function name against equation keywords.
fn best_fuzzy_match(
    fn_name: &str,
    eq_keywords: &[(String, String, Vec<String>)],
) -> Option<(String, String, f64)> {
    let mut best: Option<(String, String, f64)> = None;

    for (stem, eq_name, keywords) in eq_keywords {
        let eq_norm = normalize_name(eq_name);

        // Check if fn_name contains the equation name or vice versa
        let score = if fn_name.contains(&eq_norm) || eq_norm.contains(fn_name) {
            0.85
        } else {
            // Check keyword overlap
            let fn_tokens: HashSet<String> = fn_name
                .split('_')
                .filter(|w| w.len() > 2)
                .map(str::to_lowercase)
                .collect();
            let keyword_set: HashSet<&str> = keywords.iter().map(String::as_str).collect();

            let overlap = fn_tokens
                .iter()
                .filter(|t| keyword_set.contains(t.as_str()))
                .count();

            if overlap > 0 && !fn_tokens.is_empty() {
                #[allow(clippy::cast_precision_loss)]
                { 0.5 + 0.3 * (overlap as f64 / fn_tokens.len() as f64) }
            } else {
                0.0
            }
        };

        if score > 0.5 && !best.as_ref().is_some_and(|b| b.2 >= score) {
            best = Some((stem.clone(), eq_name.clone(), score));
        }
    }

    best
}

/// Extract already-bound function names from binding.yaml.
fn extract_bound_fn_names(binding_path: &Path) -> HashSet<String> {
    let mut names = HashSet::new();
    if let Ok(content) = std::fs::read_to_string(binding_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            let func_line = trimmed.strip_prefix("- ").unwrap_or(trimmed);
            if let Some(rest) = func_line.strip_prefix("function:") {
                let fname = rest.trim().trim_matches('"').trim();
                let short = fname.rsplit("::").next().unwrap_or(fname);
                names.insert(normalize_name(short));
            }
        }
    }
    names
}

/// Infer contract tier from file path.
fn infer_tier_from_path(path: &str) -> u8 {
    let p = path.to_lowercase();
    if p.contains("kernel") || p.contains("simd") || p.contains("avx") || p.contains("neon") {
        1 // Foundation kernel
    } else if p.contains("attention") || p.contains("transformer") {
        2 // Composite kernel
    } else if p.contains("cache") || p.contains("scheduler") || p.contains("dispatch") {
        3 // System
    } else if p.contains("train") || p.contains("optim") || p.contains("grad") {
        4 // Training
    } else if p.contains("ml") || p.contains("cluster") || p.contains("classify") {
        5 // Classical ML
    } else {
        3 // Default to system
    }
}

/// Suggest a contract name from a function name.
fn suggest_contract_name(fn_name: &str) -> String {
    let clean = fn_name
        .to_lowercase()
        .replace("::", "-")
        .replace('_', "-");
    if clean.len() < 3 || is_trivial(fn_name) {
        return String::new();
    }
    format!("{clean}-v1")
}

/// Generate a binding.yaml entry for an inferred binding.
pub fn format_binding_entry(inferred: &InferredBinding) -> String {
    format!(
        "  - contract: {}.yaml\n    equation: {}\n    module_path: ~\n    function: \"{}\"\n    status: not_implemented\n    notes: \"Auto-inferred ({}, confidence {:.0}%)\"",
        inferred.contract_stem,
        inferred.equation,
        inferred.function.path,
        inferred.strategy,
        inferred.confidence * 100.0,
    )
}

/// Generate a contract YAML stub for a new contract suggestion.
pub fn format_contract_stub(suggestion: &ContractSuggestion) -> String {
    format!(
        r#"metadata:
  version: "1.0.0"
  description: "Auto-suggested contract for {fn_name}"
  references:
    - "pv infer — auto-generated"

equations:
  {eq_name}:
    formula: "TODO: define equation"
    domain: "TODO: define domain"

proof_obligations:
  - type: invariant
    property: "TODO: define invariant"

falsification_tests:
  - id: FALSIFY-{prefix}-001
    rule: "TODO"
    prediction: "TODO"
    if_fails: "TODO"

kani_harnesses:
  - id: KANI-{prefix}-001
    obligation: "TODO"
    bound: 16
    strategy: bounded_int
    solver: cadical
"#,
        fn_name = suggestion.function.path,
        eq_name = suggestion.function.path.to_lowercase().replace("::", "_"),
        prefix = suggestion.function.path.to_uppercase().chars().take(3).collect::<String>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("softmax_kernel_v1"), "softmax");
        assert_eq!(normalize_name("RMSNorm"), "rmsnorm");
        assert_eq!(normalize_name("ssm-scan"), "ssm_scan");
    }

    #[test]
    fn test_is_trivial() {
        assert!(is_trivial("new"));
        assert!(is_trivial("default"));
        assert!(is_trivial("with_capacity"));
        assert!(is_trivial("set_threshold"));
        assert!(!is_trivial("softmax"));
        assert!(!is_trivial("rmsnorm"));
    }

    #[test]
    fn test_infer_tier() {
        assert_eq!(infer_tier_from_path("src/kernels/softmax.rs"), 1);
        assert_eq!(infer_tier_from_path("src/nn/transformer/attention.rs"), 2);
        assert_eq!(infer_tier_from_path("src/scheduler/mod.rs"), 3);
        assert_eq!(infer_tier_from_path("src/train/optimizer.rs"), 4);
    }

    #[test]
    fn test_suggest_contract_name() {
        assert_eq!(suggest_contract_name("softmax"), "softmax-v1");
        assert_eq!(suggest_contract_name("batch_norm"), "batch-norm-v1");
        assert_eq!(suggest_contract_name("new"), ""); // trivial
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("exp(x_i - max(x)) / sum");
        assert!(tokens.contains(&"exp".to_string()));
        assert!(tokens.contains(&"max".to_string()));
        assert!(tokens.contains(&"sum".to_string()));
    }

    #[test]
    fn test_fuzzy_match() {
        let eq_keywords = vec![
            ("softmax-kernel-v1".into(), "softmax".into(), vec!["softmax".into(), "exp".into(), "sum".into()]),
            ("rmsnorm-kernel-v1".into(), "rmsnorm".into(), vec!["rmsnorm".into(), "sqrt".into(), "mean".into()]),
        ];

        // Direct substring match
        let result = best_fuzzy_match("softmax_avx2", &eq_keywords);
        assert!(result.is_some());
        let (stem, _, conf) = result.unwrap();
        assert_eq!(stem, "softmax-kernel-v1");
        assert!(conf > 0.8);

        // No match
        let result = best_fuzzy_match("parse_config", &eq_keywords);
        assert!(result.is_none());
    }
}
