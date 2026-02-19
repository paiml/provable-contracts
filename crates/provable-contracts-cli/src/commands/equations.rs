use std::path::Path;

use provable_contracts::schema::{parse_contract, Equation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Latex,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "text" => Ok(Self::Text),
            "latex" => Ok(Self::Latex),
            other => Err(format!(
                "unknown format '{other}', expected 'text' or 'latex'"
            )),
        }
    }
}

pub fn run(path: &Path, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    match format {
        OutputFormat::Text => render_text(name, &contract.equations),
        OutputFormat::Latex => render_latex(name, &contract.equations),
    }

    Ok(())
}

fn render_text(name: &str, equations: &std::collections::BTreeMap<String, Equation>) {
    println!("Equations for {name}");
    println!("{}", "=".repeat(40 + name.len()));
    println!();

    for (id, eq) in equations {
        println!("  {id}");
        println!("    formula:  {}", eq.formula);
        if let Some(ref dom) = eq.domain {
            println!("    domain:   {dom}");
        }
        if let Some(ref cod) = eq.codomain {
            println!("    codomain: {cod}");
        }
        if !eq.invariants.is_empty() {
            println!("    invariants:");
            for inv in &eq.invariants {
                println!("      - {inv}");
            }
        }
        println!();
    }
}

fn render_latex(name: &str, equations: &std::collections::BTreeMap<String, Equation>) {
    let escaped_name = latex_escape(name);
    println!("% Equations for {name}");
    println!("\\section{{Equations: {escaped_name}}}");
    println!();

    for (id, eq) in equations {
        let escaped_id = latex_escape(id);
        let latex_formula = math_to_latex(&eq.formula);

        println!("\\subsection{{{escaped_id}}}");
        println!("\\label{{eq:{id}}}");
        println!();
        println!("\\begin{{equation}}");
        println!("  {latex_formula}");
        println!("\\end{{equation}}");
        println!();

        if let Some(ref dom) = eq.domain {
            println!(
                "\\textbf{{Domain:}} ${}$",
                math_to_latex(dom)
            );
            println!();
        }

        if let Some(ref cod) = eq.codomain {
            println!(
                "\\textbf{{Codomain:}} ${}$",
                math_to_latex(cod)
            );
            println!();
        }

        if !eq.invariants.is_empty() {
            println!("\\textbf{{Invariants:}}");
            println!("\\begin{{itemize}}");
            for inv in &eq.invariants {
                println!("  \\item ${}$", math_to_latex(inv));
            }
            println!("\\end{{itemize}}");
            println!();
        }
    }
}

/// Escape special LaTeX characters in plain text.
fn latex_escape(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('~', "\\textasciitilde{}")
        .replace('^', "\\textasciicircum{}")
}

/// Convert contract math notation to LaTeX math mode.
///
/// Handles common patterns found in our YAML contracts:
/// - Greek letters (ε, σ, α, etc.)
/// - Subscripts (`x_i`, `A_{ij}`)
/// - Superscripts (x^T, ℝ^n)
/// - Operators (Σ, ∈, ≈, ≤, ≥, ∀, →)
/// - Special sets (ℝ, ℤ)
/// - Functions (sqrt, exp, log, softmax, etc.)
fn math_to_latex(s: &str) -> String {
    let mut out = s.to_string();

    // Unicode → LaTeX replacements. Each entry: (unicode, latex_cmd, is_command).
    // When is_command is true, a trailing space is inserted before the next
    // alphabetic character to prevent `\foralli` instead of `\forall i`.
    let replacements: &[(&str, &str, bool)] = &[
        // Greek letters
        ("α", "\\alpha", true),
        ("β", "\\beta", true),
        ("γ", "\\gamma", true),
        ("δ", "\\delta", true),
        ("ε", "\\varepsilon", true),
        ("θ", "\\theta", true),
        ("λ", "\\lambda", true),
        ("σ", "\\sigma", true),
        ("τ", "\\tau", true),
        ("Σ", "\\sum", true),
        ("Φ", "\\Phi", true),
        ("π", "\\pi", true),
        // Operators
        ("∈", "\\in", true),
        ("∉", "\\notin", true),
        ("≈", "\\approx", true),
        ("≤", "\\leq", true),
        ("≥", "\\geq", true),
        ("≠", "\\neq", true),
        ("∀", "\\forall", true),
        ("∃", "\\exists", true),
        ("→", "\\to", true),
        ("←", "\\leftarrow", true),
        ("⊗", "\\otimes", true),
        ("⁺", "^{+}", false),
        // Special sets
        ("ℝ", "\\mathbb{R}", false),
        ("ℤ", "\\mathbb{Z}", false),
    ];
    for &(uni, tex, is_cmd) in replacements {
        if is_cmd {
            out = replace_unicode_cmd(&out, uni, tex);
        } else {
            out = out.replace(uni, tex);
        }
    }

    // sqrt(...) → \sqrt{...}
    out = replace_func(&out, "sqrt", "\\sqrt");

    // exp(...) → \exp(...)
    out = out.replace("exp(", "\\exp(");

    // log(...) → \log(...)
    out = out.replace("log(", "\\log(");

    // Escape underscores that aren't already subscripts
    // But preserve x_i, x_{...} patterns
    // This is tricky — leave underscores as-is since they're valid LaTeX subscripts

    // Escape % and # in formulas
    out = out.replace('%', "\\%");
    out = out.replace('#', "\\#");

    out
}

/// Replace a Unicode symbol with a LaTeX command, inserting a trailing
/// space when the next character is alphabetic (prevents `\foralli`).
fn replace_unicode_cmd(s: &str, uni: &str, tex: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find(uni) {
        result.push_str(&rest[..pos]);
        result.push_str(tex);
        let after = &rest[pos + uni.len()..];
        // Insert space before next alphabetic char to keep LaTeX happy
        if after.starts_with(|c: char| c.is_ascii_alphabetic()) {
            result.push(' ');
        }
        rest = after;
    }
    result.push_str(rest);
    result
}

/// Replace `func(...)` with `\cmd{...}` handling nested parens.
/// Applies recursively so `sqrt(a + sqrt(b))` becomes `\sqrt{a + \sqrt{b}}`.
fn replace_func(s: &str, func: &str, cmd: &str) -> String {
    let pattern = format!("{func}(");
    let mut result = String::with_capacity(s.len());
    let mut rest = s;

    while let Some(pos) = rest.find(&pattern) {
        result.push_str(&rest[..pos]);
        let after = &rest[pos + pattern.len()..];

        // Find matching closing paren
        let mut depth = 1;
        let mut end = 0;
        for (i, ch) in after.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if depth == 0 {
            let inner = &after[..end];
            // Recurse to handle nested calls like sqrt(a + sqrt(b))
            let inner_replaced = replace_func(inner, func, cmd);
            result.push_str(&format!("{cmd}{{{inner_replaced}}}"));
            rest = &after[end + 1..];
        } else {
            // Unmatched paren — emit as-is
            result.push_str(&pattern);
            rest = after;
        }
    }

    result.push_str(rest);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_to_latex_greek() {
        assert_eq!(math_to_latex("ε > 0"), "\\varepsilon > 0");
        assert_eq!(math_to_latex("α_t"), "\\alpha_t");
    }

    #[test]
    fn test_math_to_latex_operators() {
        assert_eq!(math_to_latex("x ∈ ℝ^n"), "x \\in \\mathbb{R}^n");
        assert_eq!(math_to_latex("a ≈ b"), "a \\approx b");
        assert_eq!(math_to_latex("∀i: x_i ≥ 0"), "\\forall i: x_i \\geq 0");
    }

    #[test]
    fn test_math_to_latex_sqrt() {
        assert_eq!(
            math_to_latex("Q / sqrt(mean(Q²) + ε)"),
            "Q / \\sqrt{mean(Q²) + \\varepsilon}"
        );
    }

    #[test]
    fn test_math_to_latex_exp() {
        assert_eq!(
            math_to_latex("exp(x_i - max(x))"),
            "\\exp(x_i - max(x))"
        );
    }

    #[test]
    fn test_replace_func_nested() {
        assert_eq!(
            replace_func("sqrt(a + sqrt(b))", "sqrt", "\\sqrt"),
            "\\sqrt{a + \\sqrt{b}}"
        );
    }

    #[test]
    fn test_latex_escape() {
        assert_eq!(latex_escape("a_b"), "a\\_b");
        assert_eq!(latex_escape("100%"), "100\\%");
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(OutputFormat::from_str("text").unwrap(), OutputFormat::Text);
        assert_eq!(OutputFormat::from_str("latex").unwrap(), OutputFormat::Latex);
        assert!(OutputFormat::from_str("json").is_err());
    }
}
