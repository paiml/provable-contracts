//! `pv score` — Quantitative contract and codebase scoring.

use std::path::Path;

use std::collections::HashSet;

use provable_contracts::binding::BindingRegistry;
use provable_contracts::query::ContractIndex;
use provable_contracts::schema::parse_contract;
use provable_contracts::scoring;
use provable_contracts::scoring::drift;
use provable_contracts::scoring::{ContractScore, ScoringWeights};

pub fn run(
    path: &Path,
    binding: Option<&Path>,
    format: &str,
    min_score: Option<f64>,
    summary: bool,
    top_gaps: usize,
    weights_json: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding_registry = binding
        .map(|p| {
            let content = std::fs::read_to_string(p)?;
            let reg: BindingRegistry = serde_yaml::from_str(&content)?;
            Ok::<_, Box<dyn std::error::Error>>(reg)
        })
        .transpose()?;

    let weights = match weights_json {
        Some(json) => serde_json::from_str::<ScoringWeights>(json)?,
        None => ScoringWeights::default(),
    };

    if path.is_dir() {
        run_directory(
            path,
            binding_registry.as_ref(),
            binding,
            format,
            min_score,
            summary,
            top_gaps,
            &weights,
        )
    } else {
        run_single(path, binding_registry.as_ref(), format, min_score, &weights)
    }
}

fn run_single(
    path: &Path,
    binding: Option<&BindingRegistry>,
    format: &str,
    min_score: Option<f64>,
    weights: &ScoringWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let score = scoring::score_contract_weighted(&contract, binding, stem, weights);

    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&score)?),
        "markdown" => print!("{}", score_to_markdown(&score)),
        _ => print!("{score}"),
    }

    if let Some(threshold) = min_score {
        if score.composite < threshold {
            return Err(format!(
                "Score {:.2} below threshold {threshold:.2}",
                score.composite,
            )
            .into());
        }
    }

    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
fn run_directory(
    dir: &Path,
    binding: Option<&BindingRegistry>,
    binding_path: Option<&Path>,
    format: &str,
    min_score: Option<f64>,
    summary: bool,
    top_gaps: usize,
    weights: &ScoringWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("yaml"))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::path);

    let mut scores = Vec::new();
    for entry in &entries {
        let path = entry.path();
        let Ok(contract) = parse_contract(&path) else {
            continue;
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        scores.push(scoring::score_contract_weighted(
            &contract, binding, stem, weights,
        ));
    }

    let mean: f64 = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64
    };

    if summary {
        print_summary_only(&scores, mean, format)?;
    } else {
        print_directory_scores(&scores, mean, format)?;
    }

    // Show top gaps by lowest score
    if top_gaps > 0 && !scores.is_empty() {
        print_top_gaps(&scores, top_gaps, format);
    }

    // Codebase scoring if binding is provided
    if let Some(binding) = binding {
        let mut parsed = Vec::new();
        for entry in &entries {
            let path = entry.path();
            let Ok(contract) = parse_contract(&path) else {
                continue;
            };
            // Binding uses filename WITH .yaml extension as stem
            let stem = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            parsed.push((stem, contract));
        }
        let refs: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();

        // Build pagerank from contract index for impact-weighted gap analysis
        let pagerank = ContractIndex::from_directory(dir).ok().map(|idx| {
            idx.entries
                .iter()
                .filter_map(|e| idx.cached_pagerank(&e.stem).map(|s| (e.stem.clone(), s)))
                .collect::<std::collections::HashMap<String, f64>>()
        });

        // CD5: Detect stale contracts via git timestamps
        let drift_score = binding_path.map(|bp| {
            let bound_stems: HashSet<&str> = binding
                .bindings
                .iter()
                .map(|b| b.contract.as_str())
                .collect();
            let stale = drift::detect_stale_contracts(dir, bp, &bound_stems);
            drift::compute_drift(stale.len(), bound_stems.len())
        });

        let codebase = scoring::score_codebase_full(&refs, binding, pagerank.as_ref(), drift_score);

        match format {
            "json" => {
                let output = serde_json::json!({ "codebase": codebase });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            "markdown" => {
                println!("\n## Codebase Score\n");
                println!("| Dimension | Value |");
                println!("|-----------|-------|");
                println!("| Coverage | {:.0}% |", codebase.contract_coverage * 100.0);
                println!(
                    "| Binding | {:.0}% |",
                    codebase.binding_completeness * 100.0
                );
                println!("| Mean Score | {:.2} |", codebase.mean_contract_score);
                println!("| Proof Depth | {:.2} |", codebase.proof_depth_dist);
                println!("| Drift | {:.2} |", codebase.drift);
                println!(
                    "\n**Composite:** {:.2} (Grade {})",
                    codebase.composite, codebase.grade
                );
            }
            _ => println!("\n{codebase}"),
        }
    }

    if let Some(threshold) = min_score {
        if mean < threshold {
            return Err(format!("Mean score {mean:.2} below threshold {threshold:.2}",).into());
        }
    }

    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn print_directory_scores(
    scores: &[ContractScore],
    mean: f64,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        "json" => {
            let output = serde_json::json!({
                "contracts": scores.len(),
                "mean_score": mean,
                "mean_grade": scoring::Grade::from_score(mean).to_string(),
                "scores": scores,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "markdown" => {
            println!("## Contract Scores\n");
            println!("| Contract | Score | Grade | Spec | Falsify | Kani | Lean | Bind |");
            println!("|----------|-------|-------|------|---------|------|------|------|");
            for s in scores {
                println!(
                    "| {} | {:.2} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                    s.stem,
                    s.composite,
                    s.grade,
                    s.spec_depth,
                    s.falsification_coverage,
                    s.kani_coverage,
                    s.lean_coverage,
                    s.binding_coverage
                );
            }
            println!(
                "\n**{} contracts** — Mean: {mean:.2} (Grade {})",
                scores.len(),
                scoring::Grade::from_score(mean)
            );
        }
        _ => {
            for s in scores {
                print!("{s}");
            }
            println!(
                "\n{} contracts — Mean: {mean:.2} (Grade {})",
                scores.len(),
                scoring::Grade::from_score(mean)
            );
        }
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn print_summary_only(
    scores: &[ContractScore],
    mean: f64,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let grade = scoring::Grade::from_score(mean);
    match format {
        "json" => {
            let output = serde_json::json!({
                "contracts": scores.len(),
                "mean_score": mean,
                "mean_grade": grade.to_string(),
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "markdown" => {
            println!(
                "**{} contracts** — Mean: {mean:.2} (Grade {grade})",
                scores.len()
            );
        }
        _ => {
            println!(
                "{} contracts — Mean: {mean:.2} (Grade {grade})",
                scores.len()
            );
        }
    }
    Ok(())
}

fn print_top_gaps(scores: &[ContractScore], n: usize, format: &str) {
    let mut sorted: Vec<_> = scores.iter().collect();
    sorted.sort_by(|a, b| {
        a.composite
            .partial_cmp(&b.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<_> = sorted.into_iter().take(n).collect();

    match format {
        "json" => {} // Already in JSON output
        "markdown" => {
            println!("\n### Top {} Gaps\n", top.len());
            for s in &top {
                println!("- **{}** — {:.2} ({})", s.stem, s.composite, s.grade);
            }
        }
        _ => {
            println!("\nTop {} gaps:", top.len());
            for s in &top {
                println!("  {} — {:.2} ({})", s.stem, s.composite, s.grade);
            }
        }
    }
}

fn score_to_markdown(score: &ContractScore) -> String {
    format!(
        "### {}\n\n- **Score:** {:.2} (Grade {})\n- Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}\n",
        score.stem,
        score.composite,
        score.grade,
        score.spec_depth,
        score.falsification_coverage,
        score.kani_coverage,
        score.lean_coverage,
        score.binding_coverage
    )
}
