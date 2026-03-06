//! `pv score` — Quantitative contract and codebase scoring.

use std::path::Path;

use provable_contracts::binding::BindingRegistry;
use provable_contracts::schema::parse_contract;
use provable_contracts::scoring;

pub fn run(
    path: &Path,
    binding: Option<&Path>,
    format: &str,
    min_score: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding_registry = binding
        .map(|p| {
            let content = std::fs::read_to_string(p)?;
            let reg: BindingRegistry = serde_yaml::from_str(&content)?;
            Ok::<_, Box<dyn std::error::Error>>(reg)
        })
        .transpose()?;

    if path.is_dir() {
        run_directory(path, binding_registry.as_ref(), format, min_score)
    } else {
        run_single(path, binding_registry.as_ref(), format, min_score)
    }
}

fn run_single(
    path: &Path,
    binding: Option<&BindingRegistry>,
    format: &str,
    min_score: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let score = scoring::score_contract(&contract, binding, stem);

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&score)?);
    } else {
        print!("{score}");
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

#[allow(clippy::cast_precision_loss)]
fn run_directory(
    dir: &Path,
    binding: Option<&BindingRegistry>,
    format: &str,
    min_score: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                == Some("yaml")
        })
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
        scores.push(scoring::score_contract(&contract, binding, stem));
    }

    let mean: f64 = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64
    };

    if format == "json" {
        let output = serde_json::json!({
            "contracts": scores.len(),
            "mean_score": mean,
            "mean_grade": scoring::Grade::from_score(mean).to_string(),
            "scores": scores,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        for s in &scores {
            print!("{s}");
        }
        println!(
            "\n{} contracts — Mean: {mean:.2} (Grade {})",
            scores.len(),
            scoring::Grade::from_score(mean)
        );
    }

    if let Some(threshold) = min_score {
        if mean < threshold {
            return Err(format!(
                "Mean score {mean:.2} below threshold {threshold:.2}",
            )
            .into());
        }
    }

    Ok(())
}
