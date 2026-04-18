use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use rust_backtester::model::TraderGlobals;
use serde_json::{Value, json};

macro_rules! values {
    ($($value:expr),* $(,)?) => {
        vec![$(json!($value)),*]
    };
}

fn main() -> anyhow::Result<()> {
    let sweep_parameters = vec![
        ("OSMIUM_CLIP", values![8, 10, 12, 14, 16]),
        ("SNIPE_POSITION_LIMIT", values![20, 25, 30]),
        ("WINDOW_SIZE", values![5, 10, 15, 20]),
        ("DEVIATION_MULTIPLIER", values![1, 2, 5, 8, 10]),
    ];
    let parameter_sets = build_parameter_sets(&sweep_parameters);

    let progress_bar = ProgressBar::new(parameter_sets.len() as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {percent}% {msg}",
        )?
        .progress_chars("=>-"),
    );

    let mut results = Vec::new();

    for parameter_set in parameter_sets {
        progress_bar.set_message(format_parameter_set(&parameter_set));
        let result = rust_backtester::cli::run(&parameter_set)?;
        results.push((parameter_set, result));
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Parameter sweep complete");

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Sorted results:");
    for (parameter_set, result) in results {
        println!("{}, result={}", format_parameter_set(&parameter_set), result);
    }

    Ok(())
}

fn build_parameter_sets(parameters: &[(&str, Vec<Value>)]) -> Vec<TraderGlobals> {
    let mut parameter_sets = vec![IndexMap::new()];

    for (name, values) in parameters {
        let mut expanded_sets = Vec::with_capacity(parameter_sets.len() * values.len());

        for parameter_set in parameter_sets {
            for value in values {
                let mut next_parameter_set = parameter_set.clone();
                next_parameter_set.insert((*name).to_string(), value.clone());
                expanded_sets.push(next_parameter_set);
            }
        }

        parameter_sets = expanded_sets;
    }

    parameter_sets
}

fn format_parameter_set(parameter_set: &TraderGlobals) -> String {
    parameter_set
        .iter()
        .map(|(name, value)| format!("{name}={}", format_value(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_value(value: &Value) -> String {
    match value {
        Value::String(inner) => inner.clone(),
        _ => value.to_string(),
    }
}
