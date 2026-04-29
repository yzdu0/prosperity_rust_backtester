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
    /*
    HYDROGEL_FAIR_VALUE = 10000.0
    HYDROGEL_SNIPE_EDGE = 9
    HYDROGEL_MAX_TAKE_SIZE = 9
    HYDROGEL_PASSIVE_ORDER_SIZE = 4
    HYDROGEL_PASSIVE_RESERVE = 12
    HYDROGEL_PASSIVE_EDGE_MULTIPLIER = 20
    HYDROGEL_INVENTORY_SKEW = 3.0


    VELVET_WINDOW_SIZE = 5
    VELVET_BLEND_PCT = 12
    VELVET_SNIPE_EDGE = 12
    VELVET_MAX_TAKE_SIZE = 60

     HYDROGEL_SNIPE_EDGE=9, HYDROGEL_MAX_TAKE_SIZE=12, HYDROGEL_PASSIVE_ORDER_SIZE=8


    VELVET_PASSIVE_ORDER_SIZE = 24
    VELVET_PASSIVE_EDGE_MULTIPLIER = 35
    VELVET_PASSIVE_RESERVE = 20
    VELVET_INVENTORY_SKEW = 5.0
        */
    let sweep_parameters = vec![
        /*("HYDROGEL_SNIPE_EDGE", values![3, 6, 9, 12, 15, 20]),
        ("HYDROGEL_MAX_TAKE_SIZE", values![3, 5, 8, 12, 20, 30]),
        ("HYDROGEL_PASSIVE_ORDER_SIZE", values![2, 4, 6, 8, 12, 20]),*/
        /*("HYDROGEL_PASSIVE_RESERVE", values![6, 12, 18, 24, 36, 48]),
        ("HYDROGEL_PASSIVE_EDGE_MULTIPLIER", values![10, 20, 30, 50, 100]),
        ("HYDROGEL_INVENTORY_SKEW", values![1, 3, 5]),*/

        /*("VELVET_WINDOW_SIZE", values![5, 10, 100]),
        ("VELVET_BLEND_PCT", values![0, 12, 25, 50]),
        ("VELVET_SNIPE_EDGE", values![6, 12, 15, 20]),
        ("VELVET_MAX_TAKE_SIZE", values![5, 8, 12, 20, 30]),*/
        //("VELVET_PASSIVE_ORDER_SIZE", values![24]),
        /*(
            "VELVET_PASSIVE_EDGE_MULTIPLIER",
            values![10, 20, 35, 50, 100],
        ),
        ("VELVET_PASSIVE_RESERVE", values![10, 20, 30, 40]),
        ("VELVET_INVENTORY_SKEW", values![1.0, 3.0, 5.0]),*/
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
        println!(
            "{}, result={}",
            format_parameter_set(&parameter_set),
            result
        );
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
