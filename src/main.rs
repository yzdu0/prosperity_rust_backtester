use indicatif::{ProgressBar, ProgressStyle};

fn main() -> anyhow::Result<()> {
    let x_values = [8, 10, 12, 14, 16];
    let y_values = [20, 25, 30];
    let z_values = [5, 10, 15, 20];
    let w_values = [1, 2, 5, 8, 10];

    let parameter_sets: Vec<_> = x_values
        .into_iter()
        .flat_map(|x| {
            y_values.into_iter().flat_map(move |y| {
                z_values
                    .into_iter()
                    .flat_map(move |z| w_values.into_iter().map(move |w| (x, y, z, w)))
            })
        })
        .collect();

    let progress_bar = ProgressBar::new(parameter_sets.len() as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {percent}% {msg}",
        )?
        .progress_chars("=>-"),
    );

    let mut results = Vec::new();

    for (x, y, z, w) in parameter_sets {
        progress_bar.set_message(format!("x={x}, y={y}, z={z}, w={w}"));
        let result = rust_backtester::cli::run(x, y, z, w)?;
        results.push((x, y, z, w, result));
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Parameter sweep complete");

    // Sort results by the fifth element (result) in descending order
    results.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    println!("Sorted results:");
    for (x, y, z, w, result) in results {
        println!("x={}, y={}, z={}, w={}, result={}", x, y, z, w, result);
    }

    Ok(())
}
