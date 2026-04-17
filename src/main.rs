fn main() -> anyhow::Result<()> {
    let mut results = Vec::new();

    for x in (8..16).step_by(2) {
        for y in (20..=50).step_by(5) {
            for z in (10..=40).step_by(10) {
                for w in (2..=10).step_by(2) {
                    println!("Running backtester with parameters: x={}, y={}, z={}, w={}", x, y, z, z);
                    let result = rust_backtester::cli::run(x, y, z, w)?;
                    results.push((x, y, z, w, result));
                }
            }
        }
    }

    // Sort results by the fifth element (result) in descending order
    results.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    println!("Sorted results:");
    for (x, y, z, w, result) in results {
        println!("x={}, y={}, z={}, w={}, result={}", x, y, z, w, result);
    }

    Ok(())
}
