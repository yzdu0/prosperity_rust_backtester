fn main() -> anyhow::Result<()> {
    for x in (10..=20).step_by(3) {
        for y in (30..=50).step_by(3) {
            for z in (20..=30).step_by(3) {

                println!("Running backtester with parameters: x={}, y={}, z={}", x, y, z);
                rust_backtester::cli::run(x, y, z);
            }
        }
    }

    Ok(())
}
