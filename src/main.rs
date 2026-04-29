use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use rust_backtester::model::TraderGlobals;
use serde_json::Value;

#[allow(unused_macros)]
macro_rules! values {
    ($($value:expr),* $(,)?) => {
        vec![$(serde_json::json!($value)),*]
    };
}

type SearchPoint = Vec<usize>;

#[derive(Debug, Clone)]
struct SearchDimension {
    name: String,
    values: Vec<Value>,
}

#[derive(Debug, Clone)]
struct SearchSpace {
    dimensions: Vec<SearchDimension>,
    total_combinations: u128,
}

impl SearchSpace {
    fn new(parameters: &[(&str, Vec<Value>)]) -> Result<Self> {
        let mut dimensions = Vec::with_capacity(parameters.len());
        let mut total_combinations = 1u128;

        for (name, values) in parameters {
            if values.is_empty() {
                bail!("parameter {name} has no candidate values");
            }

            total_combinations = total_combinations
                .checked_mul(values.len() as u128)
                .unwrap_or(u128::MAX);

            dimensions.push(SearchDimension {
                name: (*name).to_string(),
                values: values.clone(),
            });
        }

        Ok(Self {
            dimensions,
            total_combinations,
        })
    }

    fn max_unique_points(&self) -> usize {
        if self.dimensions.is_empty() {
            1
        } else {
            self.total_combinations.min(usize::MAX as u128) as usize
        }
    }

    fn materialize(&self, point: &[usize]) -> TraderGlobals {
        debug_assert_eq!(point.len(), self.dimensions.len());

        let mut parameter_set = IndexMap::with_capacity(self.dimensions.len());

        for (dimension, value_index) in self.dimensions.iter().zip(point.iter().copied()) {
            if let Some(value) = dimension.values.get(value_index) {
                parameter_set.insert(dimension.name.clone(), value.clone());
            }
        }

        parameter_set
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchConfig {
    evaluation_budget: usize,
    population_size: usize,
    elite_count: usize,
    tournament_size: usize,
    mutation_rate: f64,
    max_stagnant_generations: usize,
    seed: u64,
}

impl SearchConfig {
    fn for_space(search_space: &SearchSpace) -> Self {
        let dimension_count = search_space.dimensions.len().max(1);
        let unique_points = search_space.max_unique_points().max(1);
        let population_target = (dimension_count * 4).clamp(12, 24);
        let population_size = population_target.min(unique_points);
        let elite_count = population_size.div_ceil(5).clamp(1, population_size);
        let tournament_size = if population_size == 1 {
            1
        } else {
            population_size.min(4)
        };
        let generation_count = (dimension_count * 4).clamp(12, 24);
        let evaluation_budget = unique_points.min(population_size.saturating_mul(generation_count));
        let mutation_rate = (1.0 / dimension_count as f64).clamp(0.18, 0.40);
        let max_stagnant_generations = (dimension_count * 2).clamp(4, 10);

        Self {
            evaluation_budget,
            population_size,
            elite_count,
            tournament_size,
            mutation_rate,
            max_stagnant_generations,
            seed: default_search_seed(),
        }
    }
}

#[derive(Debug, Clone)]
struct ScoredCandidate {
    point: SearchPoint,
    score: f64,
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn seeded(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn gen_range(&mut self, upper_bound: usize) -> usize {
        if upper_bound <= 1 {
            0
        } else {
            (self.next_u64() as usize) % upper_bound
        }
    }

    fn gen_bool(&mut self, probability: f64) -> bool {
        self.next_f64() < probability
    }
}

fn main() -> Result<()> {
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

        //("ROLLING_WINDOW_SIZE", values![40, 50, 60, 70, 80, 100]),
        //("SNIPE_EDGE", values![20, 25, 30, 35, 40, 50, 70, 100])
        //("ROLLING_WINDOW_SIZE", values![5, 10, 20, 35, 50, 100]),
        //("WINDOW", values![100, 500, 1000, 1500, 2000, 2500, 3000, 5000]),
        //("Z_ENTER", values![-200, -100, -80, -50, -30, -25, -10, 0, 10, 25, 50, 80, 100, 200]),
        //("Z_PASSIVE", values![-200, -100, -50, -25, 0, 25, 50, 100,200])
    ];
    let search_space = SearchSpace::new(&sweep_parameters)?;
    let search_config = SearchConfig::for_space(&search_space);

    println!(
        "Running genetic parameter search over {} combinations with a budget of {} evaluations (population {}, elites {}, mutation {:.0}%, seed {}).",
        search_space.total_combinations,
        search_config.evaluation_budget,
        search_config.population_size,
        search_config.elite_count,
        search_config.mutation_rate * 100.0,
        search_config.seed,
    );

    let progress_bar = ProgressBar::new(search_config.evaluation_budget as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {percent}% {msg}",
        )?
        .progress_chars("=>-"),
    );

    let mut evaluate = |parameter_set: &TraderGlobals| -> Result<f64> {
        progress_bar.set_message(format_parameter_set(parameter_set));
        let result = rust_backtester::cli::run(parameter_set)?;
        progress_bar.inc(1);
        Ok(result)
    };

    let mut results = evolutionary_search(&search_space, search_config, &mut evaluate)?;

    progress_bar.finish_with_message("Parameter search complete");

    results.sort_by(|a, b| b.1.total_cmp(&a.1));

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

fn evolutionary_search<F>(
    search_space: &SearchSpace,
    search_config: SearchConfig,
    evaluator: &mut F,
) -> Result<Vec<(TraderGlobals, f64)>>
where
    F: FnMut(&TraderGlobals) -> Result<f64>,
{
    let max_unique_points = search_space.max_unique_points();
    let evaluation_budget = search_config
        .evaluation_budget
        .min(max_unique_points)
        .max(1);
    let mut rng = SimpleRng::seeded(search_config.seed);
    let mut score_cache = HashMap::new();
    let mut all_results = Vec::new();

    let initial_points = initial_population(search_space, search_config.population_size, &mut rng);
    let mut population = evaluate_population(
        initial_points,
        search_space,
        evaluator,
        &mut score_cache,
        &mut all_results,
    )?;
    population.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut best_score = population.first().map(|candidate| candidate.score);
    let mut stagnant_generations = 0usize;

    while score_cache.len() < evaluation_budget {
        let next_points = if stagnant_generations >= search_config.max_stagnant_generations {
            stagnant_generations = 0;
            reseed_population(
                search_space,
                search_config,
                &population,
                &score_cache,
                &mut rng,
            )
        } else {
            breed_next_generation(
                search_space,
                search_config,
                &population,
                &score_cache,
                &mut rng,
            )
        };

        if next_points.is_empty() {
            break;
        }

        let evaluated_count_before = score_cache.len();
        let mut next_population = evaluate_population(
            next_points,
            search_space,
            evaluator,
            &mut score_cache,
            &mut all_results,
        )?;
        next_population.sort_by(|a, b| b.score.total_cmp(&a.score));

        if next_population.is_empty() {
            break;
        }

        let generation_best = next_population.first().map(|candidate| candidate.score);
        let found_new_points = score_cache.len() > evaluated_count_before;

        if let Some(score) = generation_best {
            match best_score {
                Some(best) if score > best => {
                    best_score = Some(score);
                    stagnant_generations = 0;
                }
                Some(_) if found_new_points => {
                    stagnant_generations += 1;
                }
                Some(_) => {
                    stagnant_generations =
                        stagnant_generations.saturating_add(search_config.max_stagnant_generations);
                }
                None => {
                    best_score = Some(score);
                    stagnant_generations = 0;
                }
            }
        }

        population = next_population;

        if score_cache.len() >= max_unique_points {
            break;
        }
    }

    refine_best_candidate(
        search_space,
        evaluation_budget,
        evaluator,
        &mut score_cache,
        &mut all_results,
    )?;

    all_results.sort_by(|a, b| b.score.total_cmp(&a.score));

    Ok(all_results
        .into_iter()
        .map(|candidate| (search_space.materialize(&candidate.point), candidate.score))
        .collect())
}

fn evaluate_population<F>(
    points: Vec<SearchPoint>,
    search_space: &SearchSpace,
    evaluator: &mut F,
    score_cache: &mut HashMap<SearchPoint, f64>,
    all_results: &mut Vec<ScoredCandidate>,
) -> Result<Vec<ScoredCandidate>>
where
    F: FnMut(&TraderGlobals) -> Result<f64>,
{
    let mut population = Vec::with_capacity(points.len());
    let mut unique_points = HashSet::new();

    for point in points {
        if !unique_points.insert(point.clone()) {
            continue;
        }

        population.push(score_candidate(
            point,
            search_space,
            evaluator,
            score_cache,
            all_results,
        )?);
    }

    Ok(population)
}

fn score_candidate<F>(
    point: SearchPoint,
    search_space: &SearchSpace,
    evaluator: &mut F,
    score_cache: &mut HashMap<SearchPoint, f64>,
    all_results: &mut Vec<ScoredCandidate>,
) -> Result<ScoredCandidate>
where
    F: FnMut(&TraderGlobals) -> Result<f64>,
{
    if let Some(score) = score_cache.get(&point).copied() {
        return Ok(ScoredCandidate { point, score });
    }

    let parameter_set = search_space.materialize(&point);
    let score = evaluator(&parameter_set).with_context(|| {
        format!(
            "failed while evaluating {}",
            format_parameter_set(&parameter_set)
        )
    })?;
    score_cache.insert(point.clone(), score);
    all_results.push(ScoredCandidate {
        point: point.clone(),
        score,
    });

    Ok(ScoredCandidate { point, score })
}

fn refine_best_candidate<F>(
    search_space: &SearchSpace,
    evaluation_budget: usize,
    evaluator: &mut F,
    score_cache: &mut HashMap<SearchPoint, f64>,
    all_results: &mut Vec<ScoredCandidate>,
) -> Result<()>
where
    F: FnMut(&TraderGlobals) -> Result<f64>,
{
    let Some(mut current_best) = all_results
        .iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .cloned()
    else {
        return Ok(());
    };

    loop {
        if score_cache.len() >= evaluation_budget {
            break;
        }

        let mut best_neighbor = None;

        for (dimension_index, dimension) in search_space.dimensions.iter().enumerate() {
            for value_index in 0..dimension.values.len() {
                if value_index == current_best.point[dimension_index] {
                    continue;
                }

                if score_cache.len() >= evaluation_budget {
                    break;
                }

                let mut neighbor = current_best.point.clone();
                neighbor[dimension_index] = value_index;

                let scored =
                    score_candidate(neighbor, search_space, evaluator, score_cache, all_results)?;

                if scored.score > current_best.score
                    && best_neighbor
                        .as_ref()
                        .is_none_or(|best: &ScoredCandidate| scored.score > best.score)
                {
                    best_neighbor = Some(scored);
                }
            }
        }

        match best_neighbor {
            Some(neighbor) => current_best = neighbor,
            None => break,
        }
    }

    Ok(())
}

fn initial_population(
    search_space: &SearchSpace,
    population_size: usize,
    rng: &mut SimpleRng,
) -> Vec<SearchPoint> {
    let target_size = population_size.min(search_space.max_unique_points());
    let mut points = Vec::with_capacity(target_size);
    let mut seen = HashSet::new();

    for seed_point in [
        boundary_point(search_space, Boundary::Low),
        boundary_point(search_space, Boundary::Middle),
        boundary_point(search_space, Boundary::High),
    ] {
        if seen.insert(seed_point.clone()) {
            points.push(seed_point);
        }
    }

    while points.len() < target_size {
        let point = random_point(search_space, rng);
        if seen.insert(point.clone()) {
            points.push(point);
        }
    }

    points
}

fn breed_next_generation(
    search_space: &SearchSpace,
    search_config: SearchConfig,
    population: &[ScoredCandidate],
    score_cache: &HashMap<SearchPoint, f64>,
    rng: &mut SimpleRng,
) -> Vec<SearchPoint> {
    let target_size = search_config
        .population_size
        .min(search_space.max_unique_points());
    let mut next_points = Vec::with_capacity(target_size);
    let mut seen = HashSet::new();

    for candidate in population.iter().take(search_config.elite_count) {
        if seen.insert(candidate.point.clone()) {
            next_points.push(candidate.point.clone());
        }
    }

    let mut failed_attempts = 0usize;
    while next_points.len() < target_size && failed_attempts < target_size.saturating_mul(24) {
        let roll = rng.next_f64();
        let mut candidate = if roll < 0.15 {
            random_point(search_space, rng)
        } else if roll < 0.45 {
            let mut point = population[0].point.clone();
            mutate_point(search_space, &mut point, 0.65, rng);
            point
        } else {
            let parent_a = tournament_select(population, search_config.tournament_size, rng);
            let parent_b = tournament_select(population, search_config.tournament_size, rng);
            let mut child = crossover(&parent_a.point, &parent_b.point, rng);
            mutate_point(search_space, &mut child, search_config.mutation_rate, rng);
            child
        };

        if (score_cache.contains_key(&candidate) || seen.contains(&candidate))
            && score_cache.len() < search_space.max_unique_points()
        {
            if let Some(unseen_point) = random_unseen_point(search_space, score_cache, &seen, rng) {
                candidate = unseen_point;
            }
        }

        if seen.insert(candidate.clone()) {
            next_points.push(candidate);
            failed_attempts = 0;
        } else {
            failed_attempts += 1;
        }
    }

    next_points
}

fn reseed_population(
    search_space: &SearchSpace,
    search_config: SearchConfig,
    population: &[ScoredCandidate],
    score_cache: &HashMap<SearchPoint, f64>,
    rng: &mut SimpleRng,
) -> Vec<SearchPoint> {
    let target_size = search_config
        .population_size
        .min(search_space.max_unique_points());
    let mut next_points = Vec::with_capacity(target_size);
    let mut seen = HashSet::new();

    for candidate in population.iter().take(search_config.elite_count) {
        if seen.insert(candidate.point.clone()) {
            next_points.push(candidate.point.clone());
        }
    }

    let mut failed_attempts = 0usize;
    while next_points.len() < target_size && failed_attempts < target_size.saturating_mul(24) {
        let mut candidate = if rng.gen_bool(0.50) {
            if let Some(unseen_point) = random_unseen_point(search_space, score_cache, &seen, rng) {
                unseen_point
            } else {
                random_point(search_space, rng)
            }
        } else {
            let parent = tournament_select(population, search_config.tournament_size, rng);
            let mut point = parent.point.clone();
            mutate_point(search_space, &mut point, 0.80, rng);
            point
        };

        if score_cache.contains_key(&candidate)
            && score_cache.len() < search_space.max_unique_points()
        {
            if let Some(unseen_point) = random_unseen_point(search_space, score_cache, &seen, rng) {
                candidate = unseen_point;
            }
        }

        if seen.insert(candidate.clone()) {
            next_points.push(candidate);
            failed_attempts = 0;
        } else {
            failed_attempts += 1;
        }
    }

    next_points
}

fn tournament_select<'a>(
    population: &'a [ScoredCandidate],
    tournament_size: usize,
    rng: &mut SimpleRng,
) -> &'a ScoredCandidate {
    let mut best = &population[rng.gen_range(population.len())];

    for _ in 1..tournament_size {
        let candidate = &population[rng.gen_range(population.len())];
        if candidate.score > best.score {
            best = candidate;
        }
    }

    best
}

fn crossover(left: &[usize], right: &[usize], rng: &mut SimpleRng) -> SearchPoint {
    left.iter()
        .zip(right.iter())
        .map(|(left_gene, right_gene)| {
            if rng.gen_bool(0.50) {
                *left_gene
            } else {
                *right_gene
            }
        })
        .collect()
}

fn mutate_point(
    search_space: &SearchSpace,
    point: &mut SearchPoint,
    mutation_rate: f64,
    rng: &mut SimpleRng,
) {
    let mutable_dimensions = search_space
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| (dimension.values.len() > 1).then_some(index))
        .collect::<Vec<_>>();

    if mutable_dimensions.is_empty() {
        return;
    }

    let mut mutated = false;

    for dimension_index in &mutable_dimensions {
        if rng.gen_bool(mutation_rate) {
            let choice_count = search_space.dimensions[*dimension_index].values.len();
            point[*dimension_index] =
                random_different_index(choice_count, point[*dimension_index], rng);
            mutated = true;
        }
    }

    if !mutated {
        let dimension_index = mutable_dimensions[rng.gen_range(mutable_dimensions.len())];
        let choice_count = search_space.dimensions[dimension_index].values.len();
        point[dimension_index] = random_different_index(choice_count, point[dimension_index], rng);
    }
}

fn random_different_index(choice_count: usize, current_index: usize, rng: &mut SimpleRng) -> usize {
    let next_index = rng.gen_range(choice_count - 1);
    if next_index >= current_index {
        next_index + 1
    } else {
        next_index
    }
}

fn random_point(search_space: &SearchSpace, rng: &mut SimpleRng) -> SearchPoint {
    search_space
        .dimensions
        .iter()
        .map(|dimension| rng.gen_range(dimension.values.len()))
        .collect()
}

fn random_unseen_point(
    search_space: &SearchSpace,
    score_cache: &HashMap<SearchPoint, f64>,
    seen: &HashSet<SearchPoint>,
    rng: &mut SimpleRng,
) -> Option<SearchPoint> {
    if score_cache.len() + seen.len() >= search_space.max_unique_points() {
        return None;
    }

    for _ in 0..128 {
        let point = random_point(search_space, rng);
        if !score_cache.contains_key(&point) && !seen.contains(&point) {
            return Some(point);
        }
    }

    None
}

#[derive(Debug, Clone, Copy)]
enum Boundary {
    Low,
    Middle,
    High,
}

fn boundary_point(search_space: &SearchSpace, boundary: Boundary) -> SearchPoint {
    search_space
        .dimensions
        .iter()
        .map(|dimension| match boundary {
            Boundary::Low => 0,
            Boundary::Middle => dimension.values.len() / 2,
            Boundary::High => dimension.values.len().saturating_sub(1),
        })
        .collect()
}

fn default_search_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0xD1CE_BA5E_F00D_u64)
}

fn format_parameter_set(parameter_set: &TraderGlobals) -> String {
    if parameter_set.is_empty() {
        return "<default>".to_string();
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_config_uses_fraction_of_large_grid() {
        let parameters = vec![
            ("A", values![0, 1, 2, 3, 4, 5]),
            ("B", values![0, 1, 2, 3, 4, 5]),
            ("C", values![0, 1, 2, 3, 4, 5]),
            ("D", values![0, 1, 2, 3, 4, 5]),
            ("E", values![0, 1, 2, 3, 4, 5]),
        ];
        let search_space = SearchSpace::new(&parameters).unwrap();
        let search_config = SearchConfig::for_space(&search_space);

        assert!(search_config.evaluation_budget < search_space.max_unique_points());
    }

    #[test]
    fn evolutionary_search_finds_known_peak() {
        let parameters = vec![
            ("A", values![0, 1, 2, 3]),
            ("B", values![0, 1, 2, 3]),
            ("C", values![0, 1, 2, 3]),
        ];
        let search_space = SearchSpace::new(&parameters).unwrap();
        let search_config = SearchConfig {
            evaluation_budget: 48,
            population_size: 12,
            elite_count: 3,
            tournament_size: 4,
            mutation_rate: 0.35,
            max_stagnant_generations: 4,
            seed: 7,
        };

        let results = evolutionary_search(&search_space, search_config, &mut |parameter_set| {
            let a = parameter_set.get("A").and_then(Value::as_i64).unwrap() as f64;
            let b = parameter_set.get("B").and_then(Value::as_i64).unwrap() as f64;
            let c = parameter_set.get("C").and_then(Value::as_i64).unwrap() as f64;

            Ok(500.0
                - 20.0 * (a - 2.0).powi(2)
                - 12.0 * (b - 1.0).powi(2)
                - 25.0 * (c - 3.0).powi(2))
        })
        .unwrap();

        assert!(
            results[0].1 >= 480.0,
            "expected a near-optimal score, got {}",
            results[0].1
        );
    }
}
