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

trait SweepRangeValue: Copy {
    fn to_f64(self) -> f64;
    fn add_steps(start: Self, step: Self, step_count: usize) -> Self;
    fn to_json(self) -> Value;
}

impl SweepRangeValue for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn add_steps(start: Self, step: Self, step_count: usize) -> Self {
        start + step * step_count as i64
    }

    fn to_json(self) -> Value {
        serde_json::json!(self)
    }
}

impl SweepRangeValue for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn add_steps(start: Self, step: Self, step_count: usize) -> Self {
        start + step * step_count as i32
    }

    fn to_json(self) -> Value {
        serde_json::json!(self)
    }
}

impl SweepRangeValue for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn add_steps(start: Self, step: Self, step_count: usize) -> Self {
        start + step * step_count as f64
    }

    fn to_json(self) -> Value {
        serde_json::json!(self)
    }
}

fn range_values<T>(start: T, end: T, step: T) -> Vec<Value>
where
    T: SweepRangeValue,
{
    let start_f64 = start.to_f64();
    let end_f64 = end.to_f64();
    let step_f64 = step.to_f64();

    assert!(
        start_f64.is_finite() && end_f64.is_finite() && step_f64.is_finite(),
        "range_values requires finite numeric bounds and step",
    );
    assert!(step_f64 != 0.0, "range_values step must be non-zero");

    let ascending = end_f64 >= start_f64;
    assert!(
        (ascending && step_f64 > 0.0) || (!ascending && step_f64 < 0.0),
        "range_values step direction must move from start toward end",
    );

    let epsilon = (step_f64.abs() * 1e-9).max(1e-12);
    let mut values = Vec::new();
    let mut step_count = 0usize;
    let max_points = 100_000usize;

    loop {
        assert!(
            step_count < max_points,
            "range_values generated more than {max_points} points; check your bounds and step",
        );

        let current = T::add_steps(start, step, step_count);
        let current_f64 = current.to_f64();

        if ascending {
            if current_f64 > end_f64 + epsilon {
                break;
            }
        } else if current_f64 < end_f64 - epsilon {
            break;
        }

        values.push(current.to_json());
        step_count += 1;
    }

    values
}

type SearchPoint = Vec<usize>;

#[derive(Debug, Clone, Copy)]
enum SearchAlgorithm {
    Genetic,
    SimulatedAnnealing,
}

impl SearchAlgorithm {
    fn label(self) -> &'static str {
        match self {
            Self::Genetic => "genetic",
            Self::SimulatedAnnealing => "simulated annealing",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "genetic" | "ga" => Some(Self::Genetic),
            "annealing" | "simulated_annealing" | "simulated-annealing" | "sa" => {
                Some(Self::SimulatedAnnealing)
            }
            _ => None,
        }
    }
}

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
    algorithm: SearchAlgorithm,
    evaluation_budget: usize,
    seed: u64,
    population_size: usize,
    elite_count: usize,
    tournament_size: usize,
    mutation_rate: f64,
    max_stagnant_generations: usize,
    annealing_restart_count: usize,
    annealing_start_temperature: f64,
    annealing_end_temperature: f64,
    annealing_neighbor_mutations: usize,
    annealing_random_jump_rate: f64,
}

impl SearchConfig {
    fn for_space(search_space: &SearchSpace, algorithm: SearchAlgorithm) -> Self {
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
        let annealing_restart_count = dimension_count.clamp(3, 6).min(unique_points);

        Self {
            algorithm,
            evaluation_budget,
            seed: default_search_seed(),
            population_size,
            elite_count,
            tournament_size,
            mutation_rate,
            max_stagnant_generations,
            annealing_restart_count,
            annealing_start_temperature: 1.0,
            annealing_end_temperature: 0.01,
            annealing_neighbor_mutations: dimension_count.clamp(1, 3),
            annealing_random_jump_rate: 0.10,
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
        //("WINDOW", range_values(100, 5000, 100)),
        //("Z_ENTER", range_values(-200, 200, 10)),
        //("Z_PASSIVE", range_values(-200, 200, 10)),
    ];
    let search_algorithm = configured_search_algorithm();
    let search_space = SearchSpace::new(&sweep_parameters)?;
    let search_config = SearchConfig::for_space(&search_space, search_algorithm);
    let suppress_log_writes = search_space.max_unique_points() > 1;

    match search_config.algorithm {
        SearchAlgorithm::Genetic => println!(
            "Running {} parameter search over {} combinations with a budget of {} evaluations (population {}, elites {}, mutation {:.0}%, seed {}).",
            search_config.algorithm.label(),
            search_space.total_combinations,
            search_config.evaluation_budget,
            search_config.population_size,
            search_config.elite_count,
            search_config.mutation_rate * 100.0,
            search_config.seed,
        ),
        SearchAlgorithm::SimulatedAnnealing => println!(
            "Running {} parameter search over {} combinations with a budget of {} evaluations (restarts {}, temperature {:.2}->{:.2}, max neighbor hops {}, seed {}).",
            search_config.algorithm.label(),
            search_space.total_combinations,
            search_config.evaluation_budget,
            search_config.annealing_restart_count,
            search_config.annealing_start_temperature,
            search_config.annealing_end_temperature,
            search_config.annealing_neighbor_mutations,
            search_config.seed,
        ),
    }

    let progress_bar = ProgressBar::new(search_config.evaluation_budget as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {percent}% {msg}",
        )?
        .progress_chars("=>-"),
    );

    let mut evaluate = |parameter_set: &TraderGlobals| -> Result<f64> {
        progress_bar.set_message(format_parameter_set(parameter_set));
        let result = rust_backtester::cli::run_with_options(
            parameter_set,
            rust_backtester::cli::RunOptions {
                suppress_log_writes,
            },
        )?;
        progress_bar.inc(1);
        Ok(result)
    };
    let progress_bar_for_status = progress_bar.clone();
    let mut report_status = move |message: String| {
        progress_bar_for_status.println(message);
    };

    let mut results = run_search(
        &search_space,
        search_config,
        &mut evaluate,
        &mut report_status,
    )?;

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

fn run_search<F, S>(
    search_space: &SearchSpace,
    search_config: SearchConfig,
    evaluator: &mut F,
    status_reporter: &mut S,
) -> Result<Vec<(TraderGlobals, f64)>>
where
    F: FnMut(&TraderGlobals) -> Result<f64>,
    S: FnMut(String),
{
    match search_config.algorithm {
        SearchAlgorithm::Genetic => evolutionary_search(search_space, search_config, evaluator),
        SearchAlgorithm::SimulatedAnnealing => {
            simulated_annealing_search(search_space, search_config, evaluator, status_reporter)
        }
    }
}

fn configured_search_algorithm() -> SearchAlgorithm {
    std::env::var("RUST_BACKTESTER_SEARCH_ALGORITHM")
        .ok()
        .and_then(|value| SearchAlgorithm::parse(&value))
        .unwrap_or(SearchAlgorithm::Genetic)
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

fn simulated_annealing_search<F, S>(
    search_space: &SearchSpace,
    search_config: SearchConfig,
    evaluator: &mut F,
    status_reporter: &mut S,
) -> Result<Vec<(TraderGlobals, f64)>>
where
    F: FnMut(&TraderGlobals) -> Result<f64>,
    S: FnMut(String),
{
    let max_unique_points = search_space.max_unique_points();
    let evaluation_budget = search_config
        .evaluation_budget
        .min(max_unique_points)
        .max(1);
    let mut rng = SimpleRng::seeded(search_config.seed);
    let mut score_cache = HashMap::new();
    let mut all_results = Vec::new();
    let restart_count = search_config
        .annealing_restart_count
        .min(evaluation_budget)
        .max(1);
    let report_interval = evaluation_budget.div_ceil(20).max(1);
    let mut next_report_at = report_interval;
    let mut total_attempted_moves = 0usize;
    let mut total_accepted_moves = 0usize;
    let mut total_worse_accepts = 0usize;
    let mut best_overall: Option<ScoredCandidate> = None;

    for restart_index in 0..restart_count {
        if score_cache.len() >= evaluation_budget {
            break;
        }

        let start_point =
            annealing_start_point(search_space, restart_index, &score_cache, &mut rng);
        let mut current = score_candidate(
            start_point,
            search_space,
            evaluator,
            &mut score_cache,
            &mut all_results,
        )?;
        if best_overall
            .as_ref()
            .is_none_or(|best| current.score > best.score)
        {
            best_overall = Some(current.clone());
        }
        let mut restart_best = current.clone();
        let remaining_restarts = restart_count.saturating_sub(restart_index).max(1);
        let remaining_budget = evaluation_budget.saturating_sub(score_cache.len());
        let steps_this_restart = remaining_budget.div_ceil(remaining_restarts).max(1);
        let restart_number = restart_index + 1;

        status_reporter(format!(
            "SA restart {restart_number}/{restart_count}: start score {:.2}, best {:.2}, evals {}/{}",
            current.score,
            best_overall
                .as_ref()
                .map(|candidate| candidate.score)
                .unwrap_or(current.score),
            score_cache.len(),
            evaluation_budget,
        ));

        for step in 0..steps_this_restart {
            if score_cache.len() >= evaluation_budget {
                break;
            }

            let temperature = annealing_temperature(search_config, step, steps_this_restart);
            let candidate_point = next_annealing_point(
                search_space,
                &current.point,
                search_config,
                &score_cache,
                &mut rng,
            );
            let candidate = score_candidate(
                candidate_point,
                search_space,
                evaluator,
                &mut score_cache,
                &mut all_results,
            )?;
            total_attempted_moves += 1;
            let previous_score = current.score;

            let improved_restart_best = candidate.score > restart_best.score;
            if improved_restart_best {
                restart_best = candidate.clone();
            }
            let found_new_best = best_overall
                .as_ref()
                .is_none_or(|best| candidate.score > best.score);
            if found_new_best {
                best_overall = Some(candidate.clone());
            }

            let accepted =
                should_accept_annealing_move(current.score, candidate.score, temperature, &mut rng);
            if accepted {
                total_accepted_moves += 1;
                if candidate.score < previous_score {
                    total_worse_accepts += 1;
                }
                current = candidate;
            }

            let should_report_progress = score_cache.len() >= next_report_at
                || step + 1 == steps_this_restart
                || found_new_best
                || improved_restart_best;

            if should_report_progress {
                let acceptance_rate = if total_attempted_moves == 0 {
                    0.0
                } else {
                    (total_accepted_moves as f64 / total_attempted_moves as f64) * 100.0
                };
                status_reporter(format!(
                    "SA progress: evals {}/{}, restart {}/{}, step {}/{}, temp {:.4}, current {:.2}, restart best {:.2}, overall best {:.2}, accept {:.1}%, worse accepts {}",
                    score_cache.len(),
                    evaluation_budget,
                    restart_number,
                    restart_count,
                    step + 1,
                    steps_this_restart,
                    temperature,
                    current.score,
                    restart_best.score,
                    best_overall
                        .as_ref()
                        .map(|candidate| candidate.score)
                        .unwrap_or(restart_best.score),
                    acceptance_rate,
                    total_worse_accepts,
                ));

                while next_report_at <= score_cache.len() {
                    next_report_at = next_report_at.saturating_add(report_interval);
                }
            }
        }

        status_reporter(format!(
            "SA restart {restart_number}/{restart_count} complete: restart best {:.2}, overall best {:.2}, evals {}/{}",
            restart_best.score,
            best_overall
                .as_ref()
                .map(|candidate| candidate.score)
                .unwrap_or(restart_best.score),
            score_cache.len(),
            evaluation_budget,
        ));
    }

    refine_best_candidate(
        search_space,
        evaluation_budget,
        evaluator,
        &mut score_cache,
        &mut all_results,
    )?;
    if let Some(best_candidate) = all_results
        .iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
    {
        let acceptance_rate = if total_attempted_moves == 0 {
            0.0
        } else {
            (total_accepted_moves as f64 / total_attempted_moves as f64) * 100.0
        };
        status_reporter(format!(
            "SA complete: best {:.2}, evals {}/{}, attempted moves {}, accept {:.1}%, worse accepts {}",
            best_candidate.score,
            score_cache.len(),
            evaluation_budget,
            total_attempted_moves,
            acceptance_rate,
            total_worse_accepts,
        ));
    }

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

fn annealing_start_point(
    search_space: &SearchSpace,
    restart_index: usize,
    score_cache: &HashMap<SearchPoint, f64>,
    rng: &mut SimpleRng,
) -> SearchPoint {
    let boundary_points = [
        boundary_point(search_space, Boundary::Middle),
        boundary_point(search_space, Boundary::Low),
        boundary_point(search_space, Boundary::High),
    ];

    if let Some(point) = boundary_points.get(restart_index) {
        if !score_cache.contains_key(point) {
            return point.clone();
        }
    }

    random_unseen_point(search_space, score_cache, &HashSet::new(), rng)
        .unwrap_or_else(|| random_point(search_space, rng))
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

fn annealing_temperature(search_config: SearchConfig, step_index: usize, step_count: usize) -> f64 {
    if step_count <= 1 {
        return search_config.annealing_end_temperature.max(f64::EPSILON);
    }

    let progress = step_index as f64 / (step_count.saturating_sub(1)) as f64;
    let start = search_config.annealing_start_temperature.max(f64::EPSILON);
    let end = search_config.annealing_end_temperature.max(f64::EPSILON);

    start.powf(1.0 - progress) * end.powf(progress)
}

fn next_annealing_point(
    search_space: &SearchSpace,
    current_point: &[usize],
    search_config: SearchConfig,
    score_cache: &HashMap<SearchPoint, f64>,
    rng: &mut SimpleRng,
) -> SearchPoint {
    if rng.gen_bool(search_config.annealing_random_jump_rate) {
        return random_unseen_point(search_space, score_cache, &HashSet::new(), rng)
            .unwrap_or_else(|| random_point(search_space, rng));
    }

    random_unseen_neighbor(search_space, current_point, search_config, score_cache, rng)
        .unwrap_or_else(|| annealing_neighbor(search_space, current_point, search_config, rng))
}

fn random_unseen_neighbor(
    search_space: &SearchSpace,
    current_point: &[usize],
    search_config: SearchConfig,
    score_cache: &HashMap<SearchPoint, f64>,
    rng: &mut SimpleRng,
) -> Option<SearchPoint> {
    if score_cache.len() >= search_space.max_unique_points() {
        return None;
    }

    for _ in 0..64 {
        let point = annealing_neighbor(search_space, current_point, search_config, rng);
        if !score_cache.contains_key(&point) {
            return Some(point);
        }
    }

    None
}

fn annealing_neighbor(
    search_space: &SearchSpace,
    current_point: &[usize],
    search_config: SearchConfig,
    rng: &mut SimpleRng,
) -> SearchPoint {
    let mutable_dimensions = mutable_dimension_indices(search_space);
    if mutable_dimensions.is_empty() {
        return current_point.to_vec();
    }

    let max_mutations = search_config
        .annealing_neighbor_mutations
        .min(mutable_dimensions.len())
        .max(1);
    let mutation_count = 1 + rng.gen_range(max_mutations);
    let mut next_point = current_point.to_vec();
    let mut used_dimensions = HashSet::new();

    while used_dimensions.len() < mutation_count {
        let dimension_index = mutable_dimensions[rng.gen_range(mutable_dimensions.len())];
        if !used_dimensions.insert(dimension_index) {
            continue;
        }

        let choice_count = search_space.dimensions[dimension_index].values.len();
        next_point[dimension_index] =
            random_different_index(choice_count, next_point[dimension_index], rng);
    }

    next_point
}

fn should_accept_annealing_move(
    current_score: f64,
    candidate_score: f64,
    temperature: f64,
    rng: &mut SimpleRng,
) -> bool {
    if candidate_score >= current_score {
        return true;
    }

    let score_delta = candidate_score - current_score;
    let acceptance_probability = (score_delta / temperature.max(f64::EPSILON))
        .max(-50.0)
        .exp()
        .clamp(0.0, 1.0);

    rng.gen_bool(acceptance_probability)
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
    let mutable_dimensions = mutable_dimension_indices(search_space);
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

fn mutable_dimension_indices(search_space: &SearchSpace) -> Vec<usize> {
    search_space
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| (dimension.values.len() > 1).then_some(index))
        .collect()
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
        let search_config = SearchConfig::for_space(&search_space, SearchAlgorithm::Genetic);

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
            algorithm: SearchAlgorithm::Genetic,
            evaluation_budget: 48,
            seed: 7,
            population_size: 12,
            elite_count: 3,
            tournament_size: 4,
            mutation_rate: 0.35,
            max_stagnant_generations: 4,
            annealing_restart_count: 4,
            annealing_start_temperature: 1.0,
            annealing_end_temperature: 0.01,
            annealing_neighbor_mutations: 2,
            annealing_random_jump_rate: 0.10,
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

    #[test]
    fn simulated_annealing_finds_known_peak() {
        let parameters = vec![
            ("A", values![0, 1, 2, 3]),
            ("B", values![0, 1, 2, 3]),
            ("C", values![0, 1, 2, 3]),
        ];
        let search_space = SearchSpace::new(&parameters).unwrap();
        let search_config = SearchConfig {
            algorithm: SearchAlgorithm::SimulatedAnnealing,
            evaluation_budget: 48,
            seed: 11,
            population_size: 12,
            elite_count: 3,
            tournament_size: 4,
            mutation_rate: 0.35,
            max_stagnant_generations: 4,
            annealing_restart_count: 4,
            annealing_start_temperature: 1.0,
            annealing_end_temperature: 0.01,
            annealing_neighbor_mutations: 2,
            annealing_random_jump_rate: 0.10,
        };

        let mut report_status = |_message: String| {};
        let results = simulated_annealing_search(
            &search_space,
            search_config,
            &mut |parameter_set| {
                let a = parameter_set.get("A").and_then(Value::as_i64).unwrap() as f64;
                let b = parameter_set.get("B").and_then(Value::as_i64).unwrap() as f64;
                let c = parameter_set.get("C").and_then(Value::as_i64).unwrap() as f64;

                Ok(500.0
                    - 20.0 * (a - 2.0).powi(2)
                    - 12.0 * (b - 1.0).powi(2)
                    - 25.0 * (c - 3.0).powi(2))
            },
            &mut report_status,
        )
        .unwrap();

        assert!(
            results[0].1 >= 480.0,
            "expected a near-optimal score, got {}",
            results[0].1
        );
    }

    #[test]
    fn range_values_supports_integers() {
        let values = range_values(100, 500, 200);

        assert_eq!(
            values
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![100, 300, 500]
        );
    }

    #[test]
    fn range_values_supports_floats() {
        let values = range_values(-0.5_f64, 0.5_f64, 0.25_f64);
        let actual = values
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let expected = vec![-0.5, -0.25, 0.0, 0.25, 0.5];

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.into_iter().zip(expected) {
            assert!((left - right).abs() < 1e-9);
        }
    }
}
