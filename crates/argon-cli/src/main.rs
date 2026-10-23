//! Command-line latency benchmark for Argon coordinator and physical-medium loops.

#![forbid(non_ascii_idents)]

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_medium::{CoupledOscillatorMedium, MediumConfig};
use argon_runtime::{ArgonCoordinator, ControlFrame};
use std::env;
use std::error::Error;
use std::time::Instant;

const DEFAULT_ITERATIONS: usize = 10_000;
const TILE_DIM: usize = 8;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.iter().any(|argument| argument == "--help") {
        println!("Usage: argon-cli [--medium] [--iterations N]");
        return Ok(());
    }
    let iterations = parse_iterations(&arguments)?;
    let with_medium = arguments.iter().any(|argument| argument == "--medium");
    let mut coordinator = coordinator()?;
    let mut samples = Vec::with_capacity(iterations);

    if with_medium {
        let mut medium = CoupledOscillatorMedium::new(MediumConfig::default())?;
        medium.perturb(0, 0.5, 0.5)?;
        for _ in 0..iterations {
            let started = Instant::now();
            let _telemetry = medium.step(&mut coordinator)?;
            samples.push(elapsed_nanos(started));
        }
    } else {
        let mut features = vec![0.25_f32; TILE_DIM];
        let mut output = vec![0.0_f32; TILE_DIM];
        for sequence_id in 0..iterations {
            let started = Instant::now();
            coordinator.submit_frame(ControlFrame {
                sequence_id: u64::try_from(sequence_id)?,
                phase_polarity: 1.0,
                features: std::mem::take(&mut features),
            })?;
            let _telemetry = coordinator.poll_and_step_recycle(&mut output, &mut features)?;
            samples.push(elapsed_nanos(started));
        }
    }

    samples.sort_unstable();
    let mode = if with_medium { "medium" } else { "coordinator" };
    println!(
        "mode={mode} iterations={iterations} p50={}ns p95={}ns p99={}ns",
        percentile(&samples, 50),
        percentile(&samples, 95),
        percentile(&samples, 99)
    );
    Ok(())
}

fn coordinator() -> Result<ArgonCoordinator, Box<dyn Error>> {
    let grid = PersistentGridConfig {
        grid_dim_x: 1,
        block_dim_x: 64,
        shared_mem_bytes: 16_384,
        tile_dim: TILE_DIM,
    };
    let plasticity = PlasticityConfig {
        learning_rate: 0.002,
        goodness_threshold: 1.0,
        weight_decay: 0.000_5,
        phase_polarity: 1.0,
    };
    Ok(ArgonCoordinator::new(2, &grid, plasticity, 8)?)
}

fn parse_iterations(arguments: &[String]) -> Result<usize, Box<dyn Error>> {
    let Some(index) = arguments
        .iter()
        .position(|argument| argument == "--iterations")
    else {
        return Ok(DEFAULT_ITERATIONS);
    };
    let value = arguments
        .get(index + 1)
        .ok_or("--iterations requires a positive integer")?;
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err("--iterations must be greater than zero".into());
    }
    Ok(parsed)
}

fn elapsed_nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    let rank = sorted
        .len()
        .saturating_mul(percent)
        .saturating_add(99)
        .saturating_div(100)
        .saturating_sub(1);
    sorted.get(rank).copied().unwrap_or(0)
}
