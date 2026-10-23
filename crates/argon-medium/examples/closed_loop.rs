//! Minimal closed-loop physical stabilization example.

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_medium::{CoupledOscillatorMedium, MediumConfig, PdController};
use argon_runtime::ArgonCoordinator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = PersistentGridConfig {
        grid_dim_x: 1,
        block_dim_x: 64,
        shared_mem_bytes: 16_384,
        tile_dim: 8,
    };
    let plasticity = PlasticityConfig {
        learning_rate: 0.002,
        goodness_threshold: 1.0,
        weight_decay: 0.000_5,
        phase_polarity: 1.0,
    };
    let mut coordinator = ArgonCoordinator::new(2, &grid, plasticity, 8)?;
    let mut medium = CoupledOscillatorMedium::new(MediumConfig::default())?;
    let mut baseline = PdController::new(4, 0.03, 0.02)?;
    medium.perturb(0, 1.0, 1.5)?;
    for _ in 0..10_000 {
        let _telemetry = medium.step_with_controller(&mut coordinator, &mut baseline)?;
    }
    println!("final_energy={}", medium.energy());
    Ok(())
}
