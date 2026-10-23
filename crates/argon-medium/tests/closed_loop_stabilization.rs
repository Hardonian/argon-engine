//! End-to-end stabilization coverage for the physical sensory-motor loop.

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_medium::{CoupledOscillatorMedium, MediumConfig, MediumError};
use argon_runtime::ArgonCoordinator;

#[test]
fn high_energy_shock_converges_without_divergence() -> Result<(), MediumError> {
    let config = MediumConfig {
        energy_boundary: 2.0,
        max_torque: 0.02,
        ..MediumConfig::default()
    };
    let mut medium = CoupledOscillatorMedium::new(config)?;
    medium.perturb(0, 1.4, 2.5)?;
    medium.perturb(2, -1.0, -1.8)?;
    let shocked_energy = medium.energy();

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

    let mut homeostasis_count = 0_u64;
    let mut final_energy = shocked_energy;
    for _ in 0..10_000 {
        let telemetry = medium.step(&mut coordinator)?;
        assert!(telemetry.energy.is_finite());
        assert!(medium.positions().iter().all(|value| value.is_finite()));
        assert!(medium.velocities().iter().all(|value| value.is_finite()));
        if telemetry.homeostasis_applied {
            homeostasis_count = homeostasis_count.saturating_add(1);
        }
        final_energy = telemetry.energy;
    }

    assert!(homeostasis_count > 0);
    assert!(
        final_energy < shocked_energy * 0.15,
        "initial={shocked_energy}, final={final_energy}"
    );
    Ok(())
}
