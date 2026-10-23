#![no_main]

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_runtime::ArgonCoordinator;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let grid = PersistentGridConfig {
        grid_dim_x: 1,
        block_dim_x: 32,
        shared_mem_bytes: 16_384,
        tile_dim: 8,
    };
    let plasticity = PlasticityConfig {
        learning_rate: 0.01,
        goodness_threshold: 1.0,
        weight_decay: 0.001,
        phase_polarity: 1.0,
    };
    let Ok(mut coordinator) = ArgonCoordinator::new(2, &grid, plasticity, 4) else {
        return;
    };
    let mut features = [0.0_f32; 8];
    for (value, byte) in features.iter_mut().zip(data) {
        *value = (f32::from(*byte) / 127.5) - 1.0;
    }
    let mut output = [0.0_f32; 8];
    let polarity = if data[0] & 1 == 0 { 1.0 } else { -1.0 };
    let _result = coordinator.step_borrowed(0, polarity, &features, &mut output);
});
