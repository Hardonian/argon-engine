//! Allocation regression coverage for the warmed-up physical control loop.

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_medium::{CoupledOscillatorMedium, MediumConfig};
use argon_runtime::ArgonCoordinator;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct TrackingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    TRACKING.with(|tracking| {
        if tracking.get() {
            ALLOCATIONS.with(|count| count.set(count.get().saturating_add(1)));
        }
    });
}

// SAFETY: Every operation delegates unchanged layouts and pointers to the
// process system allocator; thread-local bookkeeping does not own allocations.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: The caller supplies the layout required by GlobalAlloc.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller supplies a pointer/layout pair from this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: The caller supplies the layout required by GlobalAlloc.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: The caller supplies a valid allocation and requested size.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn warmed_closed_loop_performs_no_allocations() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut medium = CoupledOscillatorMedium::new(MediumConfig {
        energy_boundary: 1_000.0,
        energy_release_boundary: 900.0,
        ..MediumConfig::default()
    })?;
    let _warmup = medium.step(&mut coordinator)?;

    ALLOCATIONS.with(|count| count.set(0));
    TRACKING.with(|tracking| tracking.set(true));
    let mut succeeded = true;
    for _ in 0..1_000 {
        if medium.step(&mut coordinator).is_err() {
            succeeded = false;
            break;
        }
    }
    TRACKING.with(|tracking| tracking.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);

    assert!(succeeded);
    assert_eq!(allocations, 0, "warmed loop allocated {allocations} times");
    Ok(())
}
