//! Model check for the release/acquire publication protocol used by SPSC slots.
//!
//! This target opts out of leak detection — see [`__lsan_default_options`].

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;
use std::os::raw::c_char;

/// `LeakSanitizer` defaults for the `loom_spsc` test executable.
///
/// `loom::model` re-runs the closure for every interleaving the model checker
/// explores and intentionally keeps those allocations alive until the
/// exploration finishes, so `LeakSanitizer` reports them at process exit and
/// the `AddressSanitizer` job fails with
/// `ERROR: LeakSanitizer: detected memory leaks` even though every test
/// passed. The reported blocks come from loom's model harness, not from argon
/// code, so they are false positives.
///
/// This hook is compiled into the `loom_spsc` integration-test executable only
/// (each `tests/*.rs` file is its own binary), so leak detection stays enabled
/// for every other test target in the workspace and CI's `ASAN_OPTIONS` is
/// left untouched. The sanitizer runtime reads the returned NUL-terminated
/// string when the binary starts.
#[no_mangle]
pub const extern "C" fn __lsan_default_options() -> *const c_char {
    c"detect_leaks=0".as_ptr()
}

#[test]
fn release_head_publishes_slot_before_consumer_reads() {
    loom::model(|| {
        let slot = Arc::new(AtomicUsize::new(0));
        let head = Arc::new(AtomicUsize::new(0));
        let producer_slot = Arc::clone(&slot);
        let producer_head = Arc::clone(&head);
        let producer = thread::spawn(move || {
            producer_slot.store(42, Ordering::Relaxed);
            producer_head.store(1, Ordering::Release);
        });
        let consumer_slot = Arc::clone(&slot);
        let consumer_head = Arc::clone(&head);
        let consumer = thread::spawn(move || {
            if consumer_head.load(Ordering::Acquire) == 1 {
                assert_eq!(consumer_slot.load(Ordering::Relaxed), 42);
            }
        });
        assert!(producer.join().is_ok());
        assert!(consumer.join().is_ok());
    });
}
