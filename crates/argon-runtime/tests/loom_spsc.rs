//! Model check for the release/acquire publication protocol used by SPSC slots.

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

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
