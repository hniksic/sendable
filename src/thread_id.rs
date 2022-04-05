use std::sync::atomic::{AtomicU64, Ordering};

// Start with 1, so we never get a thread with id 0. It's often useful to have an
// "impossible" id because we can't use Option<u64> when storing the thread id in an
// atomic.
static NEXT_THREAD_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static COUNTER: u64 = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
}

// Ideally we'd use std::thread::current().as_u64(), but it's not yet stable. Also, for
// some reason this seems much faster than accessing thread_id. Before switching to
// ThreadId::as_u64(), both should be benchmarked.
pub fn current_thread() -> u64 {
    COUNTER.with(|&counter| counter)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::current_thread;

    #[test]
    fn various() {
        let id0 = current_thread();
        assert_eq!(id0, current_thread());
        let threads = (0..100).map(|_| {
            std::thread::spawn(|| {
                let id = current_thread();
                assert_eq!(id, current_thread());
                id
            })})
            .collect::<Vec<_>>();
        let mut ids = HashSet::from([id0]);
        ids.extend(threads.into_iter().map(|thr| thr.join().unwrap()));
        assert_eq!(ids.len(), 101);
        assert_eq!(id0, current_thread());
        assert!(ids.into_iter().all(|id| id != 0));
    }
}
