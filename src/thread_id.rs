use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_THREAD_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static COUNTER: u64 = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
}

// We'd like to use std::thread::current().as_u64(), but it's not yet stable.  Also, for
// some reason this also seems much faster than accessing thread_id. Before switching to
// ThreadId::as_u64(), both should be benchmarked.
pub fn current_thread() -> u64 {
    COUNTER.with(|counter| *counter)
}

#[cfg(test)]
mod tests {
    use super::current_thread;

    #[test]
    fn various() {
        let t1 = current_thread();
        assert_eq!(t1, current_thread());
        let t2 = std::thread::spawn(|| {
            let id = current_thread();
            assert_eq!(id, current_thread());
            id
        })
        .join()
        .unwrap();
        let t3 = std::thread::spawn(|| {
            let id = current_thread();
            assert_eq!(id, current_thread());
            id
        })
        .join()
        .unwrap();
        assert!(t1 != t2);
        assert!(t2 != t3);
        assert!(t1 != t3);
        assert_eq!(t1, current_thread());
    }
}
