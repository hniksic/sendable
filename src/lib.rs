//! `SendRc<T>`, a wrapper around `Rc<T>` that is `Send` if `T` is `Send`.
//!
//! Sometimes it is useful to construct a hierarchy of objects which include `Rc`s and
//! send it off to another thread. `Rc` prohibits that because it can't statically prove
//! that all the clones of an individual `Rc` have been moved to the new thread.
//! `Rc::clone()` and `Rc::drop()` access and modify the reference count without
//! synchronization, which would lead to a data race if two `Rc` clones were to exist in
//! different threads.
//!
//! Using `Arc` helps to an extent, but still requires `T` to be `Sync`, so you can't move
//! a hierarchy with `Arc<RefCell<T>>` to a different thread. The `Sync` requirement is
//! because `Arc` derefs to `&T`, so allowing `Arc` clones containing non-`Sync` values to
//! exist in different threads would break the invariant of non-`Sync` values inside -
//! e.g. it would enable an `Arc<RefCell<u32>>` to execute `borrow()` or `borrow_mut()`
//! from two threads without synchronization.
//!
//! `SendRc` resolves the above by storing the thread ID of the thread in which it was
//! created. Each deref, clone, and drop of the `SendRc` is only allowed from that
//! thread. After moving `SendRc` to a different thread, you must invoke `migrate()` to
//! migrate it to the new thread. Only after all the clones of a `SendRc` have been thus
//! marked will access to the inner value (and to `SendRc::clone()` and `SendRc::drop()`)
//! be allowed.

#![warn(missing_docs)]

use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

struct Inner<T> {
    // id of thread from which the value can be accessed
    pinned_to: AtomicU64,
    val: T,
    // ThreadSend is only needed during migration to other thread -- box it to reduce
    // memory overhead of wrapping.
    migration: Mutex<Option<Box<Migration>>>,
}

struct Migration {
    new_thread: u64,
    migrated_clones: HashSet<usize>,
}

/// Wrapper around `Rc<T>` that is `Send` if `T` is `Send`.
///
/// ```
/// # use std::cell::RefCell;
/// # use send_rc::SendRc;
/// let mut r1 = SendRc::new(RefCell::new(1));
/// let mut r2 = SendRc::clone(&r1);
/// std::thread::spawn(move || {
///     r1.migrate();
///     r2.migrate();
///     *r1.borrow_mut() += 1;
///     assert_eq!(*r2.borrow(), 2);
/// })
/// .join()
/// .unwrap();
/// ```
pub struct SendRc<T>(ManuallyDrop<Rc<Inner<T>>>);

// Safety: SendRc can be sent between threads because we prohibit access to Rc::clone(),
// Rc::drop(), and Rc::deref() except from the thread they are pinned to. After sending
// them to another thread, you have to call SendRc::migrate() on every one of them, which
// establishes synchronization between subsequent accesses and previous ones.
unsafe impl<T> Send for SendRc<T> where T: Send {}

impl<T> SendRc<T> {
    /// Constructs a new SendRc<T>
    pub fn new(val: T) -> Self {
        SendRc(ManuallyDrop::new(Rc::new(Inner {
            val,
            pinned_to: AtomicU64::new(Self::current_thread()),
            migration: Mutex::new(None),
        })))
    }

    fn current_thread() -> u64 {
        unsafe { std::mem::transmute(std::thread::current().id()) }
    }

    fn check_thread(&self) -> bool {
        self.0.pinned_to.load(Ordering::Relaxed) == Self::current_thread()
    }

    fn check_thread_panic(&self) {
        if !self.check_thread() {
            panic!("SendRc accessed from incorrect thread; call migrate()");
        }
    }

    /// Migrate this `SendRc` to the current thread.
    ///
    /// This must be called on all clones of a `SendRc` after sending them to a thread.
    /// After a `SendRc` has been sent to another thread, it panics on deref, drop, or
    /// clone, until `migrate()` has been invoked on all the clones.
    ///
    /// After `migrate()` has been called on all the clones, `SendRc` becomes functional
    /// again and the underlying value becomes accessible.
    ///
    /// Calling `migrate()` on clones of the same `SendRc` from different threads will
    /// also result in panic.
    pub fn migrate(&mut self) {
        // Note: we take &mut self although the implementation doesn't need it, in order
        // to assert that this is meant to be invoked on a SendRc we own. Since SendRc
        // isn't Sync, this is not required for soundness.

        let this_thread = Self::current_thread();

        // Temporarily pin the allocation to an impossible ThreadId, thereby disabling its
        // use while migration is in progress. This prevents clones in the original thread
        // (if they weren't all transferred) to execute clone() and drop() while we're
        // running.
        self.0.pinned_to.store(0, Ordering::Relaxed);

        let migration_opt = &mut *self.0.migration.lock();

        let done = {
            // we only care about the strong count because SendRc doesn't expose weak refs
            let clones_total = Rc::strong_count(&self.0);
            let migration = if let Some(migration) = migration_opt {
                if migration.new_thread != this_thread {
                    panic!("SendRc::<T>::migrate() invoked on the same T from different threads");
                }
                migration
            } else {
                migration_opt.insert(Box::new(Migration {
                    new_thread: this_thread,
                    migrated_clones: HashSet::with_capacity(clones_total),
                }))
            };
            migration.migrated_clones.insert(self as *const _ as usize);
            migration.migrated_clones.len() == clones_total
        };

        if done {
            *migration_opt = None;
            self.0.pinned_to.store(this_thread, Ordering::Relaxed);
        }
    }

    /// Returns the number of pointers to this allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or migrated to.
    pub fn strong_count(this: &Self) -> usize {
        this.check_thread();
        Rc::strong_count(&this.0)
    }

    /// Returns a mutable reference into the given `Rc`, if there are no other `Rc`
    /// pointers to the same allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or migrated to.
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        this.check_thread();
        Rc::get_mut(&mut this.0).map(|inner| &mut inner.val)
    }

    /// Returns true if the two `SendRc`s point to the same allocation.
    pub fn ptr_eq(this: &mut Self, other: &Self) -> bool {
        this.check_thread();
        Rc::ptr_eq(&this.0, &other.0)
    }
}

impl<T> Deref for SendRc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.check_thread_panic();
        &self.0.val
    }
}

impl<T> Clone for SendRc<T> {
    fn clone(&self) -> Self {
        self.check_thread_panic();
        SendRc(ManuallyDrop::new(Rc::clone(&self.0)))
    }
}

impl<T> Drop for SendRc<T> {
    fn drop(&mut self) {
        if self.check_thread() {
            unsafe {
                ManuallyDrop::drop(&mut self.0);
            }
        } else if !std::thread::panicking() {
            // Don't panic if we're already panicking, it brings down the program and
            // breaks unit testing. If we're already panicking, then just leak the Rc.
            panic!("access from wrong thread");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::mem::ManuallyDrop;
    use std::sync::{Arc, Barrier};

    use super::SendRc;

    #[test]
    fn trivial() {
        let r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        *r1.borrow_mut() = 2;
        assert_eq!(*r2.borrow(), 2);
    }

    #[test]
    fn ok_send() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            r1.migrate();
            r2.migrate();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn send_and_return() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let (mut r1, mut r2) = std::thread::spawn(move || {
            r1.migrate();
            r2.migrate();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
            (r1, r2)
        })
        .join()
        .unwrap();
        r1.migrate();
        r2.migrate();
        assert_eq!(*r2.borrow(), 2);
    }

    #[test]
    #[should_panic]
    fn invalid_send() {
        let r = SendRc::new(RefCell::new(1));
        std::thread::spawn(move || {
            drop(r);
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_use_1() {
        let r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            *r1.borrow_mut() = 2;
            assert_eq!(*r2.borrow(), 2);
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_use_2() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let (_r1, r2) = std::thread::spawn(move || {
            r1.migrate();
            r2.migrate();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
            (r1, r2)
        })
        .join()
        .unwrap();
        // here we must call migrate() as well
        assert_eq!(*r2.borrow(), 2);
    }

    #[test]
    #[should_panic]
    fn incomplete_send() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            r1.migrate();
            let _ = &*r2;
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn fakse_send() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            // can't call migrate twice on the same value
            r1.migrate();
            r1.migrate();
            let _ = &*r2;
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn migrate_diff_threads() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            r1.migrate();
        })
        .join()
        .unwrap();
        std::thread::spawn(move || {
            r2.migrate();
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn migrate_half_way() {
        let barrier = Arc::new(Barrier::new(2));
        let mut r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn({
            let barrier = barrier.clone();
            move || {
                r1.migrate();
                barrier.wait();
                let _ = ManuallyDrop::new(r1); // avoid another panic
            }
        });
        barrier.wait();
        // not allowed to proceed
        let _ = r2.clone();
    }
}
