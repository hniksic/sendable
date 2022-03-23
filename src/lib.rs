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
use std::fmt::Debug;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{
    AtomicU64, AtomicUsize, Ordering,
    Ordering::{Relaxed, SeqCst},
};
use std::thread::ThreadId;

use parking_lot::Mutex;

struct Inner<T> {
    // id of thread from which the value can be accessed
    pinned_to: AtomicU64,
    val: T,
    // copy of the Rc's strong count, needed for migrate() to accesss it without having to
    // synchronize with clone/drop possibly happening in the original thread
    strong_count: AtomicUsize,
    // ThreadSend is only needed during migration to other thread -- box it to reduce
    // memory overhead of wrapping.
    migration: Mutex<Option<Box<Migration>>>,
}

struct Migration {
    new_thread: u64,
    migrated_clones: HashSet<usize>,
    // Count of clones (SendRcs pointing to the same allocation) at the beginning of
    // migration to the new thread. Calls to clone() and drop() after the migration starts
    // are detected at run-time and cause a panic and prevent the migration from
    // completing.
    clone_count: usize,
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
pub struct SendRc<T>(NonNull<Inner<T>>);

// Safety: SendRc can be sent between threads because we prohibit access to Rc::clone(),
// Rc::drop(), and Rc::deref() except from the thread they are pinned to. After sending
// them to another thread, you have to call SendRc::migrate() on every one of them, which
// establishes synchronization between subsequent accesses and previous ones.
unsafe impl<T> Send for SendRc<T> where T: Send {}

impl<T> SendRc<T> {
    /// Constructs a new SendRc<T>
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(Inner {
            pinned_to: AtomicU64::new(Self::current_thread()),
            val,
            strong_count: AtomicUsize::new(1),
            migration: Mutex::new(None),
        }));
        // unwrap: we have a valid box, its pointer is not null (rustc eliminates the
        // check, https://godbolt.org/z/dsYPxxMWo)
        SendRc(NonNull::new(ptr).unwrap())
    }

    fn inner(&self) -> &Inner<T> {
        // Safety: Inner is valid for as long as self
        unsafe { self.0.as_ref() }
    }

    fn inner_mut(&mut self) -> &mut Inner<T> {
        // Safety: Inner is valid for as long as self
        unsafe { self.0.as_mut() }
    }

    // Needed until ThreadId::as_u64() is stabilized.
    fn current_thread() -> u64 {
        // This is not a guarantee that ThreadId is safe to transmute to u64, but it's
        // better than nothing.
        const _: () = assert!(std::mem::size_of::<ThreadId>() == 8);

        // Safety: ThreadId must have layout compatible with that of a u64, which is the
        // case in the stdlib where it's NonZeroU64.
        unsafe { std::mem::transmute(std::thread::current().id()) }
    }

    #[must_use]
    fn is_pinned_to(&self, thread_id: Option<u64>, ordering: Ordering) -> bool {
        self.inner().pinned_to.load(ordering) == thread_id.unwrap_or_else(Self::current_thread)
    }

    fn assert_pinned_to(&self, thread_id: Option<u64>, op: &str, ordering: Ordering) {
        if !self.is_pinned_to(thread_id, ordering) {
            panic!("SendRc::{}() attempted from incorrect thread; call migrate() first", op);
        }
    }

    /// Migrate this `SendRc` to the current thread.
    ///
    /// Migration is required after moving a `SendRc` to a thread different than the one
    /// it has been created in (or one it has been last migrated to).  For the `SendRc` to
    /// be usable, you must move *all* the `SendRc`s pointing to the same allocation, and
    /// call `migrate()` on each.
    pub fn migrate(&mut self) {
        // Note: we take &mut self although the implementation doesn't need it, in order
        // to assert that this is meant to be invoked on a SendRc we own. Since SendRc
        // isn't Sync, this is not required for soundness.

        let this_thread = Self::current_thread();
        if self.is_pinned_to(Some(this_thread), Relaxed) {
            // Allow a call to migrate() from the pinned thread after the migration is
            // successfully finished. (Multiple calls to migrate() by the same SendRc are
            // also allowed during migration, since we use a HashSet.) This allows simpler
            // the code that traverses SendRcs.
            return;
        }
        let inner = self.inner();

        // Temporarily pin the allocation to an impossible ThreadId, thereby disabling its
        // use while migration is in progress. This prevents clones in the original thread
        // (if they weren't all transferred) to execute clone() and drop() while we're
        // running.
        inner.pinned_to.store(0, SeqCst);

        let migration_opt = &mut *inner.migration.lock();

        let done = {
            let migration = if let Some(migration) = migration_opt {
                // we're continuing a migration that started with another SendRc - just
                // check that we're still in the same thread
                if migration.new_thread != this_thread {
                    panic!("SendRc::<T>::migrate() invoked on the same T from different threads");
                }
                migration
            } else {
                // we're initiating migration for this allocation
                let clone_count = self.strong_count_unchecked();
                migration_opt.insert(Box::new(Migration {
                    new_thread: this_thread,
                    migrated_clones: HashSet::with_capacity(clone_count),
                    clone_count,
                }))
            };
            migration.migrated_clones.insert(self as *const _ as usize);
            migration.migrated_clones.len() == migration.clone_count
        };

        if done {
            *migration_opt = None;
            inner.pinned_to.store(this_thread, SeqCst);
        }
    }

    fn strong_count_unchecked(&self) -> usize {
        self.inner().strong_count.load(SeqCst)
    }

    /// Returns the number of pointers to this allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn strong_count(this: &Self) -> usize {
        this.assert_pinned_to(None, "strong_count", Relaxed);
        this.strong_count_unchecked()
    }

    /// Returns the inner value, if the `SendRc` has exactly one reference.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        this.assert_pinned_to(None, "try_unwrap", Relaxed);
        if this.strong_count_unchecked() == 1 {
            // Safety: refcount is 1, so it's just us, and the pointer was obtained using
            // Box::into_raw().
            let inner_box = unsafe { Box::from_raw(this.0.as_ptr()) };
            Ok(inner_box.val)
        } else {
            Err(this)
        }
    }

    /// Returns a mutable reference into the given `SendRc`, if there are no other `SendRc`
    /// pointers to the same allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        this.assert_pinned_to(None, "get_mut", Relaxed);
        if this.strong_count_unchecked() == 1 {
            Some(&mut this.inner_mut().val)
        } else {
            None
        }
    }

    /// Returns true if the two `SendRc`s point to the same allocation.
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.0 == other.0
    }
}

impl<T: Debug> Debug for SendRc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> Deref for SendRc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // We can read pinned_to with a Relaxed ordering because here we don't need to
        // worry about a race with migrate(). It's ok to load an older value of pinned_to
        // because it means migration hasn't finished (because we haven't participated in
        // it) and it's ok for deref to succeed. If we're called from some unrelated
        // thread, then this will fail regardless of which value of pinned_to we observe.
        self.assert_pinned_to(None, "deref", Relaxed);
        &self.inner().val
    }
}

impl<T> Clone for SendRc<T> {
    fn clone(&self) -> Self {
        let this_thread = Some(Self::current_thread());
        // Increment the count first, so that this clone is taken into account by a
        // migration that is possibly starting elsewhere.
        self.inner().strong_count.fetch_add(1, SeqCst);
        // Since we don't touch the value, this check is not needed for soundness here,
        // but to ensure that subsequent deref() of the clone (which fetches pinned_to
        // with Relaxed ordering) is sound.
        self.assert_pinned_to(this_thread, "clone", SeqCst);
        SendRc(self.0)
    }
}

impl<T> Drop for SendRc<T> {
    fn drop(&mut self) {
        let this_thread = Some(Self::current_thread());
        // Instead of panicking immediately, check whether we're in the correct thread and
        // leak the Rc if we're not. Then panic, but only if we're not already panicking,
        // because panic-inside-panic aborts the program and breaks unit tests.
        if self.is_pinned_to(this_thread, SeqCst) {
            // If a migration starts just before we decrement the count, it means the
            // migration works with a strong count that is by one too high and will never
            // finish. If it starts after we decrement the count, it will run to
            // completion, but in that case there is another SendRc, so old_refcnt won't
            // drop to 1 for us.
            let old_refcnt = self.inner().strong_count.fetch_sub(1, SeqCst);
            if old_refcnt == 1 {
                unsafe {
                    std::ptr::drop_in_place(self.0.as_ptr());
                }
            }
        } else if !std::thread::panicking() {
            panic!("SendRc::drop() attempted from incorrect thread; call migrate() first");
        }
    }
}

#[cfg(feature = "deepsize")]
impl<T> deepsize::DeepSizeOf for SendRc<T>
where
    T: deepsize::DeepSizeOf,
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.inner().val.deep_size_of_children(context)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Barrier};

    use parking_lot::Mutex;

    use super::SendRc;

    #[test]
    fn trivial() {
        let r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        *r1.borrow_mut() = 2;
        assert_eq!(*r2.borrow(), 2);
    }

    #[test]
    fn test_drop() {
        struct Payload(Rc<RefCell<bool>>);
        impl Drop for Payload {
            fn drop(&mut self) {
                *self.0.as_ref().borrow_mut() = true;
            }
        }
        let make = || {
            let is_dropped = Rc::new(RefCell::new(false));
            let payload = Payload(Rc::clone(&is_dropped));
            (SendRc::new(payload), is_dropped)
        };

        let (r1, is_dropped) = make();
        assert!(!*is_dropped.borrow());
        drop(r1);
        assert!(*is_dropped.borrow());

        let (r1, is_dropped) = make();
        let r2 = SendRc::clone(&r1);
        assert!(!*is_dropped.borrow());
        drop(r1);
        assert!(!*is_dropped.borrow());
        drop(r2);
        assert!(*is_dropped.borrow());

        let (r1, is_dropped) = make();
        let r2 = SendRc::clone(&r1);
        let r3 = SendRc::clone(&r1);
        assert!(!*is_dropped.borrow());
        drop(r1);
        assert!(!*is_dropped.borrow());
        drop(r2);
        assert!(!*is_dropped.borrow());
        drop(r3);
        assert!(*is_dropped.borrow());
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
    fn missing_migrate_drop() {
        let r = SendRc::new(RefCell::new(1));
        std::thread::spawn(move || {
            drop(r);
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn missing_migrate_deref() {
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
    fn missing_second_migrate() {
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
    fn incomplete_migration() {
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
    fn faked_migrate() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            // calling migrate twice on the same value won't fool us into thinking the
            // SendRc is fully migrated
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
                std::mem::forget(r1); // avoid another panic
            }
        });
        barrier.wait();
        // not allowed to proceed
        let _ = r2.clone();
    }

    #[test]
    fn incomplete_migration_use_migrated_ref() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        let state = Arc::new(Mutex::new(0));
        let result = std::thread::spawn({
            let state = Arc::clone(&state);
            move || {
                *state.lock() = 1;
                r1.migrate();
                *state.lock() = 2;
                let _ = &*r1;
                *state.lock() = 3;
                drop(r2);
                *state.lock() = 4;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock(), 2);
    }
}
