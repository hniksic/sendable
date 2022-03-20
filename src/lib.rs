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
//! thread. After moving `SendRc` to a different thread, you must invoke `sent()` to mark
//! that the value has been migrated. Only after all the clones of a `SendRc` have been
//! thus marked will access to the inner value (and to `SendRc::clone()` and
//! `SendRc::drop()`) be allowed.

#![warn(missing_docs)]

use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

struct Wrap<T> {
    allowed_thread: AtomicU64,
    thread_send: Mutex<Option<Box<ThreadMove>>>,
    val: T,
}

struct ThreadMove {
    new_thread: u64,
    sent_clones: HashSet<usize>,
}

/// Wrapper around `Rc<T>` that is `Send` if `T` is `Send`.
///
/// ```
/// # use std::cell::RefCell;
/// # use send_rc::SendRc;
/// let mut r1 = SendRc::new(RefCell::new(1));
/// let mut r2 = SendRc::clone(&r1);
/// std::thread::spawn(move || {
///     r1.sent();
///     r2.sent();
///     *r1.borrow_mut() += 1;
///     assert_eq!(*r2.borrow(), 2);
/// })
/// .join()
/// .unwrap();
/// ```
pub struct SendRc<T>(ManuallyDrop<Rc<Wrap<T>>>);

// Safety: safe because we don't allow access to Rc::clone(), Rc::drop(), and Rc::deref()
// except from the thread they were migrated in or the thread they were created in.
unsafe impl<T> Send for SendRc<T> where T: Send {}

impl<T> SendRc<T> {
    /// Constructs a new SendRc<T>
    pub fn new(val: T) -> Self {
        SendRc(ManuallyDrop::new(Rc::new(Wrap {
            val,
            allowed_thread: AtomicU64::new(Self::current_thread()),
            thread_send: Mutex::new(None),
        })))
    }

    fn current_thread() -> u64 {
        unsafe { std::mem::transmute(std::thread::current().id()) }
    }

    fn check_thread(&self) -> bool {
        self.0.allowed_thread.load(Ordering::Relaxed) == Self::current_thread()
    }

    fn check_thread_panic(&self) {
        if !self.check_thread() {
            panic!("access from wrong thread");
        }
    }

    /// Mark this SendRc as sent from another thread.
    ///
    /// This SendRc and all its clones will remain unusable (deref, drop, and clone will
    /// panic) until sent() has been called on the remaining clones.
    ///
    /// Calling sent() on clones of the same SendRc from different threads will also
    /// result in panic.
    pub fn sent(&mut self) {
        // Note: we take &mut self although the implementation doesn't need it, in order
        // to assert that this is meant to be invoked on a SendRc we own. Since SendRc
        // isn't Sync, this is not required for soundness.

        let this_thread = Self::current_thread();

        // 0 is not a valid ThreadId, so this disables further clones, drops, and derefs
        // until all clones of this Rc have been sent.
        self.0.allowed_thread.store(0, Ordering::Relaxed);

        let thread_send = &mut *self.0.thread_send.lock();

        let done = {
            // we only care about the strong count because SendRc doesn't expose weak refs
            let clones_total = Rc::strong_count(&self.0);
            let thread_send = if let Some(thread_send) = thread_send {
                if thread_send.new_thread != this_thread {
                    panic!("SendRc::sent() invoked from different threads");
                }
                thread_send
            } else {
                thread_send.insert(Box::new(ThreadMove {
                    new_thread: this_thread,
                    sent_clones: HashSet::with_capacity(clones_total),
                }))
            };
            thread_send.sent_clones.insert(self as *const _ as usize);
            thread_send.sent_clones.len() == clones_total
        };

        if done {
            *thread_send = None;
            self.0.allowed_thread.store(this_thread, Ordering::Relaxed);
        }
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
    use super::SendRc;
    use std::cell::RefCell;

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
            r1.sent();
            r2.sent();
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
            r1.sent();
            r2.sent();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
            (r1, r2)
        })
        .join()
        .unwrap();
        r1.sent();
        r2.sent();
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
            r1.sent();
            r2.sent();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
            (r1, r2)
        })
        .join()
        .unwrap();
        // here we must call sent() as well
        assert_eq!(*r2.borrow(), 2);
    }

    #[test]
    #[should_panic]
    fn incomplete_send() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            r1.sent();
            let _ = &*r2;
        })
        .join()
        .unwrap();
    }
}
