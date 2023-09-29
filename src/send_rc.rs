//! Reference-counting pointer like `Rc<T>`, but which is `Send` if `T` is `Send`.
//!
//! [`SendRc`] is handy if you use `Rc` to create an acyclic graph or a hierarchy with
//! cross-references, which you build and use from a single thread, but which you need to
//! occasionally move to another thread wholesale.
//!
//! It is different from `Arc` because the value can still be accessed from only one
//! thread at a time, and it is only allowed to manipulate it (by dropping or cloning)
//! from a single thread. This property makes it safely `Send` even when holding
//! non-`Sync` types like `RefCell`.

use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::thread_id::current_thread;

// Safety:
//
// We require `T: Send` to make SendRc<T> Send, so there is no impediment to moving T from
// another thread, i.e. from accessing it first in one thread then in another. It's
// accessing T from two different threads in parallel that must be prevented, since T is
// not guaranteed to be Sync. (E.g. RefCell<u32> would cause undefined behavior if it were
// shared among two threads.) SendRc prevents that from happening by checking at run-time
// that access to T consistently happens from a particular thread which the value is
// "pinned" to. By default a value and all the SendRcs are pinned to the thread T was
// allocated in by a call to SendRc::new(). Moving a SendRc to a different thread without
// re-pinning will cause any access to T to panic. (Manipulation of the reference count
// through clone() and drop() from a different thread will likewise result in panic.)
//
// To unpin SendRc from thread A and pin it to thread B, we go through steps that ensure
// that references to T created in thread A are relinquished before any reference to T can
// be created in thread B:
//
// * We require all pointers to a value to be "parked" before send. We use the reference
//   count to check that we've parked all pointers, and we disable clone and drop using
//   panic to make sure new ones haven't been created in the meantime.
// * park() takes &mut SendRc, so existing references to T won't outlive that call.
// * After parking a SendRc, it is no longer possible to obtain references to the value
//   through any of the pointers. The only remaining way to get a T& is by calling park()
//   and that reference is tied to the lifetime of PreSend (which is itself not Send, and
//   must be consumed to obtain a token that is Send).
// * To pin the value to thread B, the user must call PreSend::ready(), which checks that
//   all SendRcs have parked (so no T& obtained through SendRcs exist), and which consumes
//   the PreSend (so no T& returned by calls to park() exist). The token returned by
//   ready() is Send, and has a single method that pins the values that participated in
//   parking to the new thread.
unsafe impl<T> Send for SendRc<T> where T: Send {}

struct Inner<T> {
    // id of thread from which the value can be accessed
    pinned_to: AtomicU64,
    parking: Cell<u64>,
    val: T,
    strong_count: Cell<usize>,
}

static NEXT_SENDRC_ID: AtomicUsize = AtomicUsize::new(0);
static NEXT_PRE_SEND_ID: AtomicU64 = AtomicU64::new(1);

/// Reference-counting pointer like `Rc<T>`, but which is `Send` if `T` is `Send`.
///
/// When created, a `SendRc` is pinned to the current thread, and is usable only within
/// it. Before sending it to a different thread, you must use [`SendRc::pre_send()`] to
/// _park_ all the `SendRc`s that point to the same value. This will make the values
/// temporarily inaccessible through `SendRc`s, but will allow safe transfer.  When done
/// parking, you can obtain a `PostSend` and send it to the other thread to restore access
/// to the values there. For example:
///
/// ```
/// # use std::cell::RefCell;
/// # use sendable::SendRc;
/// // create two SendRcs pointing to a shared value
/// let mut r1 = SendRc::new(RefCell::new(1));
/// let mut r2 = SendRc::clone(&r1);
///
/// // prepare to send them to a different thread
/// let pre_send = SendRc::pre_send();
/// pre_send.park(&mut r1); // r1 and r2 cannot be dereferenced from this point
/// pre_send.park(&mut r2);
/// // ready() would panic if there were unparked SendRcs pointing to the value
/// let post_send = pre_send.ready();
///
/// // move everything to a different thread
/// std::thread::spawn(move || {
///     // SendRcs are still unusable until unparked
///     post_send.unpark();
///     // they're again usable from this point, and only in this thread
///     *r1.borrow_mut() += 1;
///     assert_eq!(*r2.borrow(), 2);
/// })
/// # .join().unwrap();
/// ```
///
/// This process may be repeated to send the `SendRc`s to another thread later.
///
/// Compared to `Rc`, tradeoffs of a `SendRc` are:
///
/// * `deref()`, `clone()`, and `drop()` require a check that the shared value is not
///    parked, and a check that we're accessing it from the correct thread.
/// * a `SendRc` takes up two machine words.
/// * it currently doesn't support weak pointers.
pub struct SendRc<T> {
    ptr: NonNull<Inner<T>>,
    // Assign each SendRc an unchanging id, used to track which ones have participated in
    // migration. If malicious code forces wrap-around of the id on a 32-bit architecture,
    // we're still sound because passing two SendRcs with the same id to `PreSend::park()`
    // will just cause `PreSend::ready()` to always fail and prevent migration.
    id: usize,
}

enum PinError {
    BadThread,
    Parking,
}

impl PinError {
    fn msg(&self) -> &'static str {
        match self {
            PinError::BadThread => "SendRc accessed from wrong thread; call pre_send() first",
            PinError::Parking => "access to SendRc that is about to be sent to a new thread",
        }
    }

    fn panic(&self, what: &str) -> ! {
        panic!("{what}: {}", self.msg());
    }
}

impl<T> SendRc<T> {
    /// Constructs a new `SendRc<T>`.
    ///
    /// The `SendRc` is only usable from the current thread. To send and use it in another
    /// thread, you must call [`pre_send()`](SendRc::pre_send).
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(Inner {
            pinned_to: AtomicU64::new(current_thread()),
            parking: Cell::new(0),
            val,
            strong_count: Cell::new(1),
        }));
        SendRc::from_inner_ptr(ptr)
    }

    fn from_inner_ptr(ptr: *mut Inner<T>) -> Self {
        SendRc {
            // unwrap: we have a valid box, its pointer is not null (rustc eliminates the
            // check, https://godbolt.org/z/dsYPxxMWo)
            ptr: NonNull::new(ptr).unwrap(),
            id: NEXT_SENDRC_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn inner(&self) -> &Inner<T> {
        // Safety: Shared value is valid at least as long as self
        unsafe { self.ptr.as_ref() }
    }

    // Safety: requires reference count of 1
    unsafe fn inner_mut(&mut self) -> &mut Inner<T> {
        self.ptr.as_mut()
    }

    #[inline]
    fn check_pinned(&self) -> Result<(), PinError> {
        let inner = self.inner();
        if inner.pinned_to.load(Ordering::Relaxed) != current_thread() {
            return Err(PinError::BadThread);
        }
        if inner.parking.get() != 0 {
            return Err(PinError::Parking);
        }
        Ok(())
    }

    #[inline]
    fn assert_pinned(&self, op: &str) {
        self.check_pinned()
            .unwrap_or_else(|pinerr| pinerr.panic(op));
    }

    /// Prepare to send `SendRc`s to another thread.
    ///
    /// To move a `SendRc` to a different thread, you must call [`park()`](PreSend::park)
    /// on that pointer, as well as on all other `SendRc`s pointing to the same value.
    ///
    /// ```
    /// # use std::cell::RefCell;
    /// # use sendable::SendRc;
    /// let mut r1 = SendRc::new(RefCell::new(1));
    /// let mut r2 = SendRc::clone(&r1);
    /// let pre_send = SendRc::pre_send();
    /// pre_send.park(&mut r1);
    /// pre_send.park(&mut r2);
    /// let post_send = pre_send.ready();
    /// // post_send, r1, and r2 can now be send to a different thread, and re-enabled
    /// // by calling post_send.unpark()
    /// # post_send.unpark(); // avoid panic in doctest
    /// ```
    pub fn pre_send() -> PreSend<T> {
        PreSend {
            parked: Default::default(),
            pre_send_id: NEXT_PRE_SEND_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Returns the number of `SendRc`s pointing to the value.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last pinned to.
    pub fn strong_count(this: &Self) -> usize {
        this.assert_pinned("SendRc::strong_count()");
        this.inner().strong_count.get()
    }

    /// Returns the value if the `SendRc` has exactly one reference.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last pinned to.
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        this.assert_pinned("SendRc::try_unwrap()");
        if this.inner().strong_count.get() == 1 {
            // Safety: refcnt == 1, so it's just us
            let inner_box = unsafe { Box::from_raw(this.ptr.as_ptr()) };
            Ok(inner_box.val)
        } else {
            Err(this)
        }
    }

    /// Returns a mutable reference to the value `this` points to, if no other `SendRc`s
    /// point to the same value.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last pinned to.
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        this.assert_pinned("SendRc::get_mut()");
        if this.inner().strong_count.get() == 1 {
            // Safety: we've checked that refcnt == 1, and it can't increase because
            // this.clone() doesn't compile while the returned &mut T is live.
            unsafe { Some(&mut this.inner_mut().val) }
        } else {
            None
        }
    }

    /// Returns true if `this` and `other` point to the same value.
    ///
    /// This method can be called from any thread.
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }
}

/// Registry for parking `SendRc`s so they can be sent to another thread.
///
/// This type is not `Send`; when finished with calls to `park()`, invoke `ready()`
/// to obtain a `Send` token that can be moved to the other thread to re-enable the
/// `SendRc`s.
pub struct PreSend<T> {
    parked: RefCell<HashMap<usize, NonNull<Inner<T>>>>,
    pre_send_id: u64,
}

impl<T> PreSend<T> {
    /// Park the value pointed to by `send_rc`.
    ///
    /// Parking a value makes it inaccessible through this or any other `SendRc` that
    /// points to it. Attempts to dereference, clone, or drop either this `SendRc` or any
    /// other that points to the same value will trigger a panic. Additionally, `park()`
    /// registers `send_rc` in particular as having participated in the parking.
    ///
    /// To send a `SendRc` to a different thread, `park()` must be invoked on all the
    /// `SendRc`s that point to the value.
    ///
    /// It is allowed to park `SendRc`s pointing to different values of the same type in
    /// the same `PreSend`. Parking a `SendRc` that was already parked in the same
    /// `PreSend` is a no-op.
    ///
    /// Returns a reference to the underlying value, which may be used to visit additional
    /// `SendRc`s that are only reachable through the value.
    ///
    /// Panics when invoked from a thread different than the one `send_rc` was created in
    /// or last pinned to. Also panics when passed a `send_rc` that was already parked by
    /// a different `PreSend`.
    pub fn park<'a>(&'a self, send_rc: &'a mut SendRc<T>) -> &'a T {
        match send_rc.check_pinned() {
            Ok(()) => send_rc.inner().parking.set(self.pre_send_id),
            Err(pinerr @ PinError::BadThread) => pinerr.panic("PreSend::park()"),
            Err(PinError::Parking) => {
                // Allowing park() from a different PreSend would be unsound, see
                // same_sendrc_different_presend.
                if send_rc.inner().parking.get() != self.pre_send_id {
                    panic!("PreSend::park(): call from different PreSend");
                }
            }
        }
        self.parked.borrow_mut().insert(send_rc.id, send_rc.ptr);
        &send_rc.inner().val
    }

    /// Checks that there remain no unparked `SendRc`s pointing to values whose `SendRc`s
    /// were parked by this `PreSend`, and returns a [`PostSend`] that can pin them to
    /// another thread.
    ///
    /// At the point of invocation of `ready()`, the compiler will statically verify that
    /// there are no outstanding references to the data pointed to by `SendRc`s parked by
    /// this `PostSend`.
    ///
    /// Panics if the above check fails, i.e. if [`is_ready()`](PreSend::is_ready) would
    /// return false.
    pub fn ready(self) -> PostSend<T> {
        if !self.is_ready() {
            panic!("PreSend::ready() called before all SendRcs have been parked");
        }
        let ptrs: HashSet<_> = self.parked.into_inner().into_values().collect();
        // Pin shared values to a non-existent thread id, so that enable() can detect the
        // new thread from which we are called (even if that new thread is the current
        // thread again).
        for &ptr in &ptrs {
            // Safety: ptr belongs to at least one SendRc, so it's valid
            let inner = unsafe { &*ptr.as_ptr() };
            inner.pinned_to.store(0, Ordering::Relaxed);
        }
        PostSend { ptrs }
    }

    /// Returns true if there remain no unparked `SendRc`s pointing to values whose
    /// `SendRc`s were parked by this `PreSend`.
    ///
    /// If this returns true, it means [`ready()`](PreSend::ready) will succeed.
    ///
    /// For example:
    /// ```
    /// # use std::cell::RefCell;
    /// # use sendable::SendRc;
    /// let mut r1 = SendRc::new(RefCell::new(1));
    /// let mut r2 = SendRc::clone(&r1);
    /// let mut q1 = SendRc::new(RefCell::new(1));
    /// let mut q2 = SendRc::clone(&q1);
    /// let pre_send = SendRc::pre_send();
    /// pre_send.park(&mut r1);
    /// assert!(!pre_send.is_ready()); // r2 still unparked
    /// pre_send.park(&mut r2);
    /// assert!(pre_send.is_ready());  // r1/r2 shared value fully parked, q1/q2 not involved
    /// pre_send.park(&mut q1);
    /// assert!(!pre_send.is_ready()); // r1/r2 ok, but q2 unparked
    /// pre_send.park(&mut q2);
    /// assert!(pre_send.is_ready());  // both r1/r2 and q1/q2 shared values fully parked
    /// # pre_send.ready().unpark();
    /// ```
    pub fn is_ready(&self) -> bool {
        // Count how many `SendRc`s point to each shared value.
        let ptr_sendrc_cnt: HashMap<_, usize> =
            self.parked
                .borrow()
                .values()
                .fold(HashMap::new(), |mut map, &ptr| {
                    *map.entry(ptr).or_default() += 1;
                    map
                });
        ptr_sendrc_cnt.into_iter().all(|(ptr, cnt)| {
            // Safety: ptr is valid because there is at least 1 SendRc containing it. We
            // may access it from this thread because PreSend isn't Send.
            let inner = unsafe { &*ptr.as_ptr() };
            cnt == inner.strong_count.get()
        })
    }

    /// Describes the park status of `send_rc` and the value it points to.
    ///
    /// This is useful for:
    ///
    /// * detecting a `SendRc` that was already visited while traversing a graph of
    ///   `SendRc`s (`sendrc_parked`)
    /// * detecting whether the value behind this `SendRc` is unreachable because one or
    ///   more `SendRc`s pointing to it have been parked (`value_parked`)
    ///
    /// ```
    /// # use std::cell::RefCell;
    /// # use sendable::SendRc;
    /// let mut r1 = SendRc::new(RefCell::new(1));
    /// let mut r2 = SendRc::clone(&r1);
    /// let pre_send = SendRc::pre_send();
    /// pre_send.park(&mut r1);
    /// assert!(pre_send.park_status_of(&r1).sendrc_parked);  // r1 is parked
    /// assert!(!pre_send.park_status_of(&r2).sendrc_parked); // r2 is not yet parked
    /// assert!(pre_send.park_status_of(&r1).value_parked);   // the underlying value is parked
    /// assert!(pre_send.park_status_of(&r2).value_parked);   // the underlying value is parked
    /// # pre_send.park(&mut r2);
    /// # pre_send.ready().unpark();
    /// ```
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last pinned to.
    pub fn park_status_of(&self, send_rc: &SendRc<T>) -> ParkStatus {
        let (mut sendrc_parked, mut value_parked) = (false, false);
        match send_rc.check_pinned() {
            Ok(()) => {}
            Err(PinError::Parking) => {
                value_parked = true;
                sendrc_parked = self.parked.borrow().contains_key(&send_rc.id);
            }
            Err(pinerr @ PinError::BadThread) => pinerr.panic("PreSend::park_status_of()"),
        }
        ParkStatus {
            sendrc_parked,
            value_parked,
        }
    }
}

/// Value returned by [`PreSend::park_status_of()`].
pub struct ParkStatus {
    /// True if the `SendRc` passed to `park_status_of()` has been parked.
    pub sendrc_parked: bool,
    /// True if at least one `SendRc` that points to the same value as `SendRc` passed to
    /// `park_status_of()` has been parked.
    pub value_parked: bool,
}

/// Token for pinning the `SendRc`s parked with a `PreSend` to another thread.
///
/// The pinning is effected with a call to [`unpark()`](PostSend::unpark()).
///
/// Since `PostSend` can only be obtained via [`PreSend::ready()`], possessing a
/// `PostSend` serves as proof that all `SendRc`s pointing to the shared values involved in
/// the move have been parked.
#[must_use]
pub struct PostSend<T> {
    ptrs: HashSet<NonNull<Inner<T>>>,
}

// Safety: pointers to shared values can be sent to a new thread because PreSend::ready()
// checked that all their SendRcs have been parked.
unsafe impl<T> Send for PostSend<T> where T: Send {}

impl<T> PostSend<T> {
    /// Unpark `SendRc`s previously parked by a `PreSend` and pin the values they point to
    /// to the current thread.
    ///
    /// This re-enables all the pointers involved in the migration and makes their data
    /// accessible from this (and only this) thread.
    pub fn unpark(self) {
        let current_thread = current_thread();
        for ptr in self.ptrs {
            // Safety: ptr belongs to at least one SendRc, so it's valid
            let inner = unsafe { &*ptr.as_ptr() };
            inner.pinned_to.store(current_thread, Ordering::Relaxed);
            inner.parking.set(0);
        }
    }
}

impl<T: Display> Display for SendRc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&**self, f)
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
        self.assert_pinned("SendRc::deref()");
        &self.inner().val
    }
}

impl<T> Clone for SendRc<T> {
    fn clone(&self) -> Self {
        self.assert_pinned("SendRc::clone()");
        self.inner()
            .strong_count
            .set(self.inner().strong_count.get() + 1);
        SendRc::from_inner_ptr(self.ptr.as_ptr())
    }
}

impl<T> Drop for SendRc<T> {
    fn drop(&mut self) {
        if let Err(pinerr) = self.check_pinned() {
            // Instead of asserting that we're pinned, check it, and panic only if we're
            // not already panicking.  Panic-inside-panic aborts the program, making it
            // harder to debug and breaking unit tests.
            if std::thread::panicking() {
                return;
            }
            pinerr.panic("SendRc::drop()");
        }
        // the pin is ok - proceed with the drop
        let refcnt = self.inner().strong_count.get();
        if refcnt == 1 {
            // safety: dropping Rc with refcnt == 1, it's just us
            unsafe {
                // Dropping the Box drops the value and deallocates the heap storage.
                // Weak refs would require separating them by wrapping the value in
                // ManuallyDrop<T> and calling `inner().val.drop()` here. Dropping the box
                // would be done only once the weak count drops to 0.
                let inner_box = Box::from_raw(self.ptr.as_ptr());
                drop(inner_box);
            }
        } else {
            self.inner().strong_count.set(refcnt - 1);
        }
    }
}

impl<T> AsRef<T> for SendRc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T> Borrow<T> for SendRc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: Default> Default for SendRc<T> {
    fn default() -> SendRc<T> {
        SendRc::new(Default::default())
    }
}

impl<T: Eq> Eq for SendRc<T> {}

impl<T: PartialEq> PartialEq for SendRc<T> {
    fn eq(&self, other: &SendRc<T>) -> bool {
        SendRc::ptr_eq(self, other) || **self == **other
    }
}

impl<T: PartialOrd> PartialOrd for SendRc<T> {
    fn partial_cmp(&self, other: &SendRc<T>) -> Option<std::cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: Ord> Ord for SendRc<T> {
    fn cmp(&self, other: &SendRc<T>) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: Hash> Hash for SendRc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use super::SendRc;

    #[test]
    fn trivial() {
        let r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        *r1.borrow_mut() = 2;
        assert_eq!(*r2.borrow(), 2);
    }

    #[test]
    fn drops() {
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
        let pre_send = SendRc::pre_send();
        pre_send.park(&mut r1);
        pre_send.park(&mut r2);
        let post_send = pre_send.ready();
        std::thread::spawn(move || {
            post_send.unpark();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic = "drop()"]
    fn missing_pre_send_drop() {
        let r = SendRc::new(RefCell::new(1));
        std::thread::spawn(move || {
            drop(r);
        })
        .join()
        .map_err(|e| e.downcast::<String>().unwrap())
        .unwrap();
    }

    #[test]
    #[should_panic = "deref()"]
    fn missing_pre_send_deref() {
        let r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        std::thread::spawn(move || {
            *r1.borrow_mut() = 2; // this should panic
            assert_eq!(*r2.borrow(), 2);
        })
        .join()
        .map_err(|e| e.downcast::<String>().unwrap())
        .unwrap();
    }

    #[test]
    #[should_panic = "ready() called before"]
    fn incomplete_pre_send() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let pre_send = SendRc::pre_send();
        pre_send.park(&mut r1);
        let _ = pre_send.ready(); // panics because we didn't park _r2
    }

    #[test]
    #[should_panic = "before all SendRcs have been parked"]
    fn incomplete_pre_send_other_shared_value() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut q1 = SendRc::new(RefCell::new(1));
        let _q2 = SendRc::clone(&q1);
        let pre_send = SendRc::pre_send();
        pre_send.park(&mut r1);
        pre_send.park(&mut r2);
        pre_send.park(&mut q1);
        let _ = pre_send.ready(); // _q2 is missing
    }

    #[test]
    #[should_panic = "before all SendRcs have been parked"]
    fn faked_pre_send_count_reusing_same_ptr() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let pre_send = SendRc::pre_send();
        // parking the same SendRc twice won't fool us into thinking all SendRcs were
        // parked
        pre_send.park(&mut r1);
        pre_send.park(&mut r1);
        let _ = pre_send.ready();
    }

    #[test]
    #[should_panic = "call from different PreSend"]
    fn same_sendrc_different_presend() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let pre_send1 = SendRc::pre_send();
        pre_send1.park(&mut r1);
        pre_send1.park(&mut r2);
        let pre_send2 = SendRc::pre_send();
        let _ref1: &RefCell<u32> = pre_send2.park(&mut r1); // this must panic
        let post_send = pre_send1.ready();
        // if the above didn't panic, the code below would run and be UB
        let t = std::thread::spawn(move || {
            post_send.unpark();
            let _ref2: &RefCell<u32> = &*r2;
            //*ref2.borrow_mut() += 1; // data race with ref1
        });
        //*ref1.borrow_mut() += 1; // data race with ref2
        t.join().unwrap();
    }

    #[test]
    fn park_twice_good() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::new(RefCell::new(1));
        let pre_send = SendRc::pre_send();
        pre_send.park(&mut r1);
        pre_send.park(&mut r1);
        pre_send.park(&mut r2);
        let post_send = pre_send.ready();
        post_send.unpark();
    }

    #[test]
    fn park_twice_bad() {
        let state = Arc::new(Mutex::new(0));
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                let mut r1 = SendRc::new(RefCell::new(1));
                let mut r2 = SendRc::new(RefCell::new(1));
                let pre_send1 = SendRc::pre_send();
                let pre_send2 = SendRc::pre_send();
                *state.lock().unwrap() = 1;
                pre_send1.park(&mut r1);
                *state.lock().unwrap() = 2;
                pre_send1.park(&mut r2);
                *state.lock().unwrap() = 3;
                pre_send2.park(&mut r2); // panic
                *state.lock().unwrap() = 4;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock().unwrap(), 3);
    }


    #[derive(Default, Debug, Eq, PartialEq)]
    struct ComplexValueToBeDropped {
    }

    impl Drop for ComplexValueToBeDropped{
        fn drop(&mut self) {
            println!("Dropping complex value");
        }
    }

    #[derive(Default, Debug, Eq, PartialEq)]
    struct StructContainingReferenceToComplexValueToBeDropped {
        reference_to_complex_value_to_be_dropped:  SendRc<RefCell<ComplexValueToBeDropped>>
    }

    impl Drop for StructContainingReferenceToComplexValueToBeDropped{
        fn drop(&mut self) {
            println!("Dropping struct containing reference to complex value");
        }
    }

    #[test]
    fn test_send_rc_drop() {

        // create two SendRcs pointing to a shared value
        let mut r1: SendRc<RefCell<ComplexValueToBeDropped>> = SendRc::new(RefCell::new(ComplexValueToBeDropped::default()));
        let mut r2 = SendRc::clone(&r1);

// prepare to send them to a different thread
        let pre_send = SendRc::pre_send();
        pre_send.park(&mut r1); // r1 and r2 cannot be dereferenced from this point
        pre_send.park(&mut r2);
// ready() would panic if there were unparked SendRcs pointing to the value
        let post_send = pre_send.ready();

        let parent_struct = StructContainingReferenceToComplexValueToBeDropped {
            reference_to_complex_value_to_be_dropped: r1
        };

// move everything to a different thread
        let jh = std::thread::spawn(move || {
            // SendRcs are still unusable until unparked
            post_send.unpark();
            // they're again usable from this point, and only in this thread
            assert_eq!(*r2.borrow(), ComplexValueToBeDropped::default());
        });

        let jh2 = std::thread::spawn(move || {
            // SendRcs are still unusable until unparked
            println!("{:?}",parent_struct)
        });

        jh.join().unwrap();
        jh2.join().unwrap();
    }
}
