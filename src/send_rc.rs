//! Reference-counting pointer like `Rc<T>`, but which is `Send` if `T` is `Send`.
//!
//! [`SendRc`] is handy if you use `Rc` to create an acyclic graph or a hierarchy with
//! cross-references, which you build and use from a single thread, but which you need to
//! occasionally move to another thread wholesale.
//!
//! It is different from `Arc` because the value can still be accessed from only one
//! thread at a time, and it is only allowed to manipulate it (by dropping or cloning)
//! from a single thread. This property makes it `Send` even when holding non-`Sync` types
//! like `RefCell`.

use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::thread_id::current_thread;

struct Inner<T> {
    // id of thread from which the value can be accessed
    pinned_to: AtomicU64,
    marking: Cell<u64>,
    val: T,
    strong_count: Cell<usize>,
}

static NEXT_SENDRC_ID: AtomicUsize = AtomicUsize::new(0);
static NEXT_PRE_SEND_ID: AtomicU64 = AtomicU64::new(1);

/// Reference-counting pointer like `Rc<T>`, but which is `Send` if `T` is `Send`.
///
/// After a `SendRc` is created, it is pinned to the current thread, and usable only in
/// that thread. When sending a `SendRc` to different thread, you must call
/// `SendRc::pre_send()` and use the returned `PreSend` handle to mark all the `SendRc`s
/// that point to the same allocation with a call to
/// [`pre_send.mark_send()`](PreSend::mark_send) on each. Then you can obtain the
/// `post_send` token with [`pre_send.ready()`](PreSend::ready), and reenable them by
/// calling [`post_send.sent()`](PostSend::sent). They may later be sent to a different
/// thread using the same process.
///
/// ```
/// # use std::cell::RefCell;
/// # use sendable::SendRc;
/// // create two SendRcs pointing to the same allocation
/// let mut r1 = SendRc::new(RefCell::new(1));
/// let mut r2 = SendRc::clone(&r1);
///
/// // prepare to ship them off to a different thread
/// let pre_send = SendRc::pre_send();
/// pre_send.mark_send(&mut r1); // r1 and r2 cannot be dereferenced from this point
/// pre_send.mark_send(&mut r2);
/// // ready() would panic if there were unmarked SendRcs pointing to the allocation
/// let mut post_send = pre_send.ready();
///
/// // move everything to a different thread
/// std::thread::spawn(move || {
///     // both pointers are unusable here
///     post_send.sent(); // both are usable from this point
///     *r1.borrow_mut() += 1;
///     assert_eq!(*r2.borrow(), 2);
/// })
/// .join()
/// .unwrap();
/// ```
///
/// Compared to `Rc`, tradeoffs are:
///
/// * `deref()`, `clone()`, and `drop()` requires a check that the allocation is not
///    marked for sending, and a check that we're accessing it from the correct thread.
/// * takes up two machine words.
/// * doesn't support weak pointers (though such support could be added).
pub struct SendRc<T> {
    ptr: NonNull<Inner<T>>,
    // Associate an non-changing id with each pointer so that we can track how many have
    // participated in migration. If malicious code forces wrap-around of the id, we're
    // still sound because passing two SendRcs with the same id to `PreSend::mark_send()`
    // will just cause `PreSend::ready()` to always fail and prevent migration.
    id: usize,
}

// Safety: SendRc can be sent between threads because we prohibit access to clone, drop,
// and deref except from the thread they are pinned to. Access is granted only after all
// pointers to the same allocation have been migrated to the new thread, which is why we
// can avoid requiring T: Sync.
unsafe impl<T> Send for SendRc<T> where T: Send {}

enum PinCheck {
    Ok,
    BadThread,
    Marking,
}

impl PinCheck {
    fn errmsg(&self) -> &'static str {
        match self {
            PinCheck::Ok => "no error",
            PinCheck::BadThread => "SendRc accessed from wrong thread; call pre_send() first",
            PinCheck::Marking => "access to SendRc that is about to be sent to a new thread",
        }
    }
}

impl<T> SendRc<T> {
    /// Constructs a new `SendRc<T>`.
    ///
    /// The newly created `SendRc` is only usable from the current thread. To send it to
    /// another thread, you must call `pre_send()`, mark it, and re-enable it in the
    /// new thread.
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(Inner {
            pinned_to: AtomicU64::new(current_thread()),
            marking: Cell::new(0),
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
        // Safety: Allocation is valid at least as long as self
        unsafe { self.ptr.as_ref() }
    }

    // Safety: requires reference count of 1
    unsafe fn inner_mut(&mut self) -> &mut Inner<T> {
        self.ptr.as_mut()
    }

    #[inline]
    fn check_pinned(&self) -> PinCheck {
        let inner = self.inner();
        if inner.pinned_to.load(Ordering::Relaxed) != current_thread() {
            return PinCheck::BadThread;
        }
        if inner.marking.get() != 0 {
            return PinCheck::Marking;
        }
        PinCheck::Ok
    }

    fn assert_pinned(&self, op: &str) {
        match self.check_pinned() {
            PinCheck::Ok => {}
            check @ (PinCheck::BadThread | PinCheck::Marking) => {
                panic!("{op}: {}", check.errmsg());
            }
        }
    }

    /// Prepare to send `SendRc`s of this type to a different thread.
    ///
    /// Before moving a `SendRc` to a different thread, you must call
    /// [`mark_send()`](PreSend::mark_send) on that pointer, as well as on all other
    /// `SendRc`s pointing to the same allocation.
    ///
    /// ```
    /// # use std::cell::RefCell;
    /// # use sendable::SendRc;
    /// let mut r1 = SendRc::new(RefCell::new(1));
    /// let mut r2 = SendRc::clone(&r1);
    /// let pre_send = SendRc::pre_send();
    /// pre_send.mark_send(&mut r1);
    /// pre_send.mark_send(&mut r2);
    /// let mut post_send = pre_send.ready();
    /// // post_send, r1, and r2 can now be send to a different thread, and re-enabled
    /// // by calling post_send.sent()
    /// # post_send.sent(); // avoid panic in doctest
    /// ```
    pub fn pre_send() -> PreSend<T> {
        PreSend {
            marked: Default::default(),
            pre_send_id: NEXT_PRE_SEND_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Prepare to send a fixed collection of `SendRc`s to a different thread.
    ///
    /// Calls `SendRc::pre_send()`, followed by `mark_send()` on the provided `SendRc`s,
    /// and finally returns the post-send token returned by the call to
    /// `pre_send.ready()`. Useful for one-shot move of `SendRc`s which are easily
    /// traversable.
    ///
    /// Panics if there are allocations pointed to by `SendRc`s in `all` which have extra
    /// `SendRc`s not included in `all`.
    pub fn pre_send_ready<'a>(all: impl IntoIterator<Item = &'a mut Self>) -> PostSend<T>
    where
        T: 'a,
    {
        let pre_send = Self::pre_send();
        for send_rc in all {
            pre_send.mark_send(send_rc);
        }
        pre_send.ready()
    }

    /// Returns the number of `SendRc`s pointing to this allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn strong_count(this: &Self) -> usize {
        this.assert_pinned("SendRc::strong_count()");
        this.inner().strong_count.get()
    }

    /// Returns the inner value, if the `SendRc` has exactly one reference.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        this.assert_pinned("SendRc::try_unwrap()");
        if this.inner().strong_count.get() == 1 {
            // Safety: refcount is 1, so it's just us, and the pointer was obtained using
            // Box::into_raw().
            let inner_box = unsafe { Box::from_raw(this.ptr.as_ptr()) };
            Ok(inner_box.val)
        } else {
            Err(this)
        }
    }

    /// Returns a mutable reference into the given `SendRc`, if there no other `SendRc`s
    /// point to the same allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        this.assert_pinned("SendRc::get_mut()");
        if this.inner().strong_count.get() == 1 {
            // Safety: we've checked that refcount is 1
            unsafe { Some(&mut this.inner_mut().val) }
        } else {
            None
        }
    }

    /// Returns true if this and the other `SendRc`s point to the same allocation.
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }
}

/// Handle for marking the `SendRc`s before they are sent to another thread.
///
/// This handle is not `Send`; when done with calls to `mark_send()`, invoke `ready()` to
/// obtain a token that can be moved to the other thread to re-enable the `SendRc`s.
pub struct PreSend<T> {
    marked: RefCell<HashMap<usize, NonNull<Inner<T>>>>,
    pre_send_id: u64,
}

impl<T> PreSend<T> {
    /// Make `send_rc` temporarily unusable so it can be sent to another thread.
    ///
    /// After this call it is no longer possible to deref, clone, or drop either this
    /// `SendRc` or any other that points to the same allocation. However, the value of
    /// the allocation is reachable through the return value of this method, which may be
    /// called again on the same `SendRc`. This allows reaching other `SendRc`s of the
    /// same allocation through this `SendRc`.
    ///
    /// It is allowed to mark `SendRc`s pointing to different allocations. After calling
    /// `mark_send()` on a `SendRc` pointing to as-yet-unmarked allocation, you must also
    /// invoke it on all other `SendRc`s pointing to that allocation prior to invoking
    /// `ready()`.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to. Also panics when passed a `send_rc` that was already
    /// marked by a different `PreSend` handle.
    pub fn mark_send<'a>(&'a self, send_rc: &'a mut SendRc<T>) -> &'a T {
        match send_rc.check_pinned() {
            PinCheck::Ok => {
                send_rc.inner().marking.set(self.pre_send_id);
            }
            check @ PinCheck::BadThread => panic!("PreSend::mark_send(): {}", check.errmsg()),
            PinCheck::Marking => {
                // Don't allow mark_send() from a rogue PreSend because we return a
                // reference tied to the lifetime of self, and depend on self being
                // consumed before it's sent to the new thread.
                if send_rc.inner().marking.get() != self.pre_send_id {
                    panic!("PreSend::mark_send(): call from different PreSend");
                }
            }
        }
        self.marked.borrow_mut().insert(send_rc.id, send_rc.ptr);
        &send_rc.inner().val
    }

    /// Returns true if the `send_rc` was already marked for sending by this `PreSend`
    /// handle.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn is_marked(&self, send_rc: &SendRc<T>) -> bool {
        match send_rc.check_pinned() {
            PinCheck::Ok => false,
            PinCheck::Marking => self.marked.borrow().contains_key(&send_rc.id),
            check @ PinCheck::BadThread => panic!("PreSend::is_marked(): {}", check.errmsg()),
        }
    }

    /// Returns true if the allocation this `send_rc` points to was marked for sending to
    /// another thread.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn is_allocation_marked(&self, send_rc: &SendRc<T>) -> bool {
        match send_rc.check_pinned() {
            PinCheck::Ok => false,
            PinCheck::Marking => true,
            check @ PinCheck::BadThread => panic!("PreSend::is_marked(): {}", check.errmsg()),
        }
    }

    /// Returns true if there are no allocations whose `SendRc`s were passed to
    /// `mark_send()`, but which have outstanding `SendRc`s that haven't been so marked.
    ///
    /// If this returns true, it means [`ready()`](PreSend::ready) will succeed without
    /// panic.
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
    /// pre_send.mark_send(&mut r1);
    /// assert!(pre_send.all_marked() == false); // r2 still unmarked
    /// pre_send.mark_send(&mut r2);
    /// assert!(pre_send.all_marked() == true); // r1/r2 allocation fully marked, q1/q2 not involved
    /// pre_send.mark_send(&mut q1);
    /// assert!(pre_send.all_marked() == false); // r1/r2 ok, but q2 unmarked
    /// pre_send.mark_send(&mut q2);
    /// assert!(pre_send.all_marked() == true); // both r1/r2 and q1/q2 allocations fully marked
    /// # std::mem::forget([r1, r2, q1, q2]);
    /// ```
    pub fn all_marked(&self) -> bool {
        // Count how many `SendRc`s point to each allocation.
        let ptr_sendrc_cnt: HashMap<_, usize> =
            self.marked
                .borrow()
                .values()
                .fold(HashMap::new(), |mut map, &ptr| {
                    *map.entry(ptr).or_default() += 1;
                    map
                });
        ptr_sendrc_cnt.into_iter().all(|(ptr, cnt)| {
            // Safety: allocation is valid because there is at least 1 SendRc pointing to
            // it. We may access it from this thread because PreSend isn't Send.
            let inner = unsafe { &*ptr.as_ptr() };
            cnt == inner.strong_count.get()
        })
    }

    /// Assert that all `SendRc`s that are to be moved to a new thread have been marked
    /// for move to a new thread.
    ///
    /// Returns a [`PostSend`] token that can proves all `SendRc`s belonging to the
    /// allocations involved in the move have been marked.  This token can be sent to
    /// another thread, along with the `SendRc`s, and used to re-enable them with a call
    /// to [`sent()`](PostSend::sent).
    ///
    /// Panics if there are outstanding `SendRc`s we haven't yet marked, i.e. if
    /// `all_marked()` would return false.
    pub fn ready(self) -> PostSend<T> {
        if !self.all_marked() {
            panic!("PreSend::ready() called before all SendRcs have been marked");
        }
        let ptrs: HashSet<_> = self.marked.into_inner().into_values().collect();
        // Pin allocations to a non-existent thread id, so that enable() can detect the
        // new thread from which we are called (even if that new thread is the current
        // thread again).
        for &ptr in &ptrs {
            // Safety: ptr belongs to at least one SendRc, so it's valid
            let inner = unsafe { &*ptr.as_ptr() };
            inner.pinned_to.store(0, Ordering::Relaxed);
        }
        PostSend { ptrs }
    }
}

/// Handle for enabling the `SendRc`s after they are sent to another thread.
///
/// Since `PostSend` can only be obtained via [`PreSend::ready()`], possessing a
/// `PostSend` serves as proof that all `SendRc`s pointing to the allocations involved in
/// the move have been marked.
#[must_use]
pub struct PostSend<T> {
    ptrs: HashSet<NonNull<Inner<T>>>,
}

// Safety: pointers to allocations can be sent to a new thread because PreSend::ready()
// checked that all their SendRcs have been marked.
unsafe impl<T> Send for PostSend<T> where T: Send {}

impl<T> PostSend<T> {
    /// Re-enable all pointers involved in the move and make their data accessible from
    /// this thread.
    ///
    /// This function cannot fail.
    pub fn sent(self) {
        let current_thread = current_thread();
        for ptr in self.ptrs {
            // Safety: ptr belongs to at least one SendRc, so it's valid
            let inner = unsafe { &*ptr.as_ptr() };
            inner.pinned_to.store(current_thread, Ordering::Relaxed);
            inner.marking.set(0);
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
        // Instead of panicking immediately, check whether we're in the correct thread and
        // leak the value if we're not. Then panic, but only if we're not already
        // panicking, because panic-inside-panic aborts the program and breaks unit tests.
        match self.check_pinned() {
            PinCheck::Ok => {
                let refcnt = self.inner().strong_count.get();
                if refcnt == 1 {
                    unsafe {
                        std::ptr::drop_in_place(self.ptr.as_ptr());
                    }
                } else {
                    self.inner().strong_count.set(refcnt - 1);
                }
            }
            check @ (PinCheck::BadThread | PinCheck::Marking) => {
                if !std::thread::panicking() {
                    panic!("SendRc::drop(): {}", check.errmsg());
                }
            }
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
        let post_send = SendRc::pre_send_ready([&mut r1, &mut r2]);

        std::thread::spawn(move || {
            post_send.sent();
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn missing_pre_send_drop() {
        let r = SendRc::new(RefCell::new(1));
        std::thread::spawn(move || {
            drop(r);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn missing_pre_send_deref() {
        let r1 = SendRc::new(RefCell::new(1));
        let r2 = SendRc::clone(&r1);
        let result = std::thread::spawn(move || {
            *r1.borrow_mut() = 2; // this should panic
            assert_eq!(*r2.borrow(), 2);
        })
        .join();
        assert!(result.is_err());
    }

    #[test]
    #[should_panic]
    fn incomplete_pre_send() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let pre_send = SendRc::pre_send();
        pre_send.mark_send(&mut r1);
        let _ = pre_send.ready(); // panics because we didn't mark _r2
    }

    #[test]
    #[should_panic = "before all SendRcs have been marked"]
    fn incomplete_pre_send_other_allocation() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut q1 = SendRc::new(RefCell::new(1));
        let _q2 = SendRc::clone(&q1);
        let pre_send = SendRc::pre_send();
        pre_send.mark_send(&mut r1);
        pre_send.mark_send(&mut r2);
        pre_send.mark_send(&mut q1);
        let _ = pre_send.ready(); // _q2 is missing
    }

    #[test]
    #[should_panic]
    fn faked_pre_send_count_reusing_same_ptr() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let pre_send = SendRc::pre_send();
        // marking the same SendRc twice won't fool us into thinking all SendRcs were
        // marked
        pre_send.mark_send(&mut r1);
        pre_send.mark_send(&mut r1);
        let _ = pre_send.ready();
    }

    #[test]
    #[should_panic = "call from different PreSend"]
    fn mark_same_sendrc_in_different_presend() {
        let mut r = SendRc::new(RefCell::new(1));
        let pre_send1 = SendRc::pre_send();
        pre_send1.mark_send(&mut r);
        let pre_send2 = SendRc::pre_send();
        pre_send2.mark_send(&mut r);
        std::mem::forget(r); // prevent panic on drop
    }

    #[test]
    fn mark_twice_good() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::new(RefCell::new(1));
        let pre_send = SendRc::pre_send();
        pre_send.mark_send(&mut r1);
        pre_send.mark_send(&mut r1);
        pre_send.mark_send(&mut r2);
        let _ = pre_send.ready();
        // avoid panic on drop
        std::mem::forget([r1, r2]);
    }

    #[test]
    fn mark_twice_bad() {
        let state = Arc::new(Mutex::new(0));
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                let mut r1 = SendRc::new(RefCell::new(1));
                let mut r2 = SendRc::new(RefCell::new(1));
                let pre_send1 = SendRc::pre_send();
                let pre_send2 = SendRc::pre_send();
                *state.lock().unwrap() = 1;
                pre_send1.mark_send(&mut r1);
                *state.lock().unwrap() = 2;
                pre_send1.mark_send(&mut r2);
                *state.lock().unwrap() = 3;
                pre_send2.mark_send(&mut r2); // panic
                *state.lock().unwrap() = 4;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock().unwrap(), 3);
    }
}
