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
use std::cell::Cell;
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
    val: T,
    // the reference count
    strong_count: Cell<usize>,
}

static ID_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Reference-counting pointer like `Rc<T>`, but which is `Send` if `T` is `Send`.
///
/// After a `SendRc` is created, it is pinned to the current thread, and usable only in
/// that thread. When sending a `SendRc` to different thread, you must first disable all
/// the `SendRc`s that point to the same allocation, then send them, and finally reenable
/// them in the new thread. They may again be sent to a different thread using the same
/// process.
///
/// ```
/// # use std::cell::RefCell;
/// # use sendable::SendRc;
/// // create two SendRcs pointing to the same allocation
/// let mut r1 = SendRc::new(RefCell::new(1));
/// let mut r2 = SendRc::clone(&r1);
///
/// // prepare to ship them off to a different thread
/// let mut pre_send = SendRc::pre_send();
/// pre_send.disable(&mut r1); // r1 is unusable from this point
/// pre_send.disable(&mut r2); // r2 is unusable from this point
/// // ready() would panic on un-disabled SendRcs pointing to the allocation of r1/r2
/// let mut post_send = pre_send.ready();
///
/// // move everything to a different thread
/// std::thread::spawn(move || {
///     // both pointers are unusable here
///     post_send.enable(&mut r1); // r1 is usable from this point
///     post_send.enable(&mut r2); // r2 is usable from this point
///     *r1.borrow_mut() += 1;
///     assert_eq!(*r2.borrow(), 2);
/// })
/// .join()
/// .unwrap();
/// ```
///
/// If the `SendRc`s are edges in a graph, you'll need to visit the whole graph before and
/// after the migration to the new thread. In the pre-send phase you'll need to disable the
/// pointer after visiting its neighbors, whereas in the post-send step you'll need to first
/// re-enable the pointer and then visit the neighbors.
///
/// Compared to `Rc`, tradeoffs are:
///
/// * `deref()`, `clone()`, and `drop()` requires a check that the pointer is not disabled,
///    and a relaxed atomic load to check that we're accessing it from the correct thread.
/// * takes up two machine words.
/// * doesn't support weak pointers (though such support could be added).
pub struct SendRc<T> {
    ptr: NonNull<Inner<T>>,
    // Associate an non-changing id with each pointer so that we can track how many have
    // participated in migration. If malicious code forces wrap-around of the id, we're
    // still sound because passing two SendRcs with the same id to `PreSend::disable()`
    // will just cause `PreSend::ready()` to always fail and prevent migration.
    id: usize,
}

// Safety: SendRc can be sent between threads because we prohibit access to clone, drop,
// and deref except from the thread they are pinned to. Access is granted only after all
// pointers to the same allocation have been migrated to the new thread, which is why we
// can avoid requiring T: Sync.
unsafe impl<T> Send for SendRc<T> where T: Send {}

impl<T> SendRc<T> {
    /// Constructs a new `SendRc<T>`.
    ///
    /// The newly created `SendRc` is only usable from the current thread. To send it to
    /// another thread, you must call `pre_send()`, disable it, and re-enable it in the
    /// new thread.
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(Inner {
            pinned_to: AtomicU64::new(current_thread()),
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
            id: ID_NEXT.fetch_add(1, Ordering::Relaxed),
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
    fn check_pinned_and_valid(&self) -> Result<(), &'static str> {
        if self.ptr == NonNull::dangling() {
            return Err("attempt to use disabled SendRc");
        }
        if self.inner().pinned_to.load(Ordering::Relaxed) != current_thread() {
            return Err("attempt to use SendRc from different thread; call pre_send() first");
        }
        Ok(())
    }

    fn assert_pinned_and_valid(&self, op: &str) {
        if let Err(msg) = self.check_pinned_and_valid() {
            panic!("{op}: {msg}");
        }
    }

    /// Prepare to send `SendRc`s of this type to a different thread.
    ///
    /// Before moving a `SendRc` to a different thread, you must disable it as well as all
    /// other `SendRc`s pointing to the same allocation:
    ///
    /// ```
    /// # use std::cell::RefCell;
    /// # use sendable::SendRc;
    /// let mut r1 = SendRc::new(RefCell::new(1));
    /// let mut r2 = SendRc::clone(&r1);
    /// let mut pre_send = SendRc::pre_send();
    /// pre_send.disable(&mut r1);
    /// pre_send.disable(&mut r2);
    /// let mut post_send = pre_send.ready();
    /// // send, r1, and r2 can now be send to a different thread, and re-enabled
    /// // by calling post_send.enable(&mut r1) and post_send.enable(&mut r2)
    /// # post_send.enable(&mut r1); post_send.enable(&mut r2); // avoid panic in doctest
    /// ```
    pub fn pre_send() -> PreSend<T> {
        PreSend {
            disabled: HashMap::new(),
        }
    }

    /// Prepare to send a fixed collection of `SendRc`s to a different thread.
    ///
    /// Calls `SendRc::pre_send()`, then `disable()` on the provided `SendRc`s, then
    /// finishes the pre-send phase with a call to `ready()`. Returns the `PostSend` token
    /// which to send to the new thread along with the `SendRc`s and use to re-enable the
    /// `SendRc`s.
    ///
    /// Panics if there are allocations pointed to by `SendRc`s in `all` which have extra
    /// `SendRc`s not included in `all`.
    pub fn pre_send_ready<'a>(all: impl IntoIterator<Item = &'a mut Self>) -> PostSend<T>
    where
        T: 'a,
    {
        let mut send = Self::pre_send();
        send.disable_many(all);
        send.ready()
    }

    /// Returns true if this `SendRc` has been disabled for sending to a new thread.
    pub fn is_sendrc_disabled(&self) -> bool {
        self.ptr == NonNull::dangling()
    }

    /// Returns the number of pointers to this allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn strong_count(this: &Self) -> usize {
        this.assert_pinned_and_valid("SendRc::strong_count()");
        this.inner().strong_count.get()
    }

    /// Returns the inner value, if the `SendRc` has exactly one reference.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        this.assert_pinned_and_valid("SendRc::try_unwrap()");
        if this.inner().strong_count.get() == 1 {
            // Safety: refcount is 1, so it's just us, and the pointer was obtained using
            // Box::into_raw().
            let inner_box = unsafe { Box::from_raw(this.ptr.as_ptr()) };
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
        this.assert_pinned_and_valid("SendRc::get_mut()");
        if this.inner().strong_count.get() == 1 {
            // Safety: we've checked that refcount is 1
            unsafe { Some(&mut this.inner_mut().val) }
        } else {
            None
        }
    }

    /// Returns true if the two `SendRc`s point to the same allocation.
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }
}

/// Handle for disabling the `SendRc`s before they are sent to another thread.
///
/// This handle cannot be sent to a different thread; when done with calls to `disable()`,
/// invoke `ready()` to obtain a token that can be moved to the other thread to re-enable
/// the `SendRc`s.
pub struct PreSend<T> {
    disabled: HashMap<usize, NonNull<Inner<T>>>,
}

impl<T> PreSend<T> {
    /// Make `send_rc` temporarily unusable so it can be sent to another thread.
    ///
    /// After this call it is no longer possible to deref, clone, or drop this `SendRc`.
    ///
    /// After disabling a `SendRc`, you must disable other `SendRc`s pointing to the same
    /// allocation. This needs to be done before the call to `ready()`.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn disable(&mut self, send_rc: &mut SendRc<T>) {
        if self.disabled.contains_key(&send_rc.id) {
            // make calling disable() twice a no-op
            return;
        }
        // If send_rc.ptr is dangling, it means the send_rc was already disabled in
        // another SendRc.  This is a programming error, and will be caught by
        // assert_pinned_and_valid().
        send_rc.assert_pinned_and_valid("PreSend::disable()");
        self.disabled.insert(send_rc.id, send_rc.ptr);
        send_rc.ptr = NonNull::dangling();
    }

    /// Make multiple `SendRc`s temporarily unusable as if with `disable()`.
    pub fn disable_many<'a>(&mut self, many: impl IntoIterator<Item = &'a mut SendRc<T>>)
    where
        T: 'a,
    {
        for send_rc in many {
            self.disable(send_rc);
        }
    }

    /// Returns true if there are no allocations whose `SendRc`s we've disabled, but which
    /// have outstanding `SendRc`s we haven't yet disabled.
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
    /// let mut pre_send = SendRc::pre_send();
    /// pre_send.disable(&mut r1);
    /// assert!(pre_send.all_disabled() == false); // r2 not disabled
    /// pre_send.disable(&mut r2);
    /// assert!(pre_send.all_disabled() == true); // r1/r2 allocation fully disabled, q1/q2 doesn't participate
    /// pre_send.disable(&mut q1);
    /// assert!(pre_send.all_disabled() == false); // r1/r2 ok, but q2 not disabled
    /// pre_send.disable(&mut q2);
    /// assert!(pre_send.all_disabled() == true); // both r1/r2 allocations and q1/q2 allocations fully disabled
    /// # std::mem::forget([r1, r2, q1, q2]);
    /// ```
    pub fn all_disabled(&self) -> bool {
        // Count how many `SendRc`s point to each allocation.
        let ptr_sendrc_cnt: HashMap<_, usize> =
            self.disabled
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

    /// Returns a [`PostSend`] token that can proves all `SendRc`s have been disabled.
    ///
    /// This token can be sent to another thread and used to re-enable the `SendRc`s
    /// there.
    ///
    /// Panics if there are outstanding `SendRc`s we haven't yet disabled, i.e. if
    /// `all_disabled()` would return false.
    pub fn ready(self) -> PostSend<T> {
        if !self.all_disabled() {
            panic!("PreSend::ready() called before all SendRcs have been disabled");
        }
        // Pin allocations to a non-existent thread id, so that enable() can detect the
        // new thread from which we are called (even if that new thread is the current
        // thread again).
        for &ptr in self.disabled.values() {
            // Safety: ptr belongs to a SendRc, so it's valid
            let inner = unsafe { &*ptr.as_ptr() };
            inner.pinned_to.store(0, Ordering::Relaxed);
        }
        PostSend {
            disabled: self.disabled,
            enabled: HashSet::new(),
            new_thread: 0,
        }
    }
}

/// Handle for enabling the `SendRc`s after they are sent to another thread.
///
/// Since `PostSend` can only be obtained via [`PreSend::ready()`], possessing a
/// `PostSend` serves as proof that all `SendRc`s belonging to the allocations involved
/// in the move have been disabled.
#[must_use]
pub struct PostSend<T> {
    disabled: HashMap<usize, NonNull<Inner<T>>>,
    enabled: HashSet<usize>,
    new_thread: u64,
}

// Safety: pointers to allocations can be sent to a new thread because PreSend::ready()
// checked that all their SendRcs have been disabled.
unsafe impl<T> Send for PostSend<T> where T: Send {}

impl<T> PostSend<T> {
    /// Make `send_rc` usable again after having moved it to a new thread.
    pub fn enable(&mut self, send_rc: &mut SendRc<T>) {
        let current_thread = current_thread();
        // Disallow calling enable() from different threads in sequence, even if that
        // would be sound. If the user needs multiple threads, they can create multiple
        // PreSend/PostSend.
        match self.new_thread {
            0 => self.new_thread = current_thread,
            id if id == current_thread => {}
            _ => panic!("PostSend::enable() called from more than one thread"),
        }
        if self.enabled.contains(&send_rc.id) {
            // make calling enable() twice a no-op
            return;
        }
        if send_rc.ptr != NonNull::dangling() {
            panic!("PostSend::enable() called on a non-disabled SendRc");
        }
        // This will panic if the SendRc was disabled by another PreSend.
        let ptr = self
            .disabled
            .get(&send_rc.id)
            .copied()
            .expect("PostSend::enable() called on a SendRc disabled elsewhere");
        // Safety: SendRc guarantees that the pointer points to valid allocation.
        let inner = unsafe { &*ptr.as_ptr() };
        // Check previously pinned thread in case someone sends PostSend to another
        // thread, enables a pointer, and then sends it to a third thread and enables
        // another one.
        let old_pinned_to = inner.pinned_to.load(Ordering::Relaxed);
        if old_pinned_to == 0 {
            // We can get away with load+store without CAS because this can't happen in
            // parallel, since we take &mut self.
            inner.pinned_to.store(current_thread, Ordering::Relaxed);
        } else {
            // Since we check that enable() is always called from the same thread, it
            // shouldn't be possible for the allocation to be pinned to some other thread.
            assert_eq!(old_pinned_to, current_thread);
        }
        send_rc.ptr = ptr;
        self.enabled.insert(send_rc.id);
    }

    /// Make multiple `SendRc`s usable again as if with `enable()`.
    pub fn enable_many<'a>(&mut self, many: impl IntoIterator<Item = &'a mut SendRc<T>>)
    where
        T: 'a,
    {
        for send_rc in many {
            self.enable(send_rc);
        }
    }

    /// Returns true if all the `SendRc`s have been enabled.
    ///
    /// Useful for asserting that the post-send traversal hasn't missed a `SendRc`
    /// somewhere.
    pub fn all_enabled(&self) -> bool {
        self.enabled.len() == self.disabled.len()
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
        self.assert_pinned_and_valid("SendRc::deref()");
        &self.inner().val
    }
}

impl<T> Clone for SendRc<T> {
    fn clone(&self) -> Self {
        self.assert_pinned_and_valid("SendRc::clone()");
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
        match self.check_pinned_and_valid() {
            Ok(()) => {
                let refcnt = self.inner().strong_count.get();
                if refcnt == 1 {
                    unsafe {
                        std::ptr::drop_in_place(self.ptr.as_ptr());
                    }
                } else {
                    self.inner().strong_count.set(refcnt - 1);
                }
            }
            Err(msg) => {
                if !std::thread::panicking() {
                    panic!("SendRc::drop(): {msg}");
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

/// Common trait for `PreSend` and `PostSend`, allowing common code for traversal of
/// `SendRc`s to disable/enable them.
///
/// When `SendRc`s form a graph, you'll need to visit them before and after the `Send`,
/// and this trait is meant to help with making the code generic:
///
/// ```no_run
/// # use sendable::{SendRc, send_rc::SendAction};
/// # use std::cell::RefCell;
/// struct Node {
///     neighbor: SendRc<RefCell<Node>>,
///     // data: ...
/// }
///
/// impl Node {
///     fn migrate(&mut self, action: &mut impl SendAction<RefCell<Node>>) {
///         if action.will_disable() {
///             self.neighbor.borrow_mut().migrate(action);
///             action.apply(&mut self.neighbor);
///         } else {
///             action.apply(&mut self.neighbor);
///             self.neighbor.borrow_mut().migrate(action);
///         }
///     }
/// }
/// ```
pub trait SendAction<T> {
    /// Returns true if the action will disable the pointer, i.e. if the action is
    /// `PreSend`.
    fn will_disable(&self) -> bool;

    /// Calls `PreSend::disable()` or `PostSend::enable()` depending on the
    /// implementation.
    fn apply(&mut self, send_rc: &mut SendRc<T>);
}

impl<T> SendAction<T> for PreSend<T> {
    fn will_disable(&self) -> bool {
        true
    }

    fn apply(&mut self, send_rc: &mut SendRc<T>) {
        self.disable(send_rc);
    }
}

impl<T> SendAction<T> for PostSend<T> {
    fn will_disable(&self) -> bool {
        false
    }

    fn apply(&mut self, send_rc: &mut SendRc<T>) {
        self.enable(send_rc);
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
        let mut post_send = SendRc::pre_send_ready([&mut r1, &mut r2]);

        std::thread::spawn(move || {
            post_send.enable_many([&mut r1, &mut r2]);
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
        let mut post_send = SendRc::pre_send_ready([&mut r1, &mut r2]);
        let (mut post_send, mut r1, mut r2) = std::thread::spawn(move || {
            post_send.enable_many([&mut r1, &mut r2]);
            assert!(post_send.all_enabled());
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
            let post_send = SendRc::pre_send_ready([&mut r1, &mut r2]);
            (post_send, r1, r2)
        })
        .join()
        .unwrap();
        post_send.enable_many([&mut r1, &mut r2]);
        assert!(post_send.all_enabled());
        assert_eq!(*r2.borrow(), 2);
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
        let mut pre_send = SendRc::pre_send();
        pre_send.disable(&mut r1);
        let _ = pre_send.ready(); // panics because we didn't disable _r2
    }

    #[test]
    #[should_panic = "before all SendRcs have been disabled"]
    fn incomplete_pre_send_other_allocation() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut q1 = SendRc::new(RefCell::new(1));
        let _q2 = SendRc::clone(&q1);
        let mut pre_send = SendRc::pre_send();
        pre_send.disable(&mut r1);
        pre_send.disable(&mut r2);
        pre_send.disable(&mut q1);
        let _ = pre_send.ready(); // _q2 is missing
    }

    #[test]
    #[should_panic]
    fn faked_pre_send_count_reusing_same_ptr() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let mut pre_send = SendRc::pre_send();
        // disabling the same SendRc twice won't fool us into thinking all instances were
        // disabled
        pre_send.disable(&mut r1);
        pre_send.disable(&mut r1);
        let _ = pre_send.ready();
    }

    #[test]
    #[should_panic = "attempt to use disabled"]
    fn disable_same_sendrc_in_different_presend() {
        let mut r = SendRc::new(RefCell::new(1));
        let mut pre_send1 = SendRc::pre_send();
        pre_send1.disable(&mut r);
        let mut pre_send2 = SendRc::pre_send();
        pre_send2.disable(&mut r);
        std::mem::forget(r); // prevent panic on drop
    }

    #[test]
    fn enable_same_send_diff_threads() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut post_send = SendRc::pre_send_ready([&mut r1, &mut r2]);
        let mut post_send = std::thread::spawn(move || {
            post_send.enable(&mut r1);
            post_send
        })
        .join()
        .unwrap();
        let result = std::thread::spawn(move || {
            post_send.enable(&mut r2);
        })
        .join();
        assert!(result.is_err());
    }

    #[test]
    fn enable_half_way() {
        let state = Arc::new(Mutex::new(0));
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut post_send = SendRc::pre_send_ready([&mut r1, &mut r2]);
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                *state.lock().unwrap() = 1;
                post_send.enable(&mut r1);
                *state.lock().unwrap() = 2;
                *r1.borrow_mut() = 2;
                *state.lock().unwrap() = 3;
                let _ = &*r2; // should panic
                *state.lock().unwrap() = 4;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock().unwrap(), 3);
    }

    #[test]
    fn enable_undisabled() {
        let state = Arc::new(Mutex::new(0));
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut post_send = SendRc::pre_send_ready([&mut r1]);
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                *state.lock().unwrap() = 1;
                post_send.enable(&mut r1);
                let mut rogue = SendRc::clone(&r1);
                *state.lock().unwrap() = 2;
                post_send.enable(&mut rogue); // should panic
                *state.lock().unwrap() = 3;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock().unwrap(), 2);
    }

    #[test]
    fn enable_twice() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut post_send = SendRc::pre_send_ready([&mut r1]);
        std::thread::spawn(move || {
            post_send.enable(&mut r1);
            post_send.enable(&mut r1);
            post_send.enable(&mut r1);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn disable_twice_good() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::new(RefCell::new(1));
        let mut pre_send = SendRc::pre_send();
        pre_send.disable(&mut r1);
        pre_send.disable(&mut r1);
        pre_send.disable(&mut r2);
        let _ = pre_send.ready();
        // avoid panic on drop
        std::mem::forget([r1, r2]);
    }

    #[test]
    fn disable_twice_bad() {
        let state = Arc::new(Mutex::new(0));
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                let mut r1 = SendRc::new(RefCell::new(1));
                let mut r2 = SendRc::new(RefCell::new(1));
                let mut pre_send1 = SendRc::pre_send();
                let mut pre_send2 = SendRc::pre_send();
                *state.lock().unwrap() = 1;
                pre_send1.disable(&mut r1);
                *state.lock().unwrap() = 2;
                pre_send1.disable(&mut r2);
                *state.lock().unwrap() = 3;
                pre_send2.disable(&mut r2); // panic
                *state.lock().unwrap() = 4;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock().unwrap(), 3);
    }

    #[test]
    #[should_panic = "enable() called from more than one thread"]
    fn send_from_diff_threads() {
        let mut a = SendRc::new(RefCell::new(1));
        let mut b = SendRc::clone(&a);
        let mut post_send = SendRc::pre_send_ready([&mut a, &mut b]);
        let mut post_send = std::thread::spawn(move || {
            post_send.enable(&mut a);
            post_send
        })
        .join()
        .unwrap();
        post_send.enable(&mut b);
    }
}
