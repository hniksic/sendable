//! `SendRc<T>`, a reference-counted pointer that is `Send` if `T` is `Send`.
//!
//! Sometimes it is useful to construct a hierarchy of objects which include `Rc`s and
//! send it off to another thread. `Rc` prohibits that because it can't statically prove
//! that all the clones of an individual `Rc` have been moved to the new thread.
//! `Rc::clone()` and `Rc::drop()` access and modify the reference count without
//! synchronization, which would lead to a data race if two `Rc` clones were to exist in
//! different threads.
//!
//! `Arc` allows moves between threads, but requires `T` to be `Sync`, which prohibits
//! moving an `Arc<RefCell<T>>` to a different thread. `Sync` is required because `Arc`
//! derefs to `&T`, so sending an `Arc` to a different thread automatically implies access
//! to `&T` from different threads. Allowed that on non-`Sync` types would enable an
//! `Arc<RefCell<u32>>` to execute `borrow()` or `borrow_mut()` from two threads without
//! synchronization.
//!
//! `SendRc` resolves by pinning the underlying allocation to a particular thread. You can
//! move `SendRc` to a different thread, but if you try to deref, clone, or drop it, you
//! get a panic. Instead, you must first disable hte `SendRc`s in the original thread, and
//! then reenable them in the new thread, after which they become usable again.

#![warn(missing_docs)]

use std::borrow::Borrow;
use std::cell::Cell;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread::ThreadId;

struct Inner<T> {
    // id of thread from which the value can be accessed
    pinned_to: AtomicU64,
    val: T,
    // the reference count
    strong_count: Cell<usize>,
}

static ID_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Reference-counted pointer like `Rc<T>`, but which is `Send` if `T` is `Send`.
///
/// This is different from `Arc` because the value can still be accessed from only one
/// thread at a time, and it is only allowed to manipulate it (by dropping or cloning)
/// from a single thread. This property makes it safe to store non-`Sync` types like
/// `RefCell` inside.
///
/// After a `SendRc` is created, it is pinned to the current thread, and usable only in
/// that thread. When sending a `SendRc` to different thread, you must first disable all
/// the `SendRc`s that point to the same allocation, then send them, and finally reenable
/// them in the new thread. They may again be sent to a different thread using the same
/// process.
///
/// ```
/// # use std::cell::RefCell;
/// # use send_rc::SendRc;
/// let mut r1 = SendRc::new(RefCell::new(1));
/// let mut r2 = SendRc::clone(&r1);
/// let mut r3 = SendRc::clone(&r1);
/// let mut send = r1.pre_send_disable_all([&mut r2, &mut r3]);
/// // move the pointers to a new thread
/// std::thread::spawn(move || {
///     // pointers are unusable here
///     send.enable(&mut r1);
///     // r1 is usable here
///     send.enable_many([&mut r2, &mut r3]);
///     // all three can now be used normally
///     *r1.borrow_mut() += 1;
///     assert_eq!(*r2.borrow(), 2);
///     assert_eq!(*r3.borrow(), 2);
/// })
/// .join()
/// .unwrap();
/// ```
///
/// Compared to `Rc`, tradeoffs are:
///
/// * `deref()`, `clone()`, and `drop()` requires a check that the pointer is not disabled,
///    and a relaxed atomic load to check that we're accessing it from the correct thread.
/// * takes up two machine words.
pub struct SendRc<T> {
    ptr: NonNull<Inner<T>>,
    // associate an non-changing id with each pointer so that we can track how many
    // have participated in migration
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
        // Safety: Inner is valid for as long as self
        unsafe { self.ptr.as_ref() }
    }

    fn inner_mut(&mut self) -> &mut Inner<T> {
        // Safety: Inner is valid for as long as self
        unsafe { self.ptr.as_mut() }
    }

    #[inline]
    fn check_pinned(&self) -> Result<(), &'static str> {
        if self.ptr == NonNull::dangling() {
            return Err("attempt to use disabled SendRc");
        }
        if self.inner().pinned_to.load(Ordering::Relaxed) != current_thread() {
            return Err("attempt to use SendRc from different thread; call pre_send() first");
        }
        Ok(())
    }

    fn assert_pinned(&self, op: &str) {
        if let Err(msg) = self.check_pinned() {
            panic!("SendRc::{op}: {msg}");
        }
    }

    /// Prepare to send this `SendRc` and other `SendRc`s belonging to the same allocation
    /// to the current thread.
    ///
    /// Before moving a `SendRc` to a different thread, you must disable all pointers to
    /// this allocation by calling `pre_send()` once to get a send handle, and invoking
    /// `send.disable(send_rc)` on every `SendRc` that belongs to the same
    /// allocation. Then you can call `read()` to obtain the `PostSend` token which serves
    /// as proof that all `SendRc`s of the allocation have been disabled. You can send the
    /// `SendRc`s and the token to the new thread, and re-enable the pointers by calling
    /// `enable()` on the token.
    pub fn pre_send(&mut self) -> PreSend<T> {
        self.assert_pinned("pre_send");
        let mut pre_send = PreSend {
            ptr: self.ptr,
            disabled: HashSet::new(),
        };
        pre_send.disable(self);
        pre_send
    }

    /// Prepare to send this and other `SendRc`s to a different thread.
    ///
    /// Equivalent to calling `pre_send()`, disabling the provided `SendRc`s, and invoking
    /// `PreSend::ready()`.
    pub fn pre_send_disable_all<'a>(
        &mut self,
        everyone: impl IntoIterator<Item = &'a mut Self>,
    ) -> PostSend<T>
    where
        T: 'a,
    {
        let mut send = self.pre_send();
        send.disable_many(everyone);
        send.ready()
    }

    /// Returns the number of pointers to this allocation.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn strong_count(this: &Self) -> usize {
        this.assert_pinned("strong_count");
        this.inner().strong_count.get()
    }

    /// Returns the inner value, if the `SendRc` has exactly one reference.
    ///
    /// Panics when invoked from a different thread than the one the `SendRc` was created
    /// in or last migrated to.
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        this.assert_pinned("try_unwrap");
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
        this.assert_pinned("get_mut");
        if this.inner().strong_count.get() == 1 {
            Some(&mut this.inner_mut().val)
        } else {
            None
        }
    }

    /// Returns true if the two `SendRc`s point to the same allocation.
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }
}

// Temporary workaround until ThreadId::as_u64() is stabilized.
fn current_thread() -> u64 {
    // This is not a guarantee that ThreadId is safe to transmute to u64, but it's
    // better than nothing.
    const _: () = assert!(std::mem::size_of::<ThreadId>() == 8);

    // Safety: ThreadId must have layout compatible with that of a u64, which is the
    // case in the stdlib where it's NonZeroU64.
    unsafe { std::mem::transmute(std::thread::current().id()) }
}

/// Handle for disabling the `SendRc`s of a particular allocation before they are sent to
/// another thread.
pub struct PreSend<T> {
    ptr: NonNull<Inner<T>>,
    disabled: HashSet<usize>,
}

impl<T> PreSend<T> {
    /// Make `send_rc` temporarily unusable so it can be sent to another thread.
    ///
    /// After this call it is no longer possible to deref, clone, or drop this `SendRc`.
    ///
    /// You must call `disable()` on all `SendRc`s pointing to an allocation before moving
    /// them to another thread.
    pub fn disable(&mut self, send_rc: &mut SendRc<T>) {
        if send_rc.ptr != NonNull::dangling() && send_rc.ptr != self.ptr {
            panic!("disable() must be called with the same allocation as use for pre_send()");
        }
        self.disabled.insert(send_rc.id);
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

    /// Returns true if all the `SendRc`s have been disabled and `ready()` can be called.
    pub fn all_disabled(&self) -> bool {
        let inner = unsafe { &*self.ptr.as_ptr() };
        self.disabled.len() == inner.strong_count.get()
    }

    /// Returns a token that can proves all `SendRc`s have been disabled.
    ///
    /// This token can be sent to another thread and used to re-enable the `SendRc`s
    /// there.
    ///
    /// Panics if not all `SendRc`s have been disabled, i.e. if `all_disabled()` would
    /// return false.
    pub fn ready(self) -> PostSend<T> {
        if !self.all_disabled() {
            panic!("ready() called before all SendRcs have been disabled");
        }
        PostSend {
            ptr: self.ptr,
            enabled: HashSet::new(),
            expected: self.disabled.len(),
            prev_thread: current_thread(),
        }
    }
}

/// Handle for enabling the `SendRc`s after they are sent to another thread.
#[must_use]
pub struct PostSend<T> {
    ptr: NonNull<Inner<T>>,
    enabled: HashSet<usize>,
    expected: usize,
    prev_thread: u64,
}

unsafe impl<T> Send for PostSend<T> where T: Send {}

impl<T> PostSend<T> {
    /// Make `send_rc` usable again after having moved it to a new thread.
    pub fn enable(&mut self, send_rc: &mut SendRc<T>) {
        let inner = unsafe { &*self.ptr.as_ptr() };
        let old_pinned_to = inner.pinned_to.swap(current_thread(), Ordering::SeqCst);
        if old_pinned_to != self.prev_thread && old_pinned_to != current_thread() {
            panic!("PostSend::enable() called from multiple threads");
        }
        if send_rc.ptr != NonNull::dangling() {
            if self.enabled.contains(&send_rc.id) {
                // make calling enable() twice a no-op
                return;
            }
            // attempted "enable" of a new SendRc, possibly obtained by cloning an already
            // enabled one
            panic!("PostSend::enable() called on a non-disabled SendRc");
        }
        send_rc.ptr = self.ptr;
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
        self.enabled.len() == self.expected
    }
}

/// Common trait for PreSend::disable() and AfterSend::enable(), allowing a common
/// implementation of visiting the `SendRc`s.
pub trait SendVisit<T> {
    /// Visit the `SendRc`.
    ///
    /// When invoked on `PreSend`, this calls `disable()`.
    /// When invoked on `PostSend`, this calls `enable()`.
    fn visit(&mut self, send_rc: &mut SendRc<T>);
}

impl<T> SendVisit<T> for PreSend<T> {
    fn visit(&mut self, send_rc: &mut SendRc<T>) {
        self.disable(send_rc);
    }
}

impl<T> SendVisit<T> for PostSend<T> {
    fn visit(&mut self, send_rc: &mut SendRc<T>) {
        self.enable(send_rc);
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
        self.assert_pinned("deref");
        &self.inner().val
    }
}

impl<T> Clone for SendRc<T> {
    fn clone(&self) -> Self {
        self.assert_pinned("clone");
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
                    panic!("drop: {msg}");
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
    use std::sync::Arc;

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
        let mut send = r1.pre_send_disable_all([&mut r2]);

        std::thread::spawn(move || {
            send.enable_many([&mut r1, &mut r2]);
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
        let mut send = r1.pre_send_disable_all([&mut r2]);
        let (mut send, mut r1, mut r2) = std::thread::spawn(move || {
            send.enable_many([&mut r1, &mut r2]);
            assert!(send.all_enabled());
            *r1.borrow_mut() += 1;
            assert_eq!(*r2.borrow(), 2);
            let send = r1.pre_send_disable_all([&mut r2]);
            (send, r1, r2)
        })
        .join()
        .unwrap();
        send.enable_many([&mut r1, &mut r2]);
        assert!(send.all_enabled());
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
        let send = r1.pre_send();
        let _ = send.ready(); // panics because we didn't disable r1
    }

    #[test]
    #[should_panic]
    fn faked_pre_send_count_reusing_same_ptr() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let mut send = r1.pre_send();
        // disabling the same SendRc twice won't fool us into thinking all instances were
        // disabled
        send.disable(&mut r1);
        let _ = send.ready();
    }

    #[test]
    #[should_panic = "must be called with the same allocation"]
    fn faked_pre_send_count_using_other_allocation() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let _r2 = SendRc::clone(&r1);
        let mut rogue = SendRc::new(RefCell::new(1));
        let mut send = r1.pre_send();
        // using SendRc from a different allocation won't fool us into thinking all
        // instances were disabled
        send.disable(&mut rogue);
        let _ = send.ready(); // this should panic
        // avoid panic in drops in case the above didn't panic
        std::mem::forget(r1);
        std::mem::forget(rogue);
    }

    #[test]
    fn enable_diff_threads() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut send = r1.pre_send_disable_all([&mut r2]);
        let mut send = std::thread::spawn(move || {
            send.enable(&mut r1);
            send
        })
        .join()
        .unwrap();
        let result = std::thread::spawn(move || {
            send.enable(&mut r2);
        })
        .join();
        assert!(result.is_err());
    }

    #[test]
    fn enable_half_way() {
        let state = Arc::new(Mutex::new(0));
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut r2 = SendRc::clone(&r1);
        let mut send = r1.pre_send_disable_all([&mut r2]);
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                *state.lock() = 1;
                send.enable(&mut r1);
                *state.lock() = 2;
                *r1.borrow_mut() = 2;
                *state.lock() = 3;
                let _ = &*r2; // should panic
                *state.lock() = 4;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock(), 3);
    }

    #[test]
    fn enable_undisabled() {
        let state = Arc::new(Mutex::new(0));
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut send = r1.pre_send_disable_all([]);
        let result = std::thread::spawn({
            let state = state.clone();
            move || {
                *state.lock() = 1;
                send.enable(&mut r1);
                let mut rogue = SendRc::clone(&r1);
                *state.lock() = 2;
                send.enable(&mut rogue); // should panic
                *state.lock() = 3;
            }
        })
        .join();
        assert!(result.is_err());
        assert_eq!(*state.lock(), 2);
    }

    #[test]
    fn enable_twice() {
        let mut r1 = SendRc::new(RefCell::new(1));
        let mut send = r1.pre_send_disable_all([]);
        std::thread::spawn(move || {
            send.enable(&mut r1);
            send.enable(&mut r1);
            send.enable(&mut r1);
        })
        .join()
        .unwrap();
    }
}
