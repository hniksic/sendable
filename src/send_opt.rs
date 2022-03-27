//! Container like `Option<T>`, but `Send` even if `T` is not `Send`.
//!
//! [`SendOption`] is useful for types which are composed of `Send` data, except for an
//! optional field of a non-send type. The field is set and used only inside a particular
//! thread, and will be `None` while sent across threads, but since Rust can't prove that,
//! a field of `Option<NonSendType>` makes the entire outer type not `Send`. For example,
//! a field with a `SendOption<Rc<Arena>>` could be used to create a `Send` type that
//! refers to a single-threaded arena.

use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::thread_id::current_thread;

/// Like `Option<T>`, but `Send` even if `T` is not `Send`.
///
/// This is sound because we require `SendOption` to be `None` when transferred across
/// thread boundary, so `T` values are never actually moved across threads.  If this is
/// violated by sending a non-`None` `SendOption` to another thread, access to its
/// contents (including through drop) will be detected and prevented with panic.
///
/// To migrate `SendOption` to another thread, set it to `None`, send it across, and call
/// [`post_send()`](SendOption::post_send) to use it normally.
pub struct SendOption<T> {
    pinned_to: AtomicU64,
    inner: ManuallyDrop<Option<T>>,
}

// Safety: we don't allow a T to be sent to another thread and accessed there in any way.
unsafe impl<T> Send for SendOption<T> {}

impl<T> SendOption<T> {
    /// Create new `SendOption<T>`, pinned to the current thread.
    pub fn new(val: Option<T>) -> Self {
        SendOption {
            pinned_to: AtomicU64::new(current_thread()),
            inner: ManuallyDrop::new(val),
        }
    }

    #[inline]
    fn check_pinned(&self) -> bool {
        self.pinned_to.load(Ordering::Relaxed) == current_thread()
    }

    fn assert_pinned(&self, op: &str) {
        if !self.check_pinned() {
            panic!("{op}: attempt to use non-None SendOption from different thread");
        }
    }

    /// Called after the option has been sent to another thread.
    ///
    /// This will panic if the option is not `None`.
    pub fn post_send(&mut self) {
        if self.inner.is_some() {
            panic!("attempt to send non-None SendOption to a different thread");
        }
        self.pinned_to.store(current_thread(), Ordering::Relaxed);
    }
}

impl<T> Deref for SendOption<T> {
    type Target = Option<T>;

    fn deref(&self) -> &Self::Target {
        self.assert_pinned("SendOption::deref()");
        &self.inner
    }
}

impl<T> DerefMut for SendOption<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.assert_pinned("SendOption::deref_mut()");
        &mut self.inner
    }
}

impl<T> Drop for SendOption<T> {
    fn drop(&mut self) {
        if self.check_pinned() {
            // Safety: we call drop() only once, and we don't access self.inner after it.
            unsafe {
                ManuallyDrop::drop(&mut self.inner);
            }
        } else if !std::thread::panicking() {
            panic!("SendOption::drop(): attempt to use non-None SendOption from different thread");
        }
    }
}

impl<T: Clone> Clone for SendOption<T> {
    fn clone(&self) -> Self {
        SendOption::new((&**self).clone())
    }
}

impl<T> Default for SendOption<T> {
    fn default() -> Self {
        SendOption::new(None)
    }
}

impl<T: Debug> Debug for SendOption<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opt: &Option<T> = &**self;
        f.debug_tuple("SendOption").field(opt).finish()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::SendOption;

    #[test]
    fn trivial() {
        let mut o = SendOption::new(Some(Rc::new(0)));
        assert_eq!(o.as_deref(), Some(&0));
        *o = None;
        assert_eq!(o.as_ref(), None);
        *o = Some(Rc::new(0));
    }

    #[test]
    fn debug_impl() {
        assert_eq!(format!("{:?}", SendOption::new(Some(0))), "SendOption(Some(0))");
        assert_eq!(format!("{:?}", SendOption::new(None::<u32>)), "SendOption(None)");
    }

    #[test]
    #[should_panic]
    fn bad_deref_some() {
        let o = SendOption::new(Some(Rc::new(0)));
        std::thread::spawn(move || {
            let _ = &*o;
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn bad_deref_none() {
        let o: SendOption<Rc<u32>> = SendOption::new(None);
        std::thread::spawn(move || {
            let _ = &*o;
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn bad_deref_mut() {
        let mut o = SendOption::new(Some(Rc::new(0)));
        std::thread::spawn(move || {
            let _ = &mut *o;
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn bad_drop() {
        let o = SendOption::new(Some(Rc::new(0)));
        std::thread::spawn(move || {
            drop(o);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn good_send() {
        let mut o: SendOption<Rc<u32>> = SendOption::default();
        std::thread::spawn(move || {
            o.post_send();
            *o = Some(Rc::new(1));
        })
        .join()
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn send_and_return_bad1() {
        let mut o: SendOption<Rc<u32>> = SendOption::default();
        let o = std::thread::spawn(move || {
            o.post_send();
            *o = Some(Rc::new(1));
            o
        })
        .join()
        .unwrap();
        let _ = &*o; // must panic because we haven't called post_send()
    }

    #[test]
    #[should_panic]
    fn send_and_return_bad2() {
        let mut o: SendOption<Rc<u32>> = SendOption::default();
        let mut o = std::thread::spawn(move || {
            o.post_send();
            *o = Some(Rc::new(1));
            o
        })
        .join()
        .unwrap();
        o.post_send(); // must panic because it's not None
    }

    #[test]
    fn send_and_return_good() {
        let mut o: SendOption<Rc<u32>> = SendOption::default();
        let mut o = std::thread::spawn(move || {
            o.post_send();
            *o = Some(Rc::new(1));
            *o = None;
            o
        })
        .join()
        .unwrap();
        o.post_send();
        let _ = &*o;
    }
}
