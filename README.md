`SendRc<T>`, a reference-counted pointer that is `Send` if `T` is `Send`.

Sometimes it is useful to construct a hierarchy of objects which include `Rc`s and
send it off to another thread. `Rc` prohibits that because it can't statically prove
that all the clones of an individual `Rc` have been moved to the new thread.
`Rc::clone()` and `Rc::drop()` access and modify the reference count without
synchronization, which would lead to a data race if two `Rc` clones were to exist in
different threads.

`Arc` allows moves between threads, but requires `T` to be `Sync`, which prohibits
moving an `Arc<RefCell<T>>` to a different thread. `Sync` is required because `Arc`
derefs to `&T`, so sending an `Arc` to a different thread automatically implies access
to `&T` from different threads. Allowed that on non-`Sync` types would enable an
`Arc<RefCell<u32>>` to execute `borrow()` or `borrow_mut()` from two threads without
synchronization.

`SendRc` resolves the issue by pinning the underlying allocation to a thread. You can move
`SendRc` to a different thread, but if you try to deref, clone, or drop it, you get a
panic. Instead, you must first disable hte `SendRc`s in the original thread, and then
reenable them in the new thread, after which they become usable again.
