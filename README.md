`SendRc<T>`, a wrapper around `Rc<T>` that is `Send` if `T` is `Send`.

Sometimes it is useful to construct a hierarchy of objects which include `Rc`s and
send it off to another thread. `Rc` prohibits that because it can't statically prove
that all the clones of an individual `Rc` have been moved to the new thread.
`Rc::clone()` and `Rc::drop()` access and modify the reference count without
synchronization, which would lead to a data race if two `Rc` clones were to exist in
different threads.

Using `Arc` helps to an extent, but still requires `T` to be `Sync`, so you can't move
a hierarchy with `Arc<RefCell<T>>` to a different thread. The `Sync` requirement is
because `Arc` derefs to `&T`, so allowing `Arc` clones containing non-`Sync` values to
exist in different threads would break the invariant of non-`Sync` values inside -
e.g. it would enable an `Arc<RefCell<u32>>` to execute `borrow()` or `borrow_mut()`
from two threads without synchronization.

`SendRc` resolves the above by storing the thread ID of the thread in which it was
created. Each deref, clone, and drop of the `SendRc` is only allowed from that
thread. After moving `SendRc` to a different thread, you must invoke `migrate()` to
migrate it to the new thread. Only after all the clones of a `SendRc` have been thus
marked will access to the inner value (and to `SendRc::clone()` and `SendRc::drop()`)
be allowed.
