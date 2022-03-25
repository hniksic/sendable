# send-rc

The `send-rc` crate defines a single-threaded reference-counted pointer that can be sent
between threads. You can think of it as a variant of `Rc<T>` that is `Send` if `T` is
`Send`. This is unlike `Rc<T>` which is never `Send`, and also unlike `Arc<T>`, which
requires `T: Send + Sync` to be `Send`.

## How does it work?

When `SendRc` is first created, it stores the id of the current thread on the heap, next
to the value and the reference count. Before granting access to the value, and before
modifying the reference count through `clone()` and `drop()` it checks that the `SendRc`
is still in the thread it was created in.

When the hierarchy of `SendRc` needs to be moved to a different thread, each pointer is
marked for sending using the API provided for that purpose. Once marked for sending, the
pointer is no longer usable, even in the original thread. When all pointers are thus
marked, they can be sent to a different threads, where they are restored in the same way.
In a simple case of two pointers, the process looks like this:

```rust
let mut r1 = SendRc::new(RefCell::new(1));
let mut r2 = SendRc::clone(&r1);
let mut send = SendRc::pre_send();
send.disable(&mut r1); // r1 is unusable from this point
send.disable(&mut r2); // r2 is unusable from this point
let mut send = send.ready(); // would panic if there were un-disabled SendRcs pointing to
                             // the allocation of r1/r2
// move everything to a different thread
std::thread::spawn(move || {
    // pointers are unusable here
    send.enable(&mut r1); // r1 is usable from this point
    send.enable(&mut r2); // r2 is usable from this point
    *r1.borrow_mut() += 1;
    assert_eq!(*r2.borrow(), 2);
})
.join()
.unwrap();
```

## When is this needed?

When working inside a single thread, it is allowed and perfectly safe to create a
hierarchy that shares data using `Rc` and involves optional mutation using `Cell` and
`RefCell`.  That is both safe and efficient because it doesn't require any atomics or
mutexes, so Rust can optimize calls to `borrow()` and `borrow_mut()` where they are not
globally observable.

Now, imagine that you create such a hierarchy and want to move it to another thread.
`RefCell` and `Cell` are `Send` - they involve interior mutability, but no sharing. The
trouble is with `Rc`, which is neither `Send` nor `Sync`, and for good reason. Even though
it would be perfectly fine to move a hierarchy of `Rc`s from one thread to another, the
borrow checker cannot statically prove that you have moved _all_ of them - some might
remain in the original thread and wreak havoc with unsychronized manipulation of the
reference count.

If there were a way to demonstrate to Rust that you've sent all pointers to a particular
allocation to a different thread, there would be no problem in moving `Rc<T>` instances to
a different thread, provided that `T` itself were `Send`. As shown above, `SendRc` does
exactly that.

## Why not just use Arc?

`Arc` indeed allows moves between threads, but it fundamentally assumes that the
underlying value will be _shared_ between threads. `Arc` requires `T: Send + Sync` in
order for `Arc<T>` to be `Send` because if it only required `T: Send`, you could create an
`Arc<RefCell<u32>>`, clone it, send the clone to a different thread, and call
`borrow_mut()` from two threads on the same `RefCell` without synchronization. That is
forbidden, and is why `Arc<RefCell<T>>` is not a thing in Rust.

`SendRc` can get away with allowing this because it requires proof that all access to
value from the previous thread was revoked before allowing the value to be pinned to a new
thread. `SendRc<RefCell<u32>>` is sound because if you just send a clone to a different
thread, you won't be able to either deref, or clone it, or even drop it - either of these
operations would result in a panic.

One could fix the issue by using the full-blown `Arc<Mutex<T>>` or `Arc<RwLock<T>>`.
There are two issues with this approach: one that it's slower because in case of stdlib
mutexes it requires heap allocation of system mutex and the overhead of atomics and poison
checks. Even the most efficient mutex implementations, like `parking_lot`, don't come for
free, and bear the cost of atomic synchronization. The second issue is conceptual: it is
simply wrong to use `Arc<Mutex<T>>` if neither `Arc` nor `Mutex` is actually needed
because the code *doesn't* access the value of `T` from multiple threads in parallel.

## Why not use an arena? Or unsafe?

An arena would be an acceptable solution, but to make it `Send`, it requires that the
whole design be devoted to the arena conception from the ground up. A simple solution of
replacing `Rc` with an arena id doesn't really work because in addition to the id, the
object needs a reference to the arena. It can't have an `Option<&Arena>` field because it
would make it non-`Send` for an arena that contains non-`Sync` cells. Since we need
`Arena` to contain `RefCell`, this doesn't work.

There are arena-based designs that do work, but require completely decoupling values from
their sharing, so that the data is in the arena, and the values you work with are created
on-the-fly and have a lifetime connected to the lifetime of the arena. This requires
dealing with the lifetime everywhere and is not easy to get right for non-experts.

Finally, one can avoid the arena by just using `unsafe impl Send` on a root type that is
used to send the whole world to the new thread, borrow checker be damned. That solution is
hacky and gives up the guarantees afforded by Rust. If you make a mistake, say by leaving
an `Rc` clone in the original thread, you're back to core dumps like in C++. In Rust we
hope to do better, and `SendRc` is my attempt to make such an operation safe.

## Is this really safe?

As with any crate that uses unsafe code, one can never be 100% positive that there is no
soundness bug. I spent some time going through the design and reviewing the implementation
before I settled on the current approach. While I did find the occasional issue, the
design held.

You are invited to look at the code - it is not large - and report any issues you
encounter.

## Are the run-time checks expensive?

While they are certainly more expensive than the `Rc` which doesn't need them, they are
still quite cheap. `deref()` just checks that the pointer is not dangling (which means
that that `SendRc` was disabled) and fetches the id of the pinned-to thread using a
relaxed atomic load, which compiles into an ordinary load on Intel. It's hard to be
cheaper than that, and if you're worried, you can hold on to the reference to avoid
repeating the checks. (The borrow checker will prevent you from sending the `SendRc` to
another thread while there is an outstanding reference.) `clone()` and `drop()` do the
same kind of check.

## License

`send-rc` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).  See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for
details.  Contributing changes is assumed to signal agreement with these licensing terms.
