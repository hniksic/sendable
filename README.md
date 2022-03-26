# sendable

The `sendable` crate defines two types that facilitate sending data between threads:

* `SendRc`, a single-threaded reference-counting pointer that can be sent between
  threads. You can think of it as a variant of `Rc<T>` that is `Send` if `T` is
  `Send`. This is unlike `Rc<T>` which is never `Send`, and also unlike `Arc<T>`, which
  requires `T: Send + Sync` to be `Send`.
* `SendOption`, a container like `Option<T>` that is `Send` even if `T` is not `Send`.

## How does SendRc work?

When `SendRc` is constructed, it stores the id of the current thread next to the value and
the reference count. Before granting access to the value, and before modifying the
reference count through `clone()` and `drop()`, it checks that the `SendRc` is still in
the thread it was created in.

When a hierarchy containing `SendRc`s needs to be moved to a different thread, each
pointer is marked for sending using the API provided for that purpose. Once thus marked,
access to underlying data is disabled from that pointer, even in the original thread. When
all pointers are disabled, they can be sent across the thread boundary, and re-enabled.
In a simple case of two pointers, the process looks like this:

```rust
// create two SendRcs pointing to the same allocation
let mut r1 = SendRc::new(RefCell::new(1));
let mut r2 = SendRc::clone(&r1);

// prepare to ship them off to a different thread
let mut pre_send = SendRc::pre_send();
pre_send.disable(&mut r1); // r1 is unusable from this point
pre_send.disable(&mut r2); // r2 is unusable from this point
// ready() would panic on un-disabled SendRcs pointing to the allocation of r1/r2
let mut post_send = pre_send.ready();

// move everything to a different thread
std::thread::spawn(move || {
    // both pointers are unusable here
    post_send.enable(&mut r1); // r1 is usable from this point
    post_send.enable(&mut r2); // r2 is usable from this point
    *r1.borrow_mut() += 1;
    assert_eq!(*r2.borrow(), 2);
})
.join()
.unwrap();
```

## When is SendRc needed?

When working inside a single thread, data sharing with `Rc` and optional mutation with
`Cell` and `RefCell` are both safe and convenient. They are also efficient because they
are implemented without atomics and mutexes, allowing the compiler to inline and optimize
away calls to `borrow()` and `borrow_mut()` where they are not globally observable.

If you decouple creation of such hierarchy from its use, it is useful to be able to create
it in one thread and use it in another. After all, `RefCell` and `Cell` are `Send` - they
involve interior mutability, but no sharing. The trouble is with `Rc`, which is neither
`Send` nor `Sync`, and for good reason. Even though it would be perfectly fine to move an
entire hierarchy of `Rc<RefCell>`s from one thread to another, the borrow checker doesn't
allow it because it cannot statically prove that you have moved _all_ of them. If some
remain in the original thread, they'll wreak havoc with unsychronized manipulation of the
reference count.

If there were a way to demonstrate to Rust that you've sent all pointers to a particular
allocation to a different thread, there would be no problem in moving `Rc<T>` instances to
a different thread, provided that `T` itself were `Send`. `SendRc` does exactly that.

## Why not just use Arc?

`Arc` indeed allows moves between threads, but it fundamentally assumes that the
underlying value will be _shared_ between threads. `Arc` requires `T: Send + Sync` in
order for `Arc<T>` to be `Send` because if it only required `T: Send`, you could create an
`Arc<RefCell<u32>>`, clone it, send the clone to a different thread, and call
`borrow_mut()` from two threads on the same `RefCell` without synchronization. That is
forbidden, and is why `Arc<RefCell<T>>` is not a thing in Rust.

`SendRc` can get away with allowing this because it requires proof that all access to the
allocated value in the previous thread was relinquished before allowing the value to be
pinned to a new thread. `SendRc<RefCell<u32>>` is sound because if you clone it and send
the clone to a different thread, you won't be able to access the data, nor clone or even
drop it - any of these would result in a panic.

One could fix the issue by using the full-blown `Arc<Mutex<T>>` or `Arc<RwLock<T>>`.
However, that slows down access to data because it requires atomics, poison checks, and
calls into the pthread API. It also increases the memory overhead because it requires an
extra allocation for the system mutex. Even the most efficient mutex implementations like
`parking_lot` don't come for free, and bear the cost of atomic synchronization. But even
disregarding the cost, the issue is also conceptual: it is simply wrong to use
`Arc<Mutex<T>>` if neither `Arc` nor `Mutex` is actually needed because the code *doesn't*
access the value of `T` from multiple threads in parallel.

In summary, `SendRc<T>` is `Send`, with some guarantees enforced at run time, the same way
an `Arc<Mutex<T>>` is `Send + Sync`, with some guarantees enforced at run time. They just
serve different purposes.

## Why not use an arena? Or unsafe?

An arena would be an acceptable solution, but to make it `Send`, it requires the whole
design to be devoted to that idea from the ground up. A simple solution of replacing `Rc`
with an arena id doesn't really work because in addition to the id, the object needs a
reference to the arena. It can't have an `Option<&Arena>` field because it would make it
non-`Send` for an arena that contains non-`Sync` cells. Since we need `Arena` to contain
`RefCell`, this doesn't work.

There are arena-based designs that do work, but require more radical changes, such as
decoupling storage of values from access and sharing. All data is then in the arena, and
the accessors are created on-the-fly and have a lifetime connected to the lifetime of the
arena. This requires dealing with the lifetime everywhere and is not easy to get right for
non-experts.

Finally, one can avoid the arena by just using `unsafe impl Send` on a root type that is
used to send the whole world to the new thread, and borrow checker be damned. That
solution is hacky and gives up the guarantees afforded by Rust. If you make a mistake, say
by leaving an `Rc` clone in the original thread, you're back to core dumps like in C++. In
Rust we hope to do better, and `SendRc` is an attempt to make such a sound solution that
addresses this scenario.

## What about SendOption?

`SendOption` is an even stranger proposition: a type that holds `Option<T>` and is
_always_ `Send`, regardless of whether `T` is `Send`. Surely that can't be safe?

The idea is that `SendOption` requires you to set it to `None` before sending off the
value to another thread. If the option is `None`, it doesn't matter if `T` is `!Send`
because no `T` is actually getting sent anywhere. If you do send a non-`None`
`SendOption<T>` into another thread, `SendOption` will prevent you from accessing it in
any way (including by dropping it). That way if you cheat, the `T` will still effectively
never have been "sent" to another thread, only memcopied and forgotten, and that's safe.

`SendOption` is designed for types which are composed of `Send` data, except for an
optional field that is not `Send`. The field is set and used only inside a particular
thread, and will be `None` while sent across threads, but since Rust can't prove that, a
field of `Option<NonSend>` makes the entire type not `Send`. For example, a field with a
`SendOption<Rc<Arena>>` could be used to create a `Send` type that refers to a
single-threaded arena.

## Is this really safe?

As with any crate that involves unsafe, one can never be 100% certain that there is no
soundness bug. The code is fairly straightforward in implementing the design outlined
above. I went through several iterations of the design and the implementation before
settling on the current approach and, while I did find the occasional issue, the
underlying idea held up under scrutiny. MIRI finds no undefined behavior while running the
tests.

You are invited to review at the code - it is not large - and report any issues you
encounter.

## Are the run-time checks expensive?

While run-time checks are certainly more expensive than in case of `Rc` and `Option` which
don't need any, they are still quite cheap.

`SendRc::deref()` just checks that the `SendRc` was not disabled (by comparing to a
constant) and compares the id of the pinned-to thread fetched with a relaxed atomic load
with the current thread. The relaxed atomic load compiles to an ordinary load on Intel,
which is as cheap as it gets, and if you're worried, you can hold on to the reference to
avoid repeating the checks. (The borrow checker will prevent you from sending the `SendRc`
to another thread while there is an outstanding reference.) `SendRc::clone()` and
`SendRc::drop()` do the same kind of check.

`SendOption::deref()` and `SendOption::deref_mut()` only check that the current thread is
the pinned-to thread, the same as in `SendRc`.

Regarding memory usage, `SendRc`'s heap overhead is two machine words, the same as that of
an `Rc` (but `SendRc` doesn't support weak references). Additoinally, each individual
`SendRc` is two machine words wide because it has to carry an identity of the pointer.
`SendOption` stores a `u64` alongside the underlying option.

## License

`sendable` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).  See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for
details.  Contributing changes is assumed to signal agreement with these licensing terms.
