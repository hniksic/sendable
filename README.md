# sendable

The `sendable` crate defines types to facilitate sending data between threads:

* `SendRc`, a single-threaded reference-counting pointer that can be sent between
  threads. You can think of it as a variant of `Rc<T>` that is `Send` if `T` is
  `Send`. This is unlike `Rc<T>` which is never `Send`, and also unlike `Arc<T>`, which
  requires `T: Send + Sync` to be `Send`.
* `SendOption`, which holds an `Option<T>` and is `Send` even if `T` is not `Send`.

## When is SendRc useful?

You might consider `SendRc` if:

* your values form an acyclic graph or a hierarchy with cross-references;
* you build and use the hierarchy from a single thread;
* you need to occasionally move the whole thing to another thread.

Within the confines of a single thread, using `Rc` and `RefCell` to represent acyclic
graphs and data sharing is ergonomic and safe. It is also efficient because
single-threaded manipulation doesn't require atomics or locks, makes `deref()` trivial,
and allows the compiler to inline `borrow()` and `borrow_mut()` and even optimize them
away where they are not globally observable.

In programs that process many such graphs it comes in very useful to be able to create
them in one thread and ship them to another for processing (and possibly to a third one
for destruction). Given that types like `RefCell` and `Cell` are `Send`, the idea is not
unthinkable. The trouble is with `Rc`, which is neither `Send` nor `Sync`, and for good
reason. Even though it would be perfectly safe to move an entire hierarchy of
`Rc<RefCell<T>>`s from one thread to another, the borrow checker doesn't allow it because
it cannot statically prove that you have moved _all_ of them. If some `Rc`s pointing to
shared data remained in the original thread, unsynchronized access to the non-`Sync` cells
and unsynchronized manipulation of the reference counts would be undefined behavior and
wreak havoc.

If there were a way to demonstrate to Rust that you've sent all pointers to a particular
shared value to a different thread, there would be no problem in moving `Rc<T>` instances
to a different thread, provided that `T` itself were `Send`. `SendRc` does exactly that.

## How does SendRc work?

When a `SendRc` is constructed, it stores the current thread id next to the value and the
reference count. On access to the value, and before manipulating the reference count
through `clone()` and `drop()`, it checks that the `SendRc` is still in the thread it was
created in.

Before `SendRc`s are moved to a different thread, each pointer is explicitly "parked",
i.e. registered for sending. Once parked, access to the value it points to is prohibited,
even in the original thread. When all `SendRc`s pointing to the shared value are parked,
they can be sent across the thread boundary, and re-enabled in the new thread. In a simple
case of two `SendRc`s, the process looks like this:

```rust
// create two SendRcs pointing to the same shared value
let mut r1 = SendRc::new(RefCell::new(1));
let mut r2 = SendRc::clone(&r1);

// prepare to ship them off to a different thread
let pre_send = SendRc::pre_send();
pre_send.park(&mut r1); // r1 and r2 cannot be dereferenced from this point
pre_send.park(&mut r2);
// ready() would panic if there were unparked SendRcs pointing to the shared value
let mut post_send = pre_send.ready();

// move everything to a different thread
std::thread::spawn(move || {
    // both pointers are unusable here
    post_send.unpark();
    // they're again usable from this point, but only in this thread
    *r1.borrow_mut() += 1;
    assert_eq!(*r2.borrow(), 2);
})
.join()
.unwrap();
```

## Why not just use Arc?

`Arc` indeed allows moves between threads, but it fundamentally assumes that the
underlying value will be _shared_ between threads. `Arc` requires `T: Send + Sync` in
order for `Arc<T>` to be `Send` because if it only required `T: Send`, you could create an
`Arc<RefCell<u32>>`, clone it, send the clone to a different thread, and call
`borrow_mut()` from two threads on the same `RefCell` without synchronization. That is
forbidden, and is why `Arc<RefCell<T>>` is not a thing in Rust.

`SendRc` can get away with allowing this because it guards access to the data with a check
of the current thread. When moving data across threads, it requires proof that all access
to the allocated value in the previous thread was relinquished prior to the move.
`SendRc<RefCell<u32>>` is sound because if you clone it and send the clone to a different
thread, you won't be able to access the data, nor clone or even drop it - any of those
would result in a panic.

Using the standard library, one could fix the issue by switching to the full-blown
`Arc<Mutex<T>>` or `Arc<RwLock<T>>`.  However, that slows down access to data because it
requires strongly-ordered atomics, poison checks, and calls into the pthread API. It also
increases memory overhead because due to the mandatory allocation of the system mutex.
Even the most efficient mutex implementations like `parking_lot` don't come for free and
bear the cost of synchronization. But even disregarding the cost, on a conceptual level
it's simply wrong to use `Arc<Mutex<T>>` if neither `Arc` nor `Mutex` are actually needed
because the code *doesn't* access the value of `T` from multiple threads in parallel.

In summary, `SendRc<T>` is `Send` with certains guarantees enforced at run time, the same
way an `Arc<Mutex<T>>` is `Send + Sync` with certain guarantees enforced at run time. They
just serve different purposes.

## Why not use an arena? Or unsafe?

To make an arena `Send`, the whole design must be devoted to that idea from the ground up.
A simple solution of replacing `Rc` with an arena id doesn't really work because in
addition to the id, the object then needs a reference to the arena. It can't have a field
of type `Option<&Arena>` or `Option<Rc<Arena>>` because it would make the type non-`Send`
if the arena contains `RecCell`.

There are arena-based designs that do work, but require more radical changes, such as
decoupling storage of values from access and sharing. All data is then in the arena, and
the accessors are created on-the-fly and have a lifetime connected to the lifetime of the
arena. This requires dealing with the lifetime everywhere and is not easy to get right for
non-experts.

Finally, one can avoid the arena by just using `unsafe impl Send` on a wrapper type that
is used to send the whole world to the new thread, borrow checker be damned. That solution
is hacky and gives up the guarantees afforded by Rust. If you make a mistake, say by
leaving an `Rc` clone in the original thread, you're facing undefined behavior and core
dumps much like in C++. In Rust we hope to do better, and `SendRc` is intended to provide
a sound solution that addresses this scenario.

## What about SendOption?

`SendOption` is a related proposition: a type that holds `Option<T>` and is _always_
`Send`, regardless of whether `T` is `Send`. Surely that can't be safe?

What makes it work is that `SendOption` requires you to set the value to `None` before
sending it to another thread. If the inner `Option<T>` is `None`, it doesn't matter if `T`
is not `Send` because no `T` is actually getting sent anywhere. If you do send a
non-`None` `SendOption<T>` into another thread, `SendOption` will use panic to prevent you
from accessing it in any way (including by dropping it). Failure to abide by the rules
results in a `T` that was effectively never "sent" to another thread, only its bits were
shallow-copied and forgotten, and that's safe.

`SendOption` is designed for types which are composed of `Send` data, except for an
optional field of a non-send type. The field is set and used only inside a particular
thread, and will be `None` while being sent across threads, but since Rust can't prove
that, a field of `Option<NonSendType>` makes the entire outer type not `Send`. For
example, a field with a `SendOption<Rc<Arena>>` could be used to create a `Send` type that
refers to a single-threaded arena.

## Is this really safe?

As with any crate that involves unsafe, one can never be 100% certain that there is no
soundness bug. The code is fairly straightforward in implementing the design outlined
above. I went through several iterations of the design and the implementation before
settling on the current approach and, while I did find the occasional issue, the
underlying idea held up under scrutiny. MIRI finds no undefined behavior while running the
tests.

You are invited to review the code - it is not large - and report any issues you
encounter.

## Are the run-time checks expensive?

While run-time checks performed by `SendRc` and `SendOption` are certainly more expensive
than those of `Rc` and `Option`, which are non-existrent, they are still reasonably cheap.

`SendRc::deref()` just compares the id of the pinned-to thread fetched with a relaxed
atomic load with the current thread, and checks that migration isn't in progress with an
integer comparison. The relaxed atomic load compiles to an ordinary load on Intel, which
is as cheap as it gets, and if you're worried, you can hold on to the reference to avoid
repeating the checks. (The borrow checker will prevent you from sending the `SendRc` to
another thread while there is an outstanding reference.) `SendRc::clone()` and
`SendRc::drop()` do the same kind of check.

`SendOption::deref()` and `SendOption::deref_mut()` only check that the current thread is
the pinned-to thread, the same as in `SendRc`.

Regarding memory usage, `SendRc`'s heap overhead is two `u64`s for the pinning info, and a
machine word for the reference count, i.e. on 64-bit architecture it's one `u64` more than
`Rc`. An individual `SendRc` is two machine words wide because it has to carry an identity
of the pointer. `SendOption` stores a `u64` alongside the underlying option.

## License

`sendable` is distributed under the terms of both the MIT license and the Apache License
(Version 2.0).  See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for
details.  Contributing changes is assumed to signal agreement with these licensing terms.
