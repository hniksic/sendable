//! `Rc` and `Option` equivalents made that facilitate sending data between threads.
//!
//! This crate provides two types:
//! * `SendRc<T>`, a single-threaded reference-counted pointer like `Rc` that is `Send`
//!   if `T` is `Send`.
//! * `SendOption<T>`, a container like `Option` that is `Send` even if `T` is not `Send`.
//!
//! Both types rely on run-time checks to enforce Rust's safety guarantees. `SendRc`
//! requires the pointers to be disabled before sending to a different thread, and
//! re-enabled at destination. `SendOption` requires the option to be set to `None` before
//! sending to another thread, and then to be explicitly marked as sent.
//!
//! `SendRc` is designed for constructing single-threaded hierarchies that support data
//! sharing and interior mutability (`Cell` and `RefCell`), and which needs to be
//! transferred _en masse_ to a new thread. `Rc` doesn't allow that because it is not
//! `Send`, and `Arc` doesn't allow that because it requires `T: Sync`.
//!
//! `SendOption` is designed for optional non-`Send` fields in otherwise `Send` types.

#![warn(missing_docs)]

mod send_opt;
mod send_rc;
mod thread_id;

#[cfg(feature = "deepsize")]
mod deepsize;

pub use send_opt::SendOption;
pub use send_rc::{PostSendRc, PreSendRc, SendRc};
