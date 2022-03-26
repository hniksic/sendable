///! Optional support for the deepsize crate.

use crate::{SendRc, SendOption};

impl<T> deepsize::DeepSizeOf for SendRc<T>
where
    T: deepsize::DeepSizeOf,
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        (&**self).deep_size_of_children(context)
    }
}

impl<T> deepsize::DeepSizeOf for SendOption<T>
where
    T: deepsize::DeepSizeOf,
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        (&**self).deep_size_of_children(context)
    }
}
