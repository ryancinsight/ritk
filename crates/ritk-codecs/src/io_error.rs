//! The `std::io::Error` form of a format reader's or writer's `anyhow` error.
//!
//! Format crates report failures as `anyhow::Error` chains, while `Read`- and
//! `Write`-shaped contracts return `std::io::Error`. The conversion keeps the
//! whole chain as the source and the kind of the I/O failure at its root, so
//! a caller still tells a missing file (`NotFound`) from a malformed one
//! (`Other`).

use std::io;

/// `error` as an I/O error carrying its whole chain, with the kind of the I/O
/// failure at its root, or [`io::ErrorKind::Other`] when no I/O call failed.
///
/// # Examples
///
/// ```
/// use std::io;
///
/// let missing = anyhow::Error::new(io::Error::from(io::ErrorKind::NotFound))
///     .context("Failed to read header");
/// let error = ritk_codecs::into_io_error(missing);
/// assert_eq!(error.kind(), io::ErrorKind::NotFound);
/// assert_eq!(error.to_string(), "Failed to read header");
///
/// let malformed = anyhow::anyhow!("bad magic");
/// assert_eq!(ritk_codecs::into_io_error(malformed).kind(), io::ErrorKind::Other);
/// ```
#[must_use]
pub fn into_io_error(error: anyhow::Error) -> io::Error {
    let kind = error
        .root_cause()
        .downcast_ref::<io::Error>()
        .map_or(io::ErrorKind::Other, io::Error::kind);
    io::Error::new(kind, error)
}

#[cfg(test)]
mod tests {
    use super::into_io_error;
    use std::io;

    /// Every layer's message survives, walking the source chain from the
    /// returned error.
    #[test]
    fn the_chain_survives_the_conversion() {
        let error = anyhow::Error::new(io::Error::from(io::ErrorKind::PermissionDenied))
            .context("inner context")
            .context("outer context");
        let converted = into_io_error(error);
        assert_eq!(converted.kind(), io::ErrorKind::PermissionDenied);
        let inner = converted
            .get_ref()
            .expect("a custom error carries its payload");
        let mut messages = vec![inner.to_string()];
        let mut cause = inner.source();
        while let Some(current) = cause {
            messages.push(current.to_string());
            cause = current.source();
        }
        assert_eq!(messages[..2], ["outer context", "inner context"]);
        assert_eq!(messages.len(), 3, "{messages:?}");
    }
}
