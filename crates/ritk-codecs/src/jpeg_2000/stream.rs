//! OpenJPEG in-memory read stream backed by a byte-slice buffer.
//!
//! # Architecture
//! All unsafe FFI pointer operations are isolated in this module. The public
//! surface exposes only `J2kMemStream`, a safe Rust struct that owns its buffer.
//! A caller must keep the `J2kMemStream` alive for the entire lifetime of the
//! `*mut opj_stream_t` it vends — both are destroyed before `J2kMemStream` drops.
//!
//! # Safety boundaries
//! Every `unsafe extern "C"` callback documents:
//! (a) the invariant that `p_user_data` is a valid `*mut J2kMemStream`,
//! (b) the range guarantees on output-pointer validity,
//! (c) why no aliasing or use-after-free can occur.

use openjpeg_sys as opj;
use std::ffi::c_void;

// ─── In-memory stream state ───────────────────────────────────────────────────

/// In-memory JPEG 2000 read-stream state.
///
/// Owns the byte buffer and current read position.  Must outlive the
/// `*mut opj_stream_t` created by [`J2kMemStream::create_opj_stream`].
pub(super) struct J2kMemStream {
    data: Vec<u8>,
    pos: usize,
}

impl J2kMemStream {
    /// Wrap `data` in a new stream positioned at byte 0.
    pub(super) fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    /// Create an `opj_stream_t` backed by `self`.
    ///
    /// # Safety
    /// The returned stream **must** be destroyed with `opj_stream_destroy` before
    /// `self` is dropped.  Destroying the stream does **not** free the buffer; the
    /// `user_data_free_fn` is set to `None` precisely to prevent a double-free.
    pub(super) unsafe fn create_opj_stream(&mut self) -> *mut opj::opj_stream_t {
        // SAFETY: OPJ_STREAM_READ (=1) expressed as OPJ_BOOL; buf size is the
        // exact data length so OpenJPEG never requests more in one read call than
        // the entire buffer.
        let stream = opj::opj_stream_create(
            self.data.len() as opj::OPJ_SIZE_T,
            opj::OPJ_TRUE as opj::OPJ_BOOL,
        );
        assert!(!stream.is_null(), "opj_stream_create returned null");

        // SAFETY: `self` is mutably borrowed by the caller for the duration the
        // stream is live; no other reference to `self` exists concurrently.
        opj::opj_stream_set_user_data(stream, self as *mut Self as *mut c_void, None);
        opj::opj_stream_set_user_data_length(stream, self.data.len() as opj::OPJ_UINT64);
        opj::opj_stream_set_read_function(stream, Some(read_fn));
        opj::opj_stream_set_skip_function(stream, Some(skip_fn));
        opj::opj_stream_set_seek_function(stream, Some(seek_fn));
        stream
    }
}

// ─── OpenJPEG I/O callbacks ───────────────────────────────────────────────────

/// OpenJPEG read callback.
///
/// Copies `min(p_nb_bytes, remaining)` bytes from the stream into `p_buffer`.
/// Returns the count of bytes actually copied.  Returns `OPJ_SIZE_T::MAX` when
/// the stream is exhausted (OpenJPEG convention for end-of-stream).
///
/// # Safety
/// OpenJPEG guarantees:
/// - `p_user_data` is the pointer set via `opj_stream_set_user_data`.
/// - `p_buffer` points to a writeable region of exactly `p_nb_bytes` bytes.
/// Both invariants are maintained by the OpenJPEG library itself.
unsafe extern "C" fn read_fn(
    p_buffer: *mut c_void,
    p_nb_bytes: opj::OPJ_SIZE_T,
    p_user_data: *mut c_void,
) -> opj::OPJ_SIZE_T {
    // SAFETY: `p_user_data` was set to `self as *mut J2kMemStream` and OpenJPEG
    // never modifies or frees it between `set_user_data` and stream destroy.
    let stream = &mut *(p_user_data as *mut J2kMemStream);

    let remaining = stream.data.len().saturating_sub(stream.pos);
    if remaining == 0 {
        // OpenJPEG end-of-stream sentinel (ISO 15444-1 Annex D: OPJ_SIZE_T(-1)).
        return OPJ_STREAM_EOF;
    }
    let to_read = p_nb_bytes.min(remaining);
    // SAFETY: `p_buffer` is a valid write destination of `p_nb_bytes` bytes.
    std::ptr::copy_nonoverlapping(
        stream.data.as_ptr().add(stream.pos),
        p_buffer as *mut u8,
        to_read,
    );
    stream.pos += to_read;
    to_read
}

/// OpenJPEG skip callback.
///
/// Advances the read position by `p_nb_bytes` (clamped to the buffer boundary).
/// Returns the number of bytes actually skipped.  A negative `p_nb_bytes` is
/// treated as 0 — OpenJPEG never calls skip with a negative value but the
/// contract is documented for completeness.
///
/// # Safety
/// Same user-data invariant as [`read_fn`].
unsafe extern "C" fn skip_fn(
    p_nb_bytes: opj::OPJ_OFF_T,
    p_user_data: *mut c_void,
) -> opj::OPJ_OFF_T {
    let stream = &mut *(p_user_data as *mut J2kMemStream);
    let requested = p_nb_bytes.max(0) as usize;
    let available = stream.data.len().saturating_sub(stream.pos);
    let to_skip = requested.min(available);
    stream.pos += to_skip;
    to_skip as opj::OPJ_OFF_T
}

/// OpenJPEG seek callback.
///
/// Sets the read position to the absolute byte offset `p_nb_bytes`.
/// Returns `OPJ_TRUE` on success, `OPJ_FALSE` when the offset exceeds the
/// buffer length.
///
/// # Safety
/// Same user-data invariant as [`read_fn`].
unsafe extern "C" fn seek_fn(
    p_nb_bytes: opj::OPJ_OFF_T,
    p_user_data: *mut c_void,
) -> opj::OPJ_BOOL {
    let stream = &mut *(p_user_data as *mut J2kMemStream);
    let pos = if p_nb_bytes < 0 {
        return opj::OPJ_FALSE as opj::OPJ_BOOL;
    } else {
        p_nb_bytes as usize
    };
    if pos <= stream.data.len() {
        stream.pos = pos;
        opj::OPJ_TRUE as opj::OPJ_BOOL
    } else {
        opj::OPJ_FALSE as opj::OPJ_BOOL
    }
}

/// Sentinel returned by [`read_fn`] to signal end-of-stream (ISO 15444-1).
const OPJ_STREAM_EOF: opj::OPJ_SIZE_T = opj::OPJ_SIZE_T::MAX;
