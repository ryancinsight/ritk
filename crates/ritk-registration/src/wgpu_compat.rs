//! WGPU dispatch-limit constant for ritk-registration.
//!
//! Re-exported from `ritk_core::wgpu_compat` to ensure both crates
//! always use the same dispatch limit. Update the value in ritk-core only.
pub(crate) use ritk_core::wgpu_compat::WGPU_CHUNK_SIZE;
