//! Coeus-native differentiable registration operations.

pub mod autodiff;
pub mod dl_losses;
pub mod lncc;
pub mod mse;
pub mod ncc;
pub mod ngf;

pub use lncc::lncc_loss_native;
pub use mse::mse_value_native;
pub use ncc::ncc_loss_native;
pub use ngf::{ngf_value_native, NgfFixedPrepNative};

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests;
