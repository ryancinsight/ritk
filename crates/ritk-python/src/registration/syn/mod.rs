//! SyN-family and LDDMM registration bindings.
//!
//! This module keeps the PyO3 boundary stable while splitting each algorithm
//! family into its own leaf file to satisfy the structural size limit.

mod bspline_ffd;
mod bspline_syn;
mod greedy;
mod lddmm;
mod multires;
mod shared;

pub use bspline_ffd::*;
pub use bspline_syn::*;
pub use greedy::*;
pub use lddmm::*;
pub use multires::*;
