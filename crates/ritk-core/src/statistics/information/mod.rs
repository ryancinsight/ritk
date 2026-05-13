//! Information-theoretic measures: entropy, mutual information, variation of
//! information, and total correlation.
//!
//! # Module hierarchy
//!
//! ```text
//! information/
//! ├── entropy.rs            — H(X), H(X,Y), H(X₁,…,Xₙ)
//! ├── mutual_information.rs — I(X;Y), NMI(X,Y)
//! ├── variation_of_information.rs — VI(X,Y)
//! ├── total_correlation.rs  — TC(X₁,…,Xₙ)
//! └── tests/                — unit tests per submodule
//! ```

pub mod entropy;
pub mod mutual_information;
pub mod total_correlation;
pub mod variation_of_information;

pub use entropy::{joint_entropy, joint_entropy_n, marginal_entropy};
pub use mutual_information::{mutual_information, normalized_mutual_information};
pub use total_correlation::total_correlation;
pub use variation_of_information::variation_of_information;

#[cfg(test)]
mod tests;
