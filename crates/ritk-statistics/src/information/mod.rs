//! Information-theoretic measures: entropy, mutual information, variation of
//! information, total correlation, and O-information.
//!
//! # Module hierarchy
//!
//! ```text
//! information/
//! ├── entropy.rs            — H(X), H(X,Y), H(X₁,…,Xₙ)
//! ├── mutual_information.rs — I(X;Y), NMI(X,Y), I(X;Y|Z), II(X;Y;Z)
//! ├── variation_of_information.rs — VI(X,Y), VI_n(X₁,…,Xₙ)
//! ├── total_correlation.rs  — TC(X₁,…,Xₙ)         (Watanabe 1960)
//! ├── o_information.rs      — DTC(X₁,…,Xₙ), Ω     (Han 1978; Rosas 2019)
//! └── tests/                — unit tests per submodule
//! ```

pub mod entropy;
pub mod mutual_information;
pub mod o_information;
pub mod total_correlation;
pub mod variation_of_information;

pub use entropy::{joint_entropy, joint_entropy_n, marginal_entropy};
pub use mutual_information::{
    conditional_mutual_information, interaction_information, mutual_information,
    mutual_information_mattes, normalized_mutual_information, symmetric_uncertainty,
};
pub use o_information::{
    dual_total_correlation, o_information, o_information_direct, o_information_from_tc_dtc,
};
pub use total_correlation::total_correlation;
pub use variation_of_information::{
    multivariate_variation_of_information, variation_of_information,
};

#[cfg(test)]
mod tests;
