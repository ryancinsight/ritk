//! Trigonometric intensity transforms.
//!
//! These public names re-export the canonical sealed unary ZST operations.
//! Their implementation and Coeus-native boundary live in
//! [`crate::intensity::arithmetic::unary`].

pub use crate::intensity::arithmetic::{
    AcosImageFilter, AsinImageFilter, AtanImageFilter, BoundedReciprocalImageFilter,
    CosImageFilter, SinImageFilter, TanImageFilter,
};

#[cfg(test)]
#[path = "tests_trig.rs"]
mod tests_trig;
