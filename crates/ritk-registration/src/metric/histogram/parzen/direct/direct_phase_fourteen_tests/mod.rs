//! Phase Fourteen tests — Sprint 329 full joint normalization parity and cleanup.
//!
//! Covers:
//! - SPARSE-329-01: Full joint normalization in sparse path (inv_sum_f in SparseWFixedT)
//! - SPARSE-329-01: Direct↔sparse numerical identity
//! - PERF-329-02: FMA-idiomatic inner loop correctness
//! - MEM-329-04: Structural size regression tests
//! - CLEANUP-329-03: Dead-code annotation correctness

mod identity;
mod normalization;
mod size_and_end_to_end;
