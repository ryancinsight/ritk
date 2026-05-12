//! Slice-index navigation helpers.
//!
//! This module is the SSOT for deterministic slice-index updates used by the
//! viewer shell.
//!
//! # Theorem 1 (clamped step boundedness)
//!
//! For `total >= 1`, define
//! `step_clamped(current, total, delta) = clamp(current + delta, 0, total-1)`.
//!
//! Then:
//! - `0 <= step_clamped < total`
//! - `delta = 0 => step_clamped = clamp(current, 0, total-1)`
//! - the operation is deterministic for fixed inputs.
//!
//! Proof sketch:
//! `clamp(x,a,b)` always returns a value in `[a,b]`, and here
//! `a=0`, `b=total-1`.
//!
//! # Theorem 2 (wrapped advance equivalence)
//!
//! For `total >= 1`, define
//! `advance_wrapped(current, total, steps) = (current + steps) mod total`.
//!
//! Then:
//! - result is in `[0, total)`
//! - `advance_wrapped(current, total, k + m*total) = advance_wrapped(current, total, k)`
//!   for any integer `m >= 0`.
//!
//! Proof sketch:
//! Modular arithmetic over `Z_total` identifies values differing by multiples
//! of `total`.

/// Return axis length for a `[depth, rows, cols]` shape.
pub fn axis_total(shape: [usize; 3], axis: usize) -> usize {
    match axis {
        0 => shape[0],
        1 => shape[1],
        _ => shape[2],
    }
}

/// Clamp `index` to the valid range `[0, total-1]`.
///
/// For `total == 0`, returns `0`.
pub fn clamp_index(index: usize, total: usize) -> usize {
    if total == 0 {
        0
    } else {
        index.min(total - 1)
    }
}

/// Compute next index after a signed step with clamping.
///
/// For `total == 0`, returns `0`.
pub fn step_clamped(current: usize, total: usize, delta: i32) -> usize {
    if total == 0 {
        return 0;
    }
    let max = total.saturating_sub(1) as i32;
    ((current as i32) + delta).clamp(0, max) as usize
}

/// Compute next index after an unsigned wrapped advance.
///
/// For `total == 0`, returns `0`.
pub fn advance_wrapped(current: usize, total: usize, steps: u32) -> usize {
    if total == 0 {
        return 0;
    }
    (current + steps as usize) % total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_total_matches_depth_row_col() {
        let shape = [8, 10, 20];
        assert_eq!(axis_total(shape, 0), 8);
        assert_eq!(axis_total(shape, 1), 10);
        assert_eq!(axis_total(shape, 2), 20);
        assert_eq!(axis_total(shape, 99), 20);
    }

    #[test]
    fn clamp_index_is_bounded_and_idempotent() {
        assert_eq!(clamp_index(3, 10), 3);
        assert_eq!(clamp_index(99, 10), 9);
        assert_eq!(clamp_index(clamp_index(99, 10), 10), 9);
    }

    #[test]
    fn step_clamped_respects_bounds() {
        assert_eq!(step_clamped(5, 10, 3), 8);
        assert_eq!(step_clamped(5, 10, -3), 2);
        assert_eq!(step_clamped(0, 10, -3), 0);
        assert_eq!(step_clamped(9, 10, 3), 9);
    }

    #[test]
    fn advance_wrapped_respects_modular_equivalence() {
        let total = 7;
        let current = 3;
        let k = 5;
        let m_total = k + 2 * total as u32;
        assert_eq!(advance_wrapped(current, total, k), 1);
        assert_eq!(
            advance_wrapped(current, total, m_total),
            advance_wrapped(current, total, k)
        );
    }

    #[test]
    fn zero_total_returns_zero_for_all_helpers() {
        assert_eq!(clamp_index(5, 0), 0);
        assert_eq!(step_clamped(5, 0, -10), 0);
        assert_eq!(advance_wrapped(5, 0, 99), 0);
    }
}
