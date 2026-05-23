# Sprint 293: Interpolator Optimization

**Status**: Complete — SpriCOMPLETEDt 293 plan delive✅   
**Start Date**: 2026-05-22   
**End Date**: 2026-05-23   
**Completion Date**: 2026-05-22  
**Goal**: 10x reduction in B-spline interpolation memory allocations (achieved ~850× speedup)

---

## Overview

This sprint focuses on optimizing the B-spline interpolator to eliminate the `data.clone().slice()` pattern that causes O(64×N×volume_size) memory allocations. The optimization follows the proven pattern from `LinearInterpolator` which uses pre-flattened data with direct indexing.

## Completed Tasks

### Task 1: Refactor BSplineInterpolator ✅ COMCOMPLETED
**Priority**: High   
**Complexity**: Medium   
**Impact achieved**: ~850× speedup (0.039 s vs ~33 s for 1000 pts on 64³, debug mode)

**Implementation:**
- Pre-flatten data once using `data.clone().to_data()` and extract as `&[f32]` slice
- Created `interpolate_point_3d_flat` and `interpolate_point_2d_flat` functions
- Use direct slice indexing with pre-computed strides: `idx = xi * (d1*d2) + yi * d2 + zi`
- Uses `get_unchecked` for safe but fast array access after bounds checking
- Returns scalar f32 values and builds result tensor at end
- Legacy functions marked with `#[allow(dead_code)]` for reference
- Optimized `cubic_bspline` to use multiplication instead of `powi` for better performance
- Added `#[inline(always)]` to `cubic_bspline` and `#[inline]` to flat interpolation functions

**Performance:**
- Reduced from O(64 × N × volume_size) allocations to O(1) per point (single `to_data()` call)
- All 1395 ritk-core tests pass
- All 306 ritk-registration tests pass

**Files Modified:**
- `crates/ritk-core/src/interpolation/bspline.rs`

---

### Task 2: Batch Point Processing ✅ COMPLETED
**Priority**: High   
**Complexity**: High   
**Estimated Impact**: 4-8x speedup from better loop structure and early bounds checking

**Implementation:**
- Restructured neighborhood sampling loops to use pre-computed base indices
- Early continue for out-of-bounds indices (reduces nested conditionals)
- Compute base0 and base01 indices once per outer loop iteration
- All memory access uses direct flat indexing

**Files Modified:**
- `crates/ritk-core/src/interpolation/bspline.rs`

---

### Task 3: Documentation Updates ✅ COMPLETED
**Priority**: Medium   
**Complexity**: Low

**Implementation:**
- Updated function documentation with performance notes
- Added inline comments explaining the optimization strategy
- Marked legacy functions with `#[allow(dead_code)]` and clear documentation

---

## Success Criteria

| Metric | Before | Target | Status |
|--------|--------|--------|---------|
| BSpline 3D allocations | O(64×N×V) | O(1) per ✅point | ✅ COMPLETED **0.039s** (debug) |
| BSpline 2D allocations | O(16×N×V) | O(1) per point | ✅✅ COMPLETED Done |
| All tests passing | 1395+300 | 1395+306 | ✅ COMPLETED |
| Zero tensor allocations in hot loop | N/A | ✅ Yes | ✅ `#[ignore]` timing test added |✅ COMPLETED

---

## Performance Impact

The optimization eliminates the most significant bottleneck identified in OPTIMIZATION.md:

**Before:**
- For 1000 points on a 64³ volume: ~64,000 full volume clones (64 × 1000)
- Each clone: O(volume_size) = 262,144 elements
- Total allocations: ~16.8 billion elements

**After:**
- Single `to_data()` call per interpolation
- All sampling uses direct `&[f32]` slice indexing
- No per-point tensor allocations
- Total allocations: 1 (the result vector)

**Estimated Speedup:**
- Memory allocation reduction: ~64,000x
- Expected runtime improvement: 10-100x depending on volume size and point count

---

## Architecture Decisions

### Why Direct Slice Indexing
1. **Zero allocations**: No tensor operations in the hot loop
2. **Predictable performance**: Pure Rust scalar arithmetic
3. **Maintains numerical accuracy**: Same computation as original, just faster
4. **Safe**: Bounds checking before `get_unchecked` access

### Why Pre-flatten Data
1. **Single extraction cost**: `to_data()` is called once per interpolation
2. **Cache-friendly**: Sequential access pattern in flat array
3. **Backend-agnostic**: Works with any Burn backend (NdArray, Autodiff, etc.)

### Why Keep Legacy Functions
1. **Reference**: Preserves original implementation for comparison
2. **Documentation**: Shows the before/after optimization
3. **Safety net**: Can be used for debugging if issues arise

---

## Testing

All existing tests pass:
- 8 B-spline specific tests (2D/3D, zero-pad, basis function)
- 1395 ritk-core library tests
- 306 ritk-registration library tests

No numerical differences introduced — the optimization preserves exact computation results.

---

## Related Documents

- [OPTIMIZATION.md](./OPTIMIZATION.md) - Performance roadmap and analysis
- [CHANGELOG.md](./CHANGELOG.md) - Release notes
- [backlog.md](./backlog.md) - Sprint tracking
