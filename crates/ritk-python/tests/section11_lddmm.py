

# Section 11 — LDDMM registration parity vs SimpleITK
# =============================================================================
#
# Validates ritk.registration.lddmm_register() against numpy analytical
# contracts and SimpleITK Demons baselines.
#
# Mathematical specification (Beg et al. 2005):
#   E(v0) = lambda*norm(v0)^2_V + MSE(I∘phi1, J)
#   dE/dv0 = 2*lambda*v0 + K_sigma*[2*(I∘phi1-J)*grad(I∘phi1)]
#
# Invariants tested:
#   (1) Identity fixed-point: v0=0 when fixed==moving => MSE=0, disp=0.
#   (2) Output shape: warped in R^{nz x ny x nx}, disp in R^{3nz x ny x nx}.
#   (3) Finite-valued output for all valid inputs.
#   (4) MSE monotone improvement on a non-trivial pair.
#   (5) NCC monotone improvement: NCC(warped,fixed) >= NCC(moving,fixed).
#   (6) Both RITK LDDMM and SimpleITK Demons reduce MSE from baseline.
# =============================================================================


def _ncc_lddmm(a, b):
    """Pearson NCC: sum(F_tilde * M_tilde) / sqrt(sum(F_tilde^2) * sum(M_tilde^2))."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _make_shifted_sphere_lddmm(shift=2, size=16):
    """Gaussian sphere and its voxel-shifted variant."""
    dims = (size, size, size)
    g = np.indices(dims, dtype=np.float32)
    c = (size - 1) / 2.0
    sphere = np.exp(
        -((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) / (2.0 * (size / 8.0) ** 2)
    ).astype(np.float32)
    shifted = np.roll(sphere, shift, axis=2)
    return sphere, shifted


def _sitk_demons_mse(sphere, shifted, n_iter=20):
    """SimpleITK Demons MSE after registration (reference direction only)."""
    fixed_s = sitk.GetImageFromArray(sphere)
    moving_s = sitk.GetImageFromArray(shifted)
    reg = sitk.DemonsRegistrationFilter()
    reg.SetNumberOfIterations(n_iter)
    reg.SetStandardDeviations(2.0)
    disp = reg.Execute(fixed_s, moving_s)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_s)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.DisplacementFieldTransform(disp))
    warped_s = resampler.Execute(moving_s)
    warped_arr = sitk.GetArrayFromImage(warped_s)
    return float(np.mean((sphere - warped_arr) ** 2))


_HAS_LDDMM = hasattr(ritk, "registration") and hasattr(ritk.registration, "lddmm_register")


@pytest.mark.skipif(not _HAS_LDDMM, reason="ritk.registration.lddmm_register not available")
class TestLddmmRegistrationParity:
    """Section 11: LDDMM registration analytical and SimpleITK direction parity tests."""

    def test_identity_registration_mse_at_zero(self):
        """MSE(lddmm(I, I), I) == 0 — identity fixed-point invariant."""
        arr = np.zeros((8, 8, 8), dtype=np.float32)
        arr[2:6, 2:6, 2:6] = 1.0
        img = ritk.Image(np.ascontiguousarray(arr), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            img, img, max_iterations=3, num_time_steps=2, learning_rate=0.01
        )
        mse = float(np.mean((arr - warped.to_numpy()) ** 2))
        assert mse < 1e-8, f"identity MSE = {mse} exceeds 1e-8"

    def test_warped_shape_equals_fixed_shape(self):
        """Output warped image shape matches fixed image shape."""
        arr = np.random.RandomState(0).rand(10, 10, 10).astype(np.float32)
        fixed = ritk.Image(np.ascontiguousarray(arr), spacing=[1.0, 1.0, 1.0])
        moving = ritk.Image(np.ascontiguousarray(arr), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            fixed, moving, max_iterations=2, num_time_steps=2
        )
        assert warped.to_numpy().shape == arr.shape, (
            f"warped shape {warped.to_numpy().shape} != fixed shape {arr.shape}"
        )

    def test_displacement_field_shape_packed_three_components(self):
        """Displacement field has shape (3*nz, ny, nx) — z/y/x packed along axis-0."""
        nz, ny, nx = 8, 9, 10
        arr = np.zeros((nz, ny, nx), dtype=np.float32)
        img = ritk.Image(np.ascontiguousarray(arr), spacing=[1.0, 1.0, 1.0])
        _, disp = ritk.registration.lddmm_register(img, img, max_iterations=2, num_time_steps=2)
        expected = (3 * nz, ny, nx)
        actual = disp.to_numpy().shape
        assert actual == expected, f"displacement shape {actual} != expected {expected}"

    def test_warped_output_values_all_finite(self):
        """All warped voxel values are finite (no NaN/Inf from geodesic integration)."""
        arr = np.random.RandomState(1).rand(8, 8, 8).astype(np.float32)
        fixed = ritk.Image(np.ascontiguousarray(arr), spacing=[1.0, 1.0, 1.0])
        arr2 = np.roll(arr, 1, axis=1)
        moving = ritk.Image(np.ascontiguousarray(arr2), spacing=[1.0, 1.0, 1.0])
        warped, disp = ritk.registration.lddmm_register(
            fixed, moving, max_iterations=5, num_time_steps=3
        )
        assert np.all(np.isfinite(warped.to_numpy())), "non-finite values in warped output"
        assert np.all(np.isfinite(disp.to_numpy())), "non-finite values in displacement field"

    def test_zero_displacement_for_identical_images(self):
        """For fixed==moving the displacement field is identically zero."""
        arr = np.random.RandomState(2).rand(6, 6, 6).astype(np.float32)
        img = ritk.Image(np.ascontiguousarray(arr), spacing=[1.0, 1.0, 1.0])
        _, disp = ritk.registration.lddmm_register(img, img, max_iterations=5, num_time_steps=2)
        max_disp = float(np.max(np.abs(disp.to_numpy())))
        assert max_disp < 1e-5, f"max displacement {max_disp} for identical images; expected 0"

    def test_mse_improves_after_lddmm_on_shifted_sphere(self):
        """MSE decreases after LDDMM registration on a 2-voxel shifted Gaussian sphere."""
        sphere, shifted = _make_shifted_sphere_lddmm(shift=2, size=16)
        fixed = ritk.Image(np.ascontiguousarray(sphere), spacing=[1.0, 1.0, 1.0])
        moving = ritk.Image(np.ascontiguousarray(shifted), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            fixed, moving,
            max_iterations=20, num_time_steps=5,
            kernel_sigma=2.0, learning_rate=0.05, regularization_weight=0.01,
        )
        mse_before = float(np.mean((sphere - shifted) ** 2))
        mse_after = float(np.mean((sphere - warped.to_numpy()) ** 2))
        assert mse_after < mse_before, (
            f"MSE did not improve: before={mse_before:.6f}, after={mse_after:.6f}"
        )

    def test_ncc_improves_after_lddmm_on_shifted_sphere(self):
        """NCC(warped, fixed) >= NCC(moving, fixed) after LDDMM registration."""
        sphere, shifted = _make_shifted_sphere_lddmm(shift=2, size=16)
        fixed = ritk.Image(np.ascontiguousarray(sphere), spacing=[1.0, 1.0, 1.0])
        moving = ritk.Image(np.ascontiguousarray(shifted), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            fixed, moving,
            max_iterations=20, num_time_steps=5,
            kernel_sigma=2.0, learning_rate=0.05, regularization_weight=0.01,
        )
        ncc_before = _ncc_lddmm(sphere, shifted)
        ncc_after = _ncc_lddmm(sphere, warped.to_numpy())
        assert ncc_after >= ncc_before, (
            f"NCC did not improve: before={ncc_before:.6f}, after={ncc_after:.6f}"
        )

    def test_both_lddmm_and_sitk_demons_reduce_mse(self):
        """Both RITK LDDMM and SimpleITK Demons reduce MSE from baseline (direction parity)."""
        sphere, shifted = _make_shifted_sphere_lddmm(shift=2, size=16)
        mse_baseline = float(np.mean((sphere - shifted) ** 2))

        fixed = ritk.Image(np.ascontiguousarray(sphere), spacing=[1.0, 1.0, 1.0])
        moving = ritk.Image(np.ascontiguousarray(shifted), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            fixed, moving,
            max_iterations=20, num_time_steps=5,
            kernel_sigma=2.0, learning_rate=0.05, regularization_weight=0.01,
        )
        mse_lddmm = float(np.mean((sphere - warped.to_numpy()) ** 2))
        mse_demons = _sitk_demons_mse(sphere, shifted, n_iter=20)

        assert mse_lddmm < mse_baseline, (
            f"RITK LDDMM MSE {mse_lddmm:.6f} >= baseline {mse_baseline:.6f}"
        )
        assert mse_demons < mse_baseline, (
            f"SimpleITK Demons MSE {mse_demons:.6f} >= baseline {mse_baseline:.6f}"
        )

    def test_ncc_in_valid_range_after_lddmm(self):
        """NCC(warped, fixed) in [-1, 1] (bounded Pearson correlation)."""
        sphere, shifted = _make_shifted_sphere_lddmm(shift=1, size=12)
        fixed = ritk.Image(np.ascontiguousarray(sphere), spacing=[1.0, 1.0, 1.0])
        moving = ritk.Image(np.ascontiguousarray(shifted), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            fixed, moving, max_iterations=10, num_time_steps=3
        )
        ncc = _ncc_lddmm(sphere, warped.to_numpy())
        assert -1.0 <= ncc <= 1.0, f"NCC = {ncc} outside [-1, 1]"

    def test_lddmm_warped_positive_ncc_vs_fixed(self):
        """After LDDMM, NCC(warped, fixed) > 0 for a co-modal shifted pair."""
        sphere, shifted = _make_shifted_sphere_lddmm(shift=2, size=16)
        fixed = ritk.Image(np.ascontiguousarray(sphere), spacing=[1.0, 1.0, 1.0])
        moving = ritk.Image(np.ascontiguousarray(shifted), spacing=[1.0, 1.0, 1.0])
        warped, _ = ritk.registration.lddmm_register(
            fixed, moving,
            max_iterations=20, num_time_steps=5,
            kernel_sigma=2.0, learning_rate=0.05, regularization_weight=0.01,
        )
        ncc = _ncc_lddmm(sphere, warped.to_numpy())
        assert ncc > 0.0, f"NCC = {ncc} after LDDMM; expected positive for co-modal pair"
