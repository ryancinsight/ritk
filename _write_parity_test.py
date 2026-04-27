
import pathlib, textwrap

DEST = pathlib.Path("crates/ritk-python/tests/test_simpleitk_parity.py")

FILTER_TESTS = textwrap.dedent("""

# =========================================================================
# Section 1 -- Filter parity
# =========================================================================

def test_discrete_gaussian_interior_agrees_with_sitk():
    """DiscreteGaussian interior voxels agree within 0.01 vs SimpleITK.

    Both parameterised identically: variance=4.0, maximum_error=0.01,
    use_image_spacing=False.
    Kernel radius = ceil(sqrt(-2*4*ln(0.01))) = 7 voxels.
    Interior crop [8:-8] eliminates all boundary-condition differences.

    Tolerances:
        interior max absolute diff < 0.01
        global mean absolute diff  < 0.005
    """
    arr = _make_gradient()
    sr = _np(sitk.DiscreteGaussian(_sitk(arr), variance=4.0,
                                   maximumError=0.01, useImageSpacing=False))
    rr = ritk.filter.discrete_gaussian(_ritk(arr), variance=4.0,
                                       maximum_error=0.01,
                                       use_image_spacing=False).to_numpy()
    assert sr.shape == rr.shape
    m = 8
    diff_i = np.abs(sr[m:-m, m:-m, m:-m] - rr[m:-m, m:-m, m:-m])
    assert float(diff_i.max()) < 0.01, (
        f"DiscreteGaussian interior max diff {diff_i.max():.5f} > 0.01")
    assert float(np.abs(sr - rr).mean()) < 0.005, (
        f"DiscreteGaussian global mean diff {np.abs(sr-rr).mean():.5f} > 0.005")


def test_discrete_gaussian_constant_image_invariant():
    """DiscreteGaussian of constant=0.5 returns 0.5 everywhere.

    Invariant: conv(c, normalised_kernel) = c.
    Tolerance: max absolute deviation from 0.5 < 1e-4.
    """
    arr = np.full((SIZE, SIZE, SIZE), 0.5, dtype=np.float32)
    sr = _np(sitk.DiscreteGaussian(_sitk(arr), variance=4.0,
                                   maximumError=0.01, useImageSpacing=False))
    rr = ritk.filter.discrete_gaussian(_ritk(arr), variance=4.0,
                                       maximum_error=0.01,
                                       use_image_spacing=False).to_numpy()
    assert float(np.abs(sr - 0.5).max()) < 1e-4, (
        f"SimpleITK DiscreteGaussian constant deviation {np.abs(sr-0.5).max():.6f}")
    assert float(np.abs(rr - 0.5).max()) < 1e-4, (
        f"RITK DiscreteGaussian constant deviation {np.abs(rr-0.5).max():.6f}")


def test_median_filter_radius1_agrees_with_sitk():
    """MedianFilter radius=1 (3x3x3) matches SimpleITK to float32 precision.

    Both sort 27 neighbours and return lower median with replicate boundary.
    Tolerance: max absolute difference < 1e-4.
    """
    arr = _make_noisy()
    sr = _np(sitk.Median(_sitk(arr), [1, 1, 1]))
    rr = ritk.filter.median_filter(_ritk(arr), radius=1).to_numpy()
    assert sr.shape == rr.shape
    assert float(np.abs(sr - rr).max()) < 1e-4, (
        f"MedianFilter max diff {np.abs(sr-rr).max():.6f} > 1e-4")


def test_gradient_magnitude_interior_matches_analytical():
    """GradientMagnitude on ramp equals 1/(SIZE-1) in the interior.

    Analytical: f(z,y,x)=x/(SIZE-1)  =>  ||grad f|| = 1/(SIZE-1).
    Both use central finite differences.

    Tolerances:
        interior mean vs analytical < 0.003 per implementation
        mutual interior max diff    < 0.01
    """
    arr = _make_gradient()
    analytical = 1.0 / (SIZE - 1)
    sr = _np(sitk.GradientMagnitude(_sitk(arr)))
    rr = ritk.filter.gradient_magnitude(_ritk(arr)).to_numpy()
    m = 2
    si = sr[m:-m, m:-m, m:-m]
    ri = rr[m:-m, m:-m, m:-m]
    assert abs(float(si.mean()) - analytical) < 0.003, (
        f"SimpleITK GM interior mean {si.mean():.5f} vs analytical {analytical:.5f}")
    assert abs(float(ri.mean()) - analytical) < 0.003, (
        f"RITK GM interior mean {ri.mean():.5f} vs analytical {analytical:.5f}")
    assert float(np.abs(si - ri).max()) < 0.01, (
        f"GradientMagnitude mutual max diff {np.abs(si-ri).max():.5f} > 0.01")


def test_rescale_intensity_agrees_with_sitk():
    """RescaleIntensity [0,1] is numerically identical to SimpleITK.

    Analytical: output = (v - v_min) / (v_max - v_min).
    Tolerance: max absolute diff < 1e-4.
    """
    arr = _make_noisy()
    filt = sitk.RescaleIntensityImageFilter()
    filt.SetOutputMinimum(0.0)
    filt.SetOutputMaximum(1.0)
    sr = _np(filt.Execute(_sitk(arr)))
    rr = ritk.filter.rescale_intensity(_ritk(arr), out_min=0.0, out_max=1.0).to_numpy()
    assert float(rr.min()) >= -1e-5
    assert float(rr.max()) <= 1.0 + 1e-5
    assert float(np.abs(sr - rr).max()) < 1e-4, (
        f"RescaleIntensity max diff {np.abs(sr-rr).max():.6f} > 1e-4")


def test_binary_threshold_agrees_with_sitk_and_analytical():
    """BinaryThreshold(0.3,0.7,fg=1,bg=0) matches analytical and SimpleITK.

    Analytical: output=1.0 iff 0.3<=v<=0.7.
    Gradient image: f(z,y,x)=x/(SIZE-1), threshold maps to X slices.
    Tolerances: vs analytical < 1e-4; mutual < 1e-4.
    """
    arr = _make_gradient()
    sr = _np(sitk.BinaryThreshold(_sitk(arr), lowerThreshold=0.3,
                                  upperThreshold=0.7,
                                  insideValue=1.0, outsideValue=0.0)).astype(np.float32)
    rr = ritk.filter.binary_threshold(_ritk(arr), lower_threshold=0.3,
                                      upper_threshold=0.7,
                                      foreground=1.0, background=0.0).to_numpy()
    expected = ((arr >= 0.3) & (arr <= 0.7)).astype(np.float32)
    assert float(np.abs(sr - expected).max()) < 1e-4
    assert float(np.abs(rr - expected).max()) < 1e-4
    assert float(np.abs(sr - rr).max()) < 1e-4


def test_grayscale_erosion_box_interior_agrees_with_sitk():
    """GrayscaleErosion box SE radius=1 interior matches SimpleITK sitkBox.

    RITK: (2r+1)^3 cubic SE, replicate boundary.
    SimpleITK: GrayscaleErode with kernelType=sitkBox.
    Tolerance: interior max absolute diff < 1e-4.
    """
    arr = _make_gradient()
    sr = _np(sitk.GrayscaleErode(_sitk(arr), [1, 1, 1], sitk.sitkBox))
    rr = ritk.filter.grayscale_erosion(_ritk(arr), radius=1).to_numpy()
    m = 2
    d = np.abs(sr[m:-m, m:-m, m:-m] - rr[m:-m, m:-m, m:-m])
    assert float(d.max()) < 1e-4, (
        f"GrayscaleErosion interior max diff {d.max():.6f} > 1e-4")


def test_grayscale_dilation_box_interior_agrees_with_sitk():
    """GrayscaleDilation box SE radius=1 interior matches SimpleITK sitkBox.

    Tolerance: interior max absolute diff < 1e-4.
    """
    arr = _make_gradient()
    sr = _np(sitk.GrayscaleDilate(_sitk(arr), [1, 1, 1], sitk.sitkBox))
    rr = ritk.filter.grayscale_dilation(_ritk(arr), radius=1).to_numpy()
    m = 2
    d = np.abs(sr[m:-m, m:-m, m:-m] - rr[m:-m, m:-m, m:-m])
    assert float(d.max()) < 1e-4, (
        f"GrayscaleDilation interior max diff {d.max():.6f} > 1e-4")


def test_laplacian_of_linear_image_is_zero_interior():
    """Laplacian of a linear image is zero in the interior.

    Basis: nabla^2(ax+b) = 0 -- second derivatives vanish for linear functions.
    7-point stencil: sum_d [f(p+e_d)+f(p-e_d)-2f(p)]/h_d^2.

    Tolerances: interior |values| < 1e-3; mutual max diff < 1e-3.
    """
    arr = _make_gradient()
    sr = _np(sitk.Laplacian(_sitk(arr)))
    rr = ritk.filter.laplacian(_ritk(arr)).to_numpy()
    m = 2
    si = sr[m:-m, m:-m, m:-m]
    ri = rr[m:-m, m:-m, m:-m]
    assert float(np.abs(si).max()) < 1e-3, (
        f"SimpleITK Laplacian ramp interior max {np.abs(si).max():.6f}")
    assert float(np.abs(ri).max()) < 1e-3, (
        f"RITK Laplacian ramp interior max {np.abs(ri).max():.6f}")
    assert float(np.abs(si - ri).max()) < 1e-3

""")

DEST.write_text(DEST.read_text(encoding="utf-8") + FILTER_TESTS, encoding="utf-8")
print(f"filter tests appended: {DEST.stat().st_size} bytes")
