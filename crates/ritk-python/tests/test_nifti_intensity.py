"""Compare all calibrated NIfTI samples against an independent reader."""
import numpy as np
import pytest
import ritk
import SimpleITK as sitk

from _fixture_cache import catalog_fixture


@pytest.mark.parametrize("key", [
    "openneuro/sub-01_T1w.nii.gz",
    "registration/ixi/openneuro_ds000208_sub01_T1w.nii.gz",
    "registration/ixi/openneuro_ds000208_sub02_T1w.nii.gz",
    "registration/ixi/openneuro_ds000208_sub03_T1w.nii.gz",
])
def test_catalog_volume_matches_simpleitk_intensities(key):
    path = str(catalog_fixture(key))
    actual = ritk.io.read_image(path).to_numpy()
    expected = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))
    np.testing.assert_array_equal(actual, expected)
