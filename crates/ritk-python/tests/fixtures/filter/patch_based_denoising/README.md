# Patch-based denoising oracle

`ra_float_expected.npy` is the Linux x86-64 `float32` output of SimpleITK 2.5.5
`PatchBasedDenoisingImageFilter` for `RA-Float.nrrd` with NumPy `default_rng(0)`
Gaussian noise scaled by 5.0 and these filter settings:

- kernel-bandwidth estimation disabled;
- one iteration;
- 200 sample patches;
- patch radius 2;
- one work unit.

SHA-512:
`01db7807faef849df34f4be941d1662c56984c788a732a60b71d38eb5e4087eb3db1ffad967df1a861baa2f0206c8244453c89260b91d8aa73c68f8fd51bf190`.

The source input remains independently content-addressed by
`tests/sitk_input_manifest.json`. The test reconstructs the noisy input and
compares every output voxel within one `float32` ULP; the fixture removes only
the repeated reference computation from the timed test.
