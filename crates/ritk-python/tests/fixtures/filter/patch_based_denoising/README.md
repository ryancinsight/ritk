# Patch-based denoising oracle

`ra_float_expected.npy` is the `float32` output of SimpleITK 2.5.5
`PatchBasedDenoisingImageFilter` for `RA-Float.nrrd` with NumPy `default_rng(0)`
Gaussian noise scaled by 5.0 and these filter settings:

- kernel-bandwidth estimation disabled;
- one iteration;
- 200 sample patches;
- patch radius 2;
- one work unit.

SHA-512:
`5f56b453f539d143088e44539816f3ced61aea4f29a300162f7e922b611c94ee7e0da997555dbb9ee5d2299d6554cdc1d5f50f374d58ad0c97731129f260ef95`.

The source input remains independently content-addressed by
`tests/sitk_input_manifest.json`. The test reconstructs the noisy input and
compares every output voxel within one `float32` ULP; the fixture removes only
the repeated reference computation from the timed test.
