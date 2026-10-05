# ritk-nifti

`ritk-nifti` reads and writes single-file NIfTI-1 and NIfTI-2 images. It owns
the format boundary used by `ritk-io`; Analyze `.hdr`/`.img` pairs belong to
`ritk-analyze`.

The legacy image API reads `f32` volumes and acquisition series and provides
label-map helpers. The stored-volume API reads rank-three NIfTI-1/2 `.nii` and
`.nii.gz` files without changing any of RITK's ten fixed-width scalar sample
types or their bits. It carries Cartesian geometry in LPS millimeters and the
NIfTI linear intensity calibration. The typed writer emits NIfTI-2, retaining
the `f64` affine and calibration fields; `.nii.gz` selects gzip output.

The stored API rejects unsupported ranks and NIfTI extension records. The
writer rejects non-Cartesian coordinate maps, modality lookup calibration,
varying per-frame calibration, and zero-slope calibration before creating or
truncating the destination. NIfTI's scalar datatype codes follow the
[NIfTI-1 format specification](https://nifti.nimh.nih.gov/dfwg/presentations/nifti1_cox.pdf/download).
Its `scl_slope` and `scl_inter` behavior follows the official
[data-scaling description](https://nifti.nimh.nih.gov/dfwg/presentations/nifti-1-rationale.html).

```no_run
use ritk_image_io::ImageReadBudget;
use ritk_nifti::{read_nifti_stored, write_nifti_stored};

fn main() -> anyhow::Result<()> {
    let volume = read_nifti_stored("scan.nii.gz", ImageReadBudget::DEFAULT)?;
    write_nifti_stored("copy.nii.gz", &volume)?;
    Ok(())
}
```

See the [NIfTI format chapter](../../docs/book/nifti_format.md) for the
spatial, acquisition, calibration, and failure contracts.
