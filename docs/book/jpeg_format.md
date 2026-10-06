# JPEG Format Boundary

JPEG is a visualization and interchange boundary, not a quantitatively exact
archival format. It is appropriate for previews, overlays, reports, and
lightweight exports after metric-sensitive computation has completed.

The native facade infers the format from the path:

~~~rust,ignore
let preview = ritk_io::read_image_native("aligned_preview.jpg")?;
ritk_io::write_image_native("aligned_preview_copy.jpg", &preview)?;
~~~

JPEG quantization changes intensities, so a round trip must use a bounded error
or perceptual check rather than bitwise equality. Keep NIfTI, NRRD, or
MetaImage as the quantitative source for registration and metric evaluation.
After decoding, the result is an ordinary RITK image and can enter the same
filter pipeline as any other input.

## Quality-75 reconstruction oracle

The writer uses the Annex K luminance table and the quality scaling implemented
by the locked `jpeg-encoder` 0.7.1 provider. At quality 75 the scale is
`200 - 2 * 75 = 50`, and each quantizer is
`Q = clamp((K * 50 + 50) / 100, 1, 255)`. The executable oracle checks the
complete emitted table and the coefficients needed by four analytical 8 x 8
blocks: `Q(0,0) = 8`, `Q(0,4) = 12`, `Q(4,0) = 9`, and `Q(4,4) = 34`.

A constant centered level `a` has DCT coefficient `F(0,0) = 8a`. The fourth
orthonormal basis has signs `+--++--+`; horizontal, vertical, and product blocks
therefore have `F(0,4)`, `F(4,0)`, or `F(4,4) = 8a`. Amplitudes 12, 9, and 17
are exact multiples of their quantizers, so forward quantization and inverse
DCT reconstruct every integer sample exactly: the derived error bound is zero.
Tests also require exact agreement with the independent decoder already used by
the crate. A second oracle applies the Annex A forward DCT, emitted quality-75
quantizers, and inverse DCT to the rounded source gradient. Its exact decoded
samples preserve source-sensitive coverage across the gradient's odd-frequency
coefficients. A changed quality table fails the table oracle, while an
inverse-DCT normalization, covered basis, or level-shift mutation fails the
analytical pixels.

The file reader maps the JPEG frame's full encoded range to 0–255 with nearest
integer rounding before grayscale conversion or RGB image construction. This
display policy applies equally to low-precision lossless and 12-bit DCT input.
DICOM decoding uses the clinical path instead: it preserves full integer sample
values and requires the frame precision to match BitsStored.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native JPEG read/write boundary | Available | Uses path inference and toleranced pixel validation. |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Representative visualization workflow where compressed export is acceptable after alignment is complete. |
