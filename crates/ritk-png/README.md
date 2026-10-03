# ritk-png

Grayscale PNG input and output for RITK.

The reader preserves 8-bit and 16-bit grayscale samples in the RITK `f32`
image carrier. Other PNG color types are converted to 8-bit luminance. PNG
does not carry the physical-space geometry stored by medical image formats, so
decoded images use unit spacing, zero origin, and identity direction.

`write_png` writes one slice and does not rescale. It accepts finite,
nonnegative integral samples through 65535, choosing 8-bit encoding when all
samples fit in 0–255 and 16-bit encoding otherwise. It rejects volumes,
fractional values, negative values, and values outside the PNG integer range.
