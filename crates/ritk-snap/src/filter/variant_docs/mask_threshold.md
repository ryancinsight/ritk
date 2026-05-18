Zero-out voxels below an intensity threshold using a binary self-mask.

Voxels with `in(x) ≤ threshold` are set to 0; others are preserved.
Equivalent to ITK `MaskImageFilter` with a thresholded binary mask derived
from the same image.
