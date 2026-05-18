Intensity clamp filter (ITK `ClampImageFilter` parity).

output(x) = clamp(I(x), lower, upper). All voxels outside [lower, upper]
are hard-clamped; interior voxels are unchanged.
