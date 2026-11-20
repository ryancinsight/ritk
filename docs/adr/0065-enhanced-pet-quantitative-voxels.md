# ADR 0065: Quantify Enhanced PET stored voxels

- Status: Proposed
- Class: [major]

Apply the Shared or Per-Frame Real World Value Mapping to stored integer
voxels before display rescale. A declared SUVbw mapping supplies g/mL
directly. An activity-concentration mapping uses patient mass and the activity
of the referenced isotope at the frame's time reference. Decay-corrected
activity uses the correction datetime. Otherwise quantification requires
the frame reference datetime. Activity decays as 2^(−elapsed/half-life).
Agent references select a single isotope; ambiguous mixtures lack a scalar
denominator. The native viewer shares the resulting buffer across metadata
clones and uses calibrated samples at each selected voxel.

Linear and lookup-table mappings respect their input range. Padding and
unmapped samples have no SUV. Diagnostic values retain the ordinary pixel
transformation. The same geometry resampler aligns both volumes. The
storage budget accounts for both buffers. Enhanced PET declares 16-bit
stored samples, which remain exact before the double-precision mapping.

The Part 10 regression distinguishes both quantitative routes from
diagnostic rescale, checks padding and range rejection, and crosses midnight
by one known half-life. The native viewer test selection runs on Metal.
