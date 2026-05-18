Curved Planar Reformation (CPR).

Generates a 2-D "straightened" view along a curved path through a 3-D volume.
The path is interpolated from control points using a Catmull-Rom spline with
arc-length parameterisation. Cross-sectional planes perpendicular to the path
tangent are sampled via trilinear interpolation.

**Output**: 2-D image (rows = cross-section offset, columns = path position).

**References**: Kanitsar et al. (2002), "CPR — Curved Planar Reformation."
