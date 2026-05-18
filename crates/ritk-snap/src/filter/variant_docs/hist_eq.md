Global histogram equalization.

Equalizes the intensity histogram across the entire 3-D volume by
mapping each voxel through the global normalised CDF. Equivalent to
ImageJ *Enhance Contrast → Equalize Histogram*.

# Invariant
Output values lie in `[v_min, v_max]` where `v_min/v_max` are the
global minimum and maximum finite voxel values.
