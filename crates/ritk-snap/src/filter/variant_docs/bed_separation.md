CT bed/table separation mask filter.

Applies intensity thresholding, optional largest-component selection,
and morphological closing/opening to isolate the patient body.
Voxels outside the retained mask are replaced by `config.outside_value`.
