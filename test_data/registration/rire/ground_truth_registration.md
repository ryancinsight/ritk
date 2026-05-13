# RIRE Patient 001 Ground-Truth Registration

**Source**: Retrospective Image Registration Evaluation (RIRE) Project  
**Principal Investigator**: J. Michael Fitzpatrick, Vanderbilt University  
**NIH Project**: 8R01EB002124-03  
**License**: Creative Commons Attribution 3.0 United States  
**Data site**: https://rire.insight-journal.org/

## Image Metadata

### CT (training_001_ct.mha)

| Property | Value |
|---|---|
| Size (x,y,z) | 512 × 512 × 29 |
| Spacing (mm) | (0.653595, 0.653595, 4.0) |
| Origin (mm) | (0.0, 0.0, 0.0) |
| Direction | Identity (1,0,0; 0,1,0; 0,0,1) |
| Pixel type | 16-bit signed integer |
| Array shape | (29, 512, 512) |
| Value range | [-1024, 1969] HU |
| Physical extent (mm) | x∈[0, 333.987], y∈[0, 333.987], z∈[0, 112.000] |

### MR T1 (training_001_mr_T1.mha)

| Property | Value |
|---|---|
| Size (x,y,z) | 256 × 256 × 26 |
| Spacing (mm) | (1.25, 1.25, 4.0) |
| Origin (mm) | (0.0, 0.0, 0.0) |
| Direction | Identity (1,0,0; 0,1,0; 0,0,1) |
| Pixel type | 16-bit signed integer |
| Array shape | (26, 256, 256) |
| Value range | [2, 1626] |
| Physical extent (mm) | x∈[0, 318.750], y∈[0, 318.750], z∈[0, 100.000] |

**Key fact**: Both images have identity direction cosines and origin at (0,0,0). The RIRE coordinate system places the origin at the center of the first voxel, with x along columns, y along rows, and z along slices (right-handed). The images already conform to this convention, so the ground-truth transform applies directly without coordinate-system conversion.

## Ground-Truth Transform (CT → MR T1)

**Method**: Fiducial marker based registration (prospective gold standard)  
**Type**: Rigid body (Euler3D)  
**File**: `training_001_ct_to_mr_T1_ground_truth.tfm`

### Rotation matrix R

```
[  0.997000003   0.077380155  -0.001818059 ]
[ -0.077397855   0.996449628  -0.033131713 ]
[ -0.000752132   0.033173032   0.999449341 ]
```

### Translation vector t (mm)

```
[ 5.03685847, -17.49694636, -27.16499259 ]
```

### Euler3D parameters (SimpleITK/ITK convention)

The SimpleITK Euler3DTransform computes: **T(x) = Rz(aZ) · Rx(aX) · Ry(aY) · x + t**

| Parameter | Radians | Degrees |
|---|---|---|
| angleX | 0.033179119 | 1.9010° |
| angleY | 0.000752547 | 0.0431° |
| angleZ | -0.077500325 | -4.4404° |
| tx (mm) | 5.036858 | — |
| ty (mm) | -17.496946 | — |
| tz (mm) | -27.164993 | — |

**Center of rotation**: (0.0, 0.0, 0.0) — the corner of the CT volume in RIRE coordinates.

### RIRE 8-corner point verification

The RIRE standard `ct_T1.standard` specifies the transform as 8 source→destination point pairs at the CT volume corners. The Euler3D parameters above reproduce all 8 points with **maximum residual 0.000176 mm**, well below the RIRE acceptance threshold of 0.01 mm.

| Point | x (mm) | y (mm) | z (mm) | new_x (mm) | new_y (mm) | new_z (mm) | Residual (mm) |
|---|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 0.0000 | 5.0369 | -17.4970 | -27.1650 | 0.000068 |
| 2 | 333.9870 | 0.0000 | 0.0000 | 338.0219 | -43.3470 | -27.4162 | 0.000176 |
| 3 | 0.0000 | 333.9870 | 0.0000 | 30.8808 | 315.3043 | -16.0856 | 0.000046 |
| 4 | 333.9870 | 333.9870 | 0.0000 | 363.8658 | 289.4544 | -16.3368 | 0.000072 |
| 5 | 0.0000 | 0.0000 | 112.0000 | 4.8333 | -21.2077 | 84.7733 | 0.000072 |
| 6 | 333.9870 | 0.0000 | 112.0000 | 337.8183 | -47.0576 | 84.5221 | 0.000046 |
| 7 | 0.0000 | 333.9870 | 112.0000 | 30.6772 | 311.5937 | 95.8527 | 0.000176 |
| 8 | 333.9870 | 333.9870 | 112.0000 | 363.6622 | 285.7437 | 95.6015 | 0.000068 |

## Coordinate system notes

Per RIRE specification:
- Origin: center of first voxel in the voxel file
- x-axis: along first row (column direction), first→last voxel
- y-axis: along first column (row direction), first→last voxel
- z-axis: perpendicular, right-hand sense (slice direction)
- All coordinates in millimeters
- No scale factors for voxel-size differences; spacing is handled independently

## Acknowledgment

The images and the standard transformation(s) were provided as part of the project,
"Retrospective Image Registration Evaluation",
National Institutes of Health,
Project Number 8R01EB002124-03,
Principal Investigator, J. Michael Fitzpatrick,
Vanderbilt University, Nashville, TN.
