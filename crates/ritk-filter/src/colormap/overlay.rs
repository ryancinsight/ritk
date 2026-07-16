use anyhow::{bail, Result};
use ritk_image::native::{ColorVolume, Image};
use ritk_tensor_ops::native::{extract_image_vec, rebuild_image};
use std::collections::BTreeMap;

/// ITK `LabelToRGBImageFilter` default 30-colour table (labels `1..=30`; label
/// `k ≥ 1` maps to `LABEL_COLORS[(k − 1) mod 30]`). Extracted from
/// `sitk.LabelToRGB`.
const LABEL_COLORS: [[f32; 3]; 30] = [
    [0.0, 205.0, 0.0],
    [0.0, 0.0, 255.0],
    [0.0, 255.0, 255.0],
    [255.0, 0.0, 255.0],
    [255.0, 127.0, 0.0],
    [0.0, 100.0, 0.0],
    [138.0, 43.0, 226.0],
    [139.0, 35.0, 35.0],
    [0.0, 0.0, 128.0],
    [139.0, 139.0, 0.0],
    [255.0, 62.0, 150.0],
    [139.0, 76.0, 57.0],
    [0.0, 134.0, 139.0],
    [205.0, 104.0, 57.0],
    [191.0, 62.0, 255.0],
    [0.0, 139.0, 69.0],
    [199.0, 21.0, 133.0],
    [205.0, 55.0, 0.0],
    [32.0, 178.0, 170.0],
    [106.0, 90.0, 205.0],
    [255.0, 20.0, 147.0],
    [69.0, 139.0, 116.0],
    [72.0, 118.0, 255.0],
    [205.0, 79.0, 57.0],
    [0.0, 0.0, 205.0],
    [139.0, 34.0, 82.0],
    [139.0, 0.0, 139.0],
    [238.0, 130.0, 238.0],
    [139.0, 0.0, 0.0],
    [255.0, 0.0, 0.0],
];

/// Map a label image to RGB using ITK's default label-colour table
/// (`itk::LabelToRGBImageFilter` / `sitk.LabelToRGB`).
///
/// Background voxels (those equal to `background`, default `0`) map to black;
/// every other label `k` maps to `LABEL_COLORS[(k − 1) mod 30]`, cycling through
/// the 30-colour table.
#[derive(Debug, Clone, Copy)]
pub struct LabelToRGBFilter {
    background: i64,
}

impl LabelToRGBFilter {
    /// Construct with the given background label (default ITK value `0`).
    pub fn new(background: i64) -> Self {
        Self { background }
    }

    /// Apply the label-to-RGB mapping, returning a 3-component RGB image.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ColorVolume<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = extract_image_vec(image)?;
        let n = vals.len();
        let (mut r, mut g, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        for (i, &v) in vals.iter().enumerate() {
            let lbl = v.round() as i64;
            if lbl == self.background {
                continue; // black
            }
            let idx = (lbl - 1).rem_euclid(LABEL_COLORS.len() as i64) as usize;
            let [cr, cg, cb] = LABEL_COLORS[idx];
            r[i] = cr;
            g[i] = cg;
            b[i] = cb;
        }
        let mut interleaved = vec![0.0f32; n * 3];
        for i in 0..n {
            interleaved[3 * i] = r[i];
            interleaved[3 * i + 1] = g[i];
            interleaved[3 * i + 2] = b[i];
        }
        ColorVolume::<f32, B, 3>::from_flat_on(
            interleaved,
            dims,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

/// Overlay a label image on a grayscale image as RGB
/// (`itk::LabelOverlayImageFilter` / `sitk.LabelOverlay`).
///
/// Background voxels pass the grayscale value through on all three channels;
/// each labelled voxel `k` is alpha-blended with its colour from the 30-entry
/// `LABEL_COLORS` table:
///
/// ```text
/// out = floor((1 − opacity)·gray + opacity·LABEL_COLORS[(k−1) mod 30])
/// ```
///
/// The blend is truncated (C++ uint8 cast), verified against `sitk.LabelOverlay`
/// (`gray = 200`, label 2, `opacity = 0.5` → blue channel `0.5·200 + 0.5·255 =
/// 227.5 → 227`). The grayscale input is assumed already in `[0, 255]`.
#[derive(Debug, Clone, Copy)]
pub struct LabelOverlayFilter {
    opacity: f64,
    background: i64,
}

impl LabelOverlayFilter {
    /// Construct with the given `opacity` (`[0, 1]`, ITK default `0.5`) and
    /// background label (ITK default `0`).
    pub fn new(opacity: f64, background: i64) -> Self {
        Self {
            opacity,
            background,
        }
    }

    /// Overlay `label` on `image`, returning a 3-component RGB image.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        label: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ColorVolume<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (gray, dims) = extract_image_vec(image)?;
        let (lab, ldims) = extract_image_vec(label)?;
        if dims != ldims {
            bail!("LabelOverlay: image {dims:?} and label {ldims:?} shapes differ");
        }
        let o = self.opacity;
        let n = gray.len();
        let (mut r, mut g, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        for i in 0..n {
            let gv = gray[i] as f64;
            let lbl = lab[i].round() as i64;
            if lbl == self.background {
                let v = gray[i];
                r[i] = v;
                g[i] = v;
                b[i] = v;
            } else {
                let idx = (lbl - 1).rem_euclid(LABEL_COLORS.len() as i64) as usize;
                let c = LABEL_COLORS[idx];
                r[i] = ((1.0 - o) * gv + o * c[0] as f64).floor() as f32;
                g[i] = ((1.0 - o) * gv + o * c[1] as f64).floor() as f32;
                b[i] = ((1.0 - o) * gv + o * c[2] as f64).floor() as f32;
            }
        }
        let mut interleaved = vec![0.0f32; n * 3];
        for i in 0..n {
            interleaved[3 * i] = r[i];
            interleaved[3 * i + 1] = g[i];
            interleaved[3 * i + 2] = b[i];
        }
        ColorVolume::<f32, B, 3>::from_flat_on(
            interleaved,
            dims,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

/// Offsets of an ITK `FlatStructuringElement::Ball(radius)`: a neighbour offset
/// `d` is in the element iff `Σ (d_a)² ≤ (r_a + 0.5)²` per axis combined as an
/// ellipsoid `Σ (d_a / (r_a + 0.5))² ≤ 1`.  Axes of size 1 (degenerate, e.g. a
/// 2-D `z = 1` volume) contribute no off-plane offset.
fn ball_offsets(radius: [usize; 3], dims: [usize; 3]) -> Vec<[isize; 3]> {
    let mut offs = Vec::new();
    let r = radius.map(|x| x as isize);
    let denom: [f64; 3] = std::array::from_fn(|a| {
        let rr = radius[a] as f64 + 0.5;
        rr * rr
    });
    for dz in -r[0]..=r[0] {
        if dims[0] == 1 && dz != 0 {
            continue;
        }
        for dy in -r[1]..=r[1] {
            if dims[1] == 1 && dy != 0 {
                continue;
            }
            for dx in -r[2]..=r[2] {
                if dims[2] == 1 && dx != 0 {
                    continue;
                }
                let s = (dz * dz) as f64 / denom[0]
                    + (dy * dy) as f64 / denom[1]
                    + (dx * dx) as f64 / denom[2];
                if s <= 1.0 {
                    offs.push([dz, dy, dx]);
                }
            }
        }
    }
    offs
}

/// Overlay the **contours** of a label image on a grayscale image as RGB
/// (`itk::LabelMapContourOverlayImageFilter` / `sitk.LabelMapContourOverlay`).
///
/// Each label region's contour band is `dilate(Ball(dilation_radius)) −
/// erode(Ball(contour_thickness))` (binary dilation with a background border,
/// binary erosion with ITK's default foreground border).  Contours are painted
/// in ascending-label order so the higher label wins on overlap
/// (`HIGH_LABEL_ON_TOP`), then alpha-blended onto the feature image with the same
/// functor as [`LabelOverlayFilter`].
///
/// Geometric parameters are in ritk axis order `[z, y, x]`.  SimpleITK's default
/// (`dilation_radius = contour_thickness = 1`, `CONTOUR`, `HIGH_LABEL_ON_TOP`) is
/// reproduced bit-for-bit in both 2-D (`z = 1`) and 3-D.
#[derive(Debug, Clone, Copy)]
pub struct LabelMapContourOverlayFilter {
    opacity: f64,
    background: i64,
    dilation_radius: [usize; 3],
    contour_thickness: [usize; 3],
}

impl LabelMapContourOverlayFilter {
    /// Construct with SimpleITK's default geometry (`dilation_radius =
    /// contour_thickness = [1, 1, 1]`).
    pub fn new(opacity: f64, background: i64) -> Self {
        Self {
            opacity,
            background,
            dilation_radius: [1, 1, 1],
            contour_thickness: [1, 1, 1],
        }
    }

    /// Override the per-axis dilation radius and contour thickness (ritk `[z, y, x]`).
    pub fn with_geometry(
        mut self,
        dilation_radius: [usize; 3],
        contour_thickness: [usize; 3],
    ) -> Self {
        self.dilation_radius = dilation_radius;
        self.contour_thickness = contour_thickness;
        self
    }

    /// Overlay label contours on `feature`, returning a 3-component RGB image.
    pub fn apply<B>(
        &self,
        feature: &Image<f32, B, 3>,
        label: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ColorVolume<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (lab, dims) = extract_image_vec(label)?;
        let n = lab.len();
        let [_, dy_n, dx_n] = dims;
        let strides = [dy_n * dx_n, dx_n, 1usize];

        // Collect each non-background label's voxel coordinates.
        let mut by_label: BTreeMap<i64, Vec<[usize; 3]>> = BTreeMap::new();
        for (i, &v) in lab.iter().enumerate() {
            let l = v.round() as i64;
            if l == self.background {
                continue;
            }
            let z = i / strides[0];
            let rem = i % strides[0];
            let y = rem / strides[1];
            let x = rem % strides[1];
            by_label.entry(l).or_default().push([z, y, x]);
        }

        let dil = ball_offsets(self.dilation_radius, dims);
        let thick = ball_offsets(self.contour_thickness, dims);
        // Window margin per axis: dilation reach + thickness reach.
        let margin = [
            self.dilation_radius[0] + self.contour_thickness[0],
            self.dilation_radius[1] + self.contour_thickness[1],
            self.dilation_radius[2] + self.contour_thickness[2],
        ];

        // Contour-label image; ascending label order → higher label wins.
        let mut contour_labels = vec![0.0_f32; n];
        for (&lbl, coords) in &by_label {
            contour_band(
                coords,
                dims,
                strides,
                &dil,
                &thick,
                margin,
                lbl,
                &mut contour_labels,
            );
        }

        let cl_image = rebuild_image(contour_labels, dims, label, backend)?;
        LabelOverlayFilter::new(self.opacity, self.background).apply(feature, &cl_image, backend)
    }
}

/// Write the contour band of one label into `out` (flat `[z,y,x]` buffer).
///
/// Computes `dilate(dil) − erode(thick)` over a clamped window around the label's
/// bounding box: dilation treats out-of-image as background, erosion treats it as
/// foreground (ITK `BinaryErode` default).  Sets `out[idx] = label` on the band.
#[allow(clippy::too_many_arguments)]
fn contour_band(
    coords: &[[usize; 3]],
    dims: [usize; 3],
    strides: [usize; 3],
    dil: &[[isize; 3]],
    thick: &[[isize; 3]],
    margin: [usize; 3],
    label: i64,
    out: &mut [f32],
) {
    // Bounding box.
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for c in coords {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    // Clamped window.
    let wlo: [usize; 3] = std::array::from_fn(|a| lo[a].saturating_sub(margin[a]));
    let whi: [usize; 3] = std::array::from_fn(|a| (hi[a] + margin[a]).min(dims[a] - 1));
    let wext: [usize; 3] = std::array::from_fn(|a| whi[a] - wlo[a] + 1);
    let wn = wext[0] * wext[1] * wext[2];
    let wstr = [wext[1] * wext[2], wext[2], 1usize];

    let in_window = |z: isize, y: isize, x: isize| -> bool {
        z >= wlo[0] as isize
            && z <= whi[0] as isize
            && y >= wlo[1] as isize
            && y <= whi[1] as isize
            && x >= wlo[2] as isize
            && x <= whi[2] as isize
    };
    let in_image = |z: isize, y: isize, x: isize| -> bool {
        z >= 0
            && z < dims[0] as isize
            && y >= 0
            && y < dims[1] as isize
            && x >= 0
            && x < dims[2] as isize
    };
    let wlocal = |z: usize, y: usize, x: usize| -> usize {
        (z - wlo[0]) * wstr[0] + (y - wlo[1]) * wstr[1] + (x - wlo[2]) * wstr[2]
    };

    // mask grid (label voxels within the window).
    let mut mask = vec![false; wn];
    for c in coords {
        mask[wlocal(c[0], c[1], c[2])] = true;
    }

    // dilation: OR over `dil` offsets; out-of-window neighbour is background.
    let mut dilate = vec![false; wn];
    for z in wlo[0]..=whi[0] {
        for y in wlo[1]..=whi[1] {
            for x in wlo[2]..=whi[2] {
                let mut v = false;
                for o in dil {
                    let (nz, ny, nx) = (z as isize + o[0], y as isize + o[1], x as isize + o[2]);
                    if in_window(nz, ny, nx) && mask[wlocal(nz as usize, ny as usize, nx as usize)]
                    {
                        v = true;
                        break;
                    }
                }
                dilate[wlocal(z, y, x)] = v;
            }
        }
    }

    // erosion of `dilate` then contour = dilate & !erode.
    for z in wlo[0]..=whi[0] {
        for y in wlo[1]..=whi[1] {
            for x in wlo[2]..=whi[2] {
                let li = wlocal(z, y, x);
                if !dilate[li] {
                    continue;
                }
                let mut eroded = true;
                for o in thick {
                    let (nz, ny, nx) = (z as isize + o[0], y as isize + o[1], x as isize + o[2]);
                    let neigh = if in_window(nz, ny, nx) {
                        dilate[wlocal(nz as usize, ny as usize, nx as usize)]
                    } else if in_image(nz, ny, nx) {
                        false // in image, out of window → background of dilate
                    } else {
                        true // out of image → ITK foreground border
                    };
                    if !neigh {
                        eroded = false;
                        break;
                    }
                }
                if !eroded {
                    let gi = z * strides[0] + y * strides[1] + x * strides[2];
                    out[gi] = label as f32;
                }
            }
        }
    }
}
