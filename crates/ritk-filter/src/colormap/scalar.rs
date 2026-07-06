use anyhow::{bail, Result};
use ritk_image::tensor::Backend;
use ritk_image::{ColorVolume, Image};
use ritk_tensor_ops::extract_vec;

/// Linear-LUT colormaps (those expressible without a piecewise table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    /// Greyscale — `(s, s, s)`. ITK default.
    Grey,
    /// Red ramp — `(s, 0, 0)`.
    Red,
    /// Green ramp — `(0, s, 0)`.
    Green,
    /// Blue ramp — `(0, 0, s)`.
    Blue,
}

impl Colormap {
    /// Parse a case-insensitive colormap name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "grey" | "gray" => Ok(Self::Grey),
            "red" => Ok(Self::Red),
            "green" => Ok(Self::Green),
            "blue" => Ok(Self::Blue),
            other => bail!(
                "ScalarToRGBColormap: colormap '{other}' is not a linear LUT; \
                 only grey/red/green/blue are supported"
            ),
        }
    }

    #[inline]
    fn rgb(self, s: f32) -> [f32; 3] {
        match self {
            Self::Grey => [s, s, s],
            Self::Red => [s, 0.0, 0.0],
            Self::Green => [0.0, s, 0.0],
            Self::Blue => [0.0, 0.0, s],
        }
    }
}

/// Scalar-to-RGB colormap filter.
#[derive(Debug, Clone, Copy)]
pub struct ScalarToRGBColormapFilter {
    colormap: Colormap,
}

impl ScalarToRGBColormapFilter {
    /// Construct with the given colormap.
    pub fn new(colormap: Colormap) -> Self {
        Self { colormap }
    }

    /// Apply the colormap, returning a 3-component RGB image (channel values in
    /// `[0, 255]`).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<ColorVolume<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let (min, max) = vals
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let range = max - min;

        let n = vals.len();
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for (i, &v) in vals.iter().enumerate() {
            let t = if range > 0.0 {
                ((v - min) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let s = (t * 255.0).floor();
            let [cr, cg, cb] = self.colormap.rgb(s);
            r[i] = cr;
            g[i] = cg;
            b[i] = cb;
        }

        ColorVolume::<B, 3>::from_component_buffers(
            &[r, g, b],
            dims,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            &image.data().device(),
        )
    }
}
