//! White and black top-hat filters for 3-D grayscale images.
//!
//! WTH_B(f)(x) = f(x) - opening_B(f)(x) = f(x) - D_B(E_B(f))(x)
//! BTH_B(f)(x) = closing_B(f)(x) - f(x) = E_B(D_B(f))(x) - f(x)
//!
//! Properties:
//! - WTH and BTH of constant images are 0.
//! - WTH(f) >= 0 (opening is anti-extensive).
//! - BTH(f) >= 0 (closing is extensive).
//!
//! References:
//! - Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.
//! - Soille, P. (2003). Morphological Image Analysis, 2nd ed. Springer.

use super::grayscale_dilation::GrayscaleDilation;
use super::grayscale_erosion::GrayscaleErosion;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// White top-hat filter: WTH_B(f) = f - opening_B(f).
/// Isolates bright structures smaller than the structuring element.
#[derive(Debug, Clone)]
pub struct WhiteTopHatFilter { pub radius: usize }
impl WhiteTopHatFilter {
    pub fn new(radius: usize) -> Self { Self { radius } }
    pub fn with_radius(mut self, radius: usize) -> Self { self.radius = radius; self }
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let eroded = GrayscaleErosion::new(self.radius).apply(image)?;
        let opened = GrayscaleDilation::new(self.radius).apply(&eroded)?;
        sub_clamp(image, &opened)
    }
}

/// Black top-hat filter: BTH_B(f) = closing_B(f) - f.
/// Isolates dark structures smaller than the structuring element.
#[derive(Debug, Clone)]
pub struct BlackTopHatFilter { pub radius: usize }
impl BlackTopHatFilter {
    pub fn new(radius: usize) -> Self { Self { radius } }
    pub fn with_radius(mut self, radius: usize) -> Self { self.radius = radius; self }
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dilated = GrayscaleDilation::new(self.radius).apply(image)?;
        let closed = GrayscaleErosion::new(self.radius).apply(&dilated)?;
        sub_clamp(&closed, image)
    }
}

fn sub_clamp<B: Backend>(a: &Image<B, 3>, b: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
    use burn::tensor::{Shape, Tensor, TensorData};
    let dims = a.shape();
    let ad = a.data().clone().into_data();
    let bd = b.data().clone().into_data();
    let av = ad.as_slice::<f32>().map_err(|e| anyhow::anyhow!("sub_clamp a: {:?}", e))?;
    let bv = bd.as_slice::<f32>().map_err(|e| anyhow::anyhow!("sub_clamp b: {:?}", e))?;
    let result: Vec<f32> = av.iter().zip(bv.iter()).map(|(&ai, &bi)| (ai - bi).max(0.0)).collect();
    let device = a.data().device();
    let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
    Ok(Image::new(t, *a.origin(), *a.spacing(), *a.direction()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    type B = NdArray<f32>;
    fn img(v: Vec<f32>, d: [usize;3]) -> Image<B,3> {
        let t=Tensor::<B,3>::from_data(TensorData::new(v,Shape::new(d)),&Default::default());
        Image::new(t,Point::new([0.0,0.0,0.0]),Spacing::new([1.0,1.0,1.0]),Direction::identity())
    }
    fn vv(i: &Image<B,3>) -> Vec<f32> { i.data().clone().into_data().as_slice::<f32>().unwrap().to_vec() }

    #[test] fn test_wth_constant_zero() {
        let d=[8,8,8]; let n=d[0]*d[1]*d[2];
        let out=vv(&WhiteTopHatFilter::new(1).apply(&img(vec![5.0;n],d)).unwrap());
        for &v in &out { assert!(v.abs()<1e-5,"WTH(const)={v}"); }
    }
    #[test] fn test_wth_bright_spike() {
        let d=[9,9,9]; let [nz,ny,nx]=d; let n=nz*ny*nx;
        let bg=2.0_f32; let mut v=vec![bg;n]; let c=4*ny*nx+4*nx+4; v[c]=10.0;
        let out=vv(&WhiteTopHatFilter::new(1).apply(&img(v,d)).unwrap());
        assert!(out[c]>1.0,"WTH spike not detected: {}",out[c]);
        for &x in &out { assert!(x>=0.0,"WTH non-negative"); }
    }
    #[test] fn test_wth_radius_zero() {
        let d=[6,6,6]; let n=d[0]*d[1]*d[2];
        let v: Vec<f32>=(0..n).map(|i|(i%10)as f32).collect();
        let out=vv(&WhiteTopHatFilter::new(0).apply(&img(v,d)).unwrap());
        for &x in &out { assert!(x.abs()<1e-5,"WTH(r=0)={x}"); }
    }
    #[test] fn test_wth_metadata() {
        let d=[5,5,5]; let n=d[0]*d[1]*d[2];
        let t=Tensor::<B,3>::from_data(TensorData::new(vec![1.0_f32;n],Shape::new(d)),&Default::default());
        let o=Point::new([1.0,2.0,3.0]); let s=Spacing::new([0.5,0.5,0.5]);
        let r=WhiteTopHatFilter::new(1).apply(&Image::new(t,o,s,Direction::identity())).unwrap();
        assert_eq!(*r.origin(),o); assert_eq!(*r.spacing(),s);
    }
    #[test] fn test_wth_non_negative() {
        let d=[8,8,8]; let n=d[0]*d[1]*d[2];
        let v: Vec<f32>=(0..n).map(|i|(i as f32*0.7+1.0)%20.0).collect();
        let out=vv(&WhiteTopHatFilter::new(1).apply(&img(v,d)).unwrap());
        for &x in &out { assert!(x>=-1e-5,"WTH neg: {x}"); }
    }
    #[test] fn test_bth_constant_zero() {
        let d=[8,8,8]; let n=d[0]*d[1]*d[2];
        let out=vv(&BlackTopHatFilter::new(1).apply(&img(vec![5.0;n],d)).unwrap());
        for &v in &out { assert!(v.abs()<1e-5,"BTH(const)={v}"); }
    }
    #[test] fn test_bth_dark_hole() {
        let d=[9,9,9]; let [nz,ny,nx]=d; let n=nz*ny*nx;
        let bg=10.0_f32; let mut v=vec![bg;n]; let c=4*ny*nx+4*nx+4; v[c]=2.0;
        let out=vv(&BlackTopHatFilter::new(1).apply(&img(v,d)).unwrap());
        assert!(out[c]>1.0,"BTH hole not detected: {}",out[c]);
        for &x in &out { assert!(x>=0.0,"BTH non-negative"); }
    }
    #[test] fn test_bth_radius_zero() {
        let d=[6,6,6]; let n=d[0]*d[1]*d[2];
        let v: Vec<f32>=(0..n).map(|i|(i%10)as f32).collect();
        let out=vv(&BlackTopHatFilter::new(0).apply(&img(v,d)).unwrap());
        for &x in &out { assert!(x.abs()<1e-5,"BTH(r=0)={x}"); }
    }
    #[test] fn test_bth_metadata() {
        let d=[5,5,5]; let n=d[0]*d[1]*d[2];
        let t=Tensor::<B,3>::from_data(TensorData::new(vec![1.0_f32;n],Shape::new(d)),&Default::default());
        let o=Point::new([1.0,2.0,3.0]); let s=Spacing::new([0.5,0.5,0.5]);
        let r=BlackTopHatFilter::new(1).apply(&Image::new(t,o,s,Direction::identity())).unwrap();
        assert_eq!(*r.origin(),o); assert_eq!(*r.spacing(),s);
    }
    #[test] fn test_bth_non_negative() {
        let d=[8,8,8]; let n=d[0]*d[1]*d[2];
        let v: Vec<f32>=(0..n).map(|i|(i as f32*0.5+1.0)%15.0).collect();
        let out=vv(&BlackTopHatFilter::new(1).apply(&img(v,d)).unwrap());
        for &x in &out { assert!(x>=-1e-5,"BTH neg: {x}"); }
    }
}
