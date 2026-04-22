//! Binary hit-or-miss transform for 3-D images.
//!
//! # Mathematical Specification
//!
//! (M circledast (SE1, SE2))(x) = (M erode SE1)(x) AND (Mc erode SE2)(x)
//! SE1 = cubic box half-width fg_radius (includes origin).
//! SE2 = ring shell outer half-width bg_radius (EXCLUDES origin).
//!
//! # References
//! - Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.
//! - Soille, P. (2003). Morphological Image Analysis, 2nd ed. Springer.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

#[derive(Debug, Clone)]
pub struct HitOrMissTransform {
    pub fg_radius: usize,
    pub bg_radius: usize,
}

impl HitOrMissTransform {
    pub fn new(fg_radius: usize, bg_radius: usize) -> Self {
        Self { fg_radius, bg_radius }
    }
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td.as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("HitOrMissTransform requires f32 data: {:?}", e))?
            .to_vec();
        let result = hit_or_miss_3d(&vals, dims, self.fg_radius, self.bg_radius);
        let device = image.data().device();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(tensor, *image.origin(), *image.spacing(), *image.direction()))
    }
}

fn check_box(data: &[f32], dims: [usize; 3], iz: usize, iy: usize, ix: usize, r: isize) -> bool {
    let [nz, ny, nx] = dims;
    for dz in -r..=r { for dy in -r..=r { for dx in -r..=r {
        let (zz,yy,xx)=(iz as isize+dz, iy as isize+dy, ix as isize+dx);
        if zz<0||zz>=nz as isize||yy<0||yy>=ny as isize||xx<0||xx>=nx as isize { return false; }
        if data[zz as usize*ny*nx+yy as usize*nx+xx as usize]<0.5 { return false; }
    }}}
    true
}

fn erode_binary_box(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut out = vec![0.0_f32; nz*ny*nx];
    for iz in 0..nz { for iy in 0..ny { for ix in 0..nx {
        out[iz*ny*nx+iy*nx+ix] = if check_box(data,dims,iz,iy,ix,r) { 1.0 } else { 0.0 };
    }}}
    out
}

fn check_ring(data: &[f32], dims: [usize; 3], iz: usize, iy: usize, ix: usize, r: isize) -> bool {
    let [nz, ny, nx] = dims;
    for dz in -r..=r { for dy in -r..=r { for dx in -r..=r {
        if dz==0 && dy==0 && dx==0 { continue; }
        let (zz,yy,xx)=(iz as isize+dz, iy as isize+dy, ix as isize+dx);
        if zz<0||zz>=nz as isize||yy<0||yy>=ny as isize||xx<0||xx>=nx as isize { return false; }
        if data[zz as usize*ny*nx+yy as usize*nx+xx as usize]<0.5 { return false; }
    }}}
    true
}

fn erode_binary_ring(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    if radius == 0 { return vec![1.0_f32; nz*ny*nx]; }
    let r = radius as isize;
    let mut out = vec![0.0_f32; nz*ny*nx];
    for iz in 0..nz { for iy in 0..ny { for ix in 0..nx {
        out[iz*ny*nx+iy*nx+ix] = if check_ring(data,dims,iz,iy,ix,r) { 1.0 } else { 0.0 };
    }}}
    out
}

fn hit_or_miss_3d(data: &[f32], dims: [usize; 3], fg_r: usize, bg_r: usize) -> Vec<f32> {
    let n = dims[0]*dims[1]*dims[2];
    let compl: Vec<f32> = data.iter().map(|&v| 1.0 - v.clamp(0.0,1.0)).collect();
    let efg = erode_binary_box(data, dims, fg_r);
    let ebg = erode_binary_ring(&compl, dims, bg_r);
    let mut out = vec![0.0_f32; n];
    for i in 0..n { out[i] = if efg[i]>0.5 && ebg[i]>0.5 { 1.0 } else { 0.0 }; }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    type B = NdArray<f32>;
    fn img(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let t = Tensor::<B,3>::from_data(TensorData::new(vals,Shape::new(dims)),&Default::default());
        Image::new(t,Point::new([0.0,0.0,0.0]),Spacing::new([1.0,1.0,1.0]),Direction::identity())
    }
    fn vv(i: &Image<B,3>) -> Vec<f32> {
        i.data().clone().into_data().as_slice::<f32>().unwrap().to_vec()
    }
    #[test]
    fn test_identity_both_zero() {
        let dims=[6,6,6]; let n=dims[0]*dims[1]*dims[2];
        let v: Vec<f32> = (0..n).map(|i| if i%3==0{1.0}else{0.0}).collect();
        let r = HitOrMissTransform::new(0,0).apply(&img(v.clone(),dims)).unwrap();
        for (i,(&e,&a)) in v.iter().zip(vv(&r).iter()).enumerate() {
            assert!((a-e).abs()<1e-6,"voxel {i}: expected {e}, got {a}");
        }
    }
    #[test]
    fn test_constant_fg_zeroes() {
        let dims=[8,8,8]; let n=dims[0]*dims[1]*dims[2];
        let out = vv(&HitOrMissTransform::new(0,1).apply(&img(vec![1.0;n],dims)).unwrap());
        for (i,&v) in out.iter().enumerate() { assert!(v<0.5,"voxel {i}={v}"); }
    }
    #[test]
    fn test_isolated_voxel_detected() {
        let dims=[9,9,9]; let [nz,ny,nx]=dims; let n=nz*ny*nx;
        let mut v=vec![0.0_f32;n]; let c=4*ny*nx+4*nx+4; v[c]=1.0;
        let out=vv(&HitOrMissTransform::new(0,1).apply(&img(v,dims)).unwrap());
        assert!(out[c]>0.5,"centre must be detected, got {}",out[c]);
        for i in 0..n { if i!=c { assert!(out[i]<0.5,"voxel {i}={}",out[i]); }}
    }
    #[test]
    fn test_metadata_preserved() {
        let dims=[5,5,5]; let n=dims[0]*dims[1]*dims[2];
        let t=Tensor::<B,3>::from_data(TensorData::new(vec![1.0_f32;n],Shape::new(dims)),&Default::default());
        let o=Point::new([1.0,2.0,3.0]); let s=Spacing::new([0.5,0.5,0.5]);
        let r=HitOrMissTransform::new(0,0).apply(&Image::new(t,o,s,Direction::identity())).unwrap();
        assert_eq!(*r.origin(),o); assert_eq!(*r.spacing(),s);
    }
    #[test]
    fn test_anti_extensivity() {
        let dims=[7,7,7]; let n=dims[0]*dims[1]*dims[2];
        let v: Vec<f32> = (0..n).map(|i| if i%2==0{1.0}else{0.0}).collect();
        let out=vv(&HitOrMissTransform::new(1,1).apply(&img(v.clone(),dims)).unwrap());
        for (i,(&orig,&res)) in v.iter().zip(out.iter()).enumerate() {
            assert!(res<=orig+1e-6,"anti-ext at {i}: res={res}>orig={orig}");
        }
    }
}
