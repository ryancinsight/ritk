use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::Image;
type Backend = burn_ndarray::NdArray<f32>;

fn make_mask(vals: Vec<f32>, shape: [usize; 3]) -> Image<Backend, 3> {
    let device = Default::default();
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

#[test]
fn test_fill_holes_solid_sphere_unchanged() {
    let shape = [5usize, 5, 5];
    let n = 125;
    let mut vals = vec![0.0f32; n];
    for iz in 0..5usize {
        for iy in 0..5usize {
            for ix in 0..5usize {
                let d2 = ((iz as i32 - 2).pow(2) + (iy as i32 - 2).pow(2) + (ix as i32 - 2).pow(2))
                    as f32;
                if d2 <= 1.0 {
                    vals[iz * 25 + iy * 5 + ix] = 1.0;
                }
            }
        }
    }
    let mask = make_mask(vals.clone(), shape);
    let result = BinaryFillHoles.apply(&mask);
    result.with_data_slice(|out_vals| {
        assert_eq!(out_vals, vals.as_slice(), "solid sphere must be unchanged");
    });
}

#[test]
fn test_fill_holes_hollow_sphere_fills_interior() {
    let shape = [7usize, 7, 7];
    let n = 343;
    let mut vals = vec![0.0f32; n];
    for iz in 0..7usize {
        for iy in 0..7usize {
            for ix in 0..7usize {
                let d2 = ((iz as i32 - 3).pow(2) + (iy as i32 - 3).pow(2) + (ix as i32 - 3).pow(2))
                    as f32;
                if (4.0..=9.0).contains(&d2) {
                    vals[iz * 49 + iy * 7 + ix] = 1.0;
                }
            }
        }
    }
    let mask = make_mask(vals.clone(), shape);
    let result = BinaryFillHoles.apply(&mask);
    result.with_data_slice(|out_vals| {
        for iz in 0..7usize {
            for iy in 0..7usize {
                for ix in 0..7usize {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 < 4.0 {
                        assert_eq!(
                            out_vals[iz * 49 + iy * 7 + ix],
                            1.0,
                            "interior hole at ({},{},{}) must be filled",
                            iz,
                            iy,
                            ix
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn test_fill_holes_all_zero_unchanged() {
    let shape = [3usize, 3, 3];
    let vals = vec![0.0f32; 27];
    let mask = make_mask(vals.clone(), shape);
    let result = BinaryFillHoles.apply(&mask);
    result.with_data_slice(|out_vals| {
        assert_eq!(out_vals, vals.as_slice(), "all-zero mask must be unchanged");
    });
}

#[test]
fn test_fill_holes_all_one_unchanged() {
    let shape = [3usize, 3, 3];
    let vals = vec![1.0f32; 27];
    let mask = make_mask(vals.clone(), shape);
    let result = BinaryFillHoles.apply(&mask);
    result.with_data_slice(|out_vals| {
        assert_eq!(out_vals, vals.as_slice(), "all-one mask must be unchanged");
    });
}

#[test]
fn test_fill_holes_output_shape_preserved() {
    let shape = [4usize, 5, 6];
    let mask = make_mask(vec![0.0f32; 120], shape);
    let result = BinaryFillHoles.apply(&mask);
    assert_eq!(result.shape(), shape);
}
