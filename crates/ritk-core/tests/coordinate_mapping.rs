use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use proptest::prelude::*;

type Backend = NdArray<f32>;
const D: usize = 3;

fn make_rotation(angle_x: f64, angle_y: f64, angle_z: f64) -> Direction<D> {
    let cx = angle_x.cos(); let sx = angle_x.sin();
    let cy = angle_y.cos(); let sy = angle_y.sin();
    let cz = angle_z.cos(); let sz = angle_z.sin();

    // Rx * Ry * Rz
    let mut rot = Direction::<D>::identity();
    let m = rot.inner_mut();
    
    // Rz
    let rz = nalgebra::SMatrix::<f64, 3, 3>::new(
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    );
    
    // Ry
    let ry = nalgebra::SMatrix::<f64, 3, 3>::new(
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    );
    
    // Rx
    let rx = nalgebra::SMatrix::<f64, 3, 3>::new(
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    );
    
    *m = rx * ry * rz;
    rot
}

proptest! {
    #[test]
    fn test_coordinate_roundtrip(
        ox in -100.0f64..100.0, oy in -100.0f64..100.0, oz in -100.0f64..100.0,
        sx in 0.1f64..5.0, sy in 0.1f64..5.0, sz in 0.1f64..5.0,
        ax in -3.14f64..3.14, ay in -3.14f64..3.14, az in -3.14f64..3.14,
        px in -50.0f64..50.0, py in -50.0f64..50.0, pz in -50.0f64..50.0
    ) {
        let device = Default::default();
        // Use minimal data tensor as we don't access it
        let data = Tensor::<Backend, D>::zeros([2, 2, 2], &device);
        
        let origin = Point::<D>::new([ox, oy, oz]);
        let spacing = Spacing::<D>::new([sx, sy, sz]);
        let direction = make_rotation(ax, ay, az);
        
        let image = Image::new(data, origin, spacing, direction);
        let point = Point::<D>::new([px, py, pz]);
        
        let index = image.transform_physical_point_to_continuous_index(&point);
        let recovered = image.transform_continuous_index_to_physical_point(&index);
        
        prop_assert!((point[0] - recovered[0]).abs() < 1e-4, "X mismatch: {} vs {}", point[0], recovered[0]);
        prop_assert!((point[1] - recovered[1]).abs() < 1e-4, "Y mismatch: {} vs {}", point[1], recovered[1]);
        prop_assert!((point[2] - recovered[2]).abs() < 1e-4, "Z mismatch: {} vs {}", point[2], recovered[2]);
    }

    #[test]
    fn test_tensor_batch_consistency(
        ox in -10.0f64..10.0,
        sx in 0.5f64..2.0,
        px in -10.0f64..10.0
    ) {
        // Simplified test for tensor ops (single dimension effectively for simplicity in proptest setup)
        // We manually construct 3D inputs from these scalars
        let device = Default::default();
        let data = Tensor::<Backend, D>::zeros([2, 2, 2], &device);
        
        let origin = Point::<D>::new([ox, ox, ox]);
        let spacing = Spacing::<D>::new([sx, sx, sx]);
        let direction = Direction::<D>::identity();
        
        let image = Image::new(data, origin, spacing, direction);
        
        let point_val = Point::<D>::new([px, px, px]);
        let index_val = image.transform_physical_point_to_continuous_index(&point_val);
        
        // Tensor op
        let points_tensor = Tensor::<Backend, 2>::from_floats([[px as f32, px as f32, px as f32]], &device);
        let indices_tensor = image.world_to_index_tensor(points_tensor);
        let indices_data = indices_tensor.into_data();
        let indices_slice = indices_data.as_slice::<f32>().unwrap();
        
        prop_assert!((indices_slice[0] - index_val[0] as f32).abs() < 1e-4);
        prop_assert!((indices_slice[1] - index_val[1] as f32).abs() < 1e-4);
        prop_assert!((indices_slice[2] - index_val[2] as f32).abs() < 1e-4);
    }
}
