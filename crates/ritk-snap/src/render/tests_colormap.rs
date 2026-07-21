use super::*;
use iris::color::{ColorMap, Normalized};

fn rgb8(map: NamedColorMap, value: f32) -> [u8; 3] {
    let value = Normalized::new(value).expect("test value is normalized");
    let [red, green, blue, _] = map.sample(value).to_rgba8();
    [red, green, blue]
}

#[test]
fn iris_boundary_vectors_preserve_snap_display_contract() {
    assert_eq!(rgb8(NamedColorMap::Grayscale, 0.0), [0, 0, 0]);
    assert_eq!(rgb8(NamedColorMap::Grayscale, 1.0), [255; 3]);
    assert_eq!(rgb8(NamedColorMap::Inverted, 0.0), [255; 3]);
    assert_eq!(rgb8(NamedColorMap::Inverted, 1.0), [0; 3]);
    assert_eq!(rgb8(NamedColorMap::Hot, 0.0), [0; 3]);
    assert_eq!(rgb8(NamedColorMap::Hot, 1.0), [255; 3]);
    assert_eq!(rgb8(NamedColorMap::Cool, 0.0), [0, 255, 255]);
    assert_eq!(rgb8(NamedColorMap::Cool, 1.0), [255, 0, 255]);
    assert_eq!(rgb8(NamedColorMap::Jet, 0.0), [0, 0, 128]);
    assert_eq!(rgb8(NamedColorMap::Plasma, 1.0), [240, 249, 33]);
}

#[test]
fn iris_grayscale_remains_channel_equal_and_monotone() {
    let mut previous = 0;
    for step in 0_u16..=255 {
        let value = f32::from(step) / 255.0;
        let [red, green, blue] = rgb8(NamedColorMap::Grayscale, value);
        assert_eq!([red, green, blue], [red; 3]);
        assert!(red >= previous, "grayscale decreased at {value}");
        previous = red;
    }
}

#[test]
fn iris_runtime_selection_is_complete_and_labeled() {
    assert_eq!(NamedColorMap::ALL.len(), 10);
    for (index, map) in NamedColorMap::ALL.iter().copied().enumerate() {
        assert!(!map.label().is_empty());
        assert!(
            NamedColorMap::ALL[index + 1..]
                .iter()
                .all(|candidate| *candidate != map),
            "duplicate map at index {index}"
        );
    }
}

#[test]
fn iris_rejects_invalid_normalized_input_before_rendering() {
    assert!(Normalized::new(-0.01).is_err());
    assert!(Normalized::new(1.01).is_err());
    assert!(Normalized::new(f32::NAN).is_err());
}
