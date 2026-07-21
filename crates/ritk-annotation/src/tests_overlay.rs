use super::*;

// Test 1a – correct length ok
#[test]
fn test_image_overlay_new_correct_length() {
    let o = ImageOverlay::new("layer", vec![0.0f32; 24], [2, 3, 4]);
    assert_eq!(o.data.len(), 24);
    assert_eq!(o.dims.0, [2, 3, 4]);
    assert_eq!(o.name, "layer");
    assert_eq!(o.visible, Visibility::Visible);
    assert_eq!(o.opacity.get(), 1.0);
    assert_eq!(o.colormap, Colormap::Grayscale);
}

// Test 1b – mismatch panics
#[test]
#[should_panic(expected = "does not match dims")]
fn test_image_overlay_new_panics_on_dims_mismatch() {
    ImageOverlay::new("bad", vec![0.0f32; 5], [2, 2, 2]);
}

// Test 2a – valid contour
#[test]
fn test_contour_overlay_add_contour_valid() {
    let mut c = ContourOverlay::new("c", 1);
    let pts = vec![[0.0f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
    assert!(c.add_contour(pts.clone()).is_ok());
    assert_eq!(c.contours.len(), 1);
    assert_eq!(c.contours[0], pts);
}

// Test 2b – single point returns Err
#[test]
fn test_contour_overlay_add_contour_single_point_returns_err() {
    let mut c = ContourOverlay::new("c", 1);
    let r = c.add_contour(vec![[0.0f64, 0.0, 0.0]]);
    assert!(r.is_err());
    let msg = r.unwrap_err();
    assert!(msg.contains("got 1"), "msg: {}", msg);
}

// Test 3 – label_count {0,1,2,2,3} -> 3
#[test]
fn test_mask_overlay_label_count() {
    let o = MaskOverlay::new("m", vec![0u32, 1, 2, 2, 3], [1, 1, 5]);
    assert_eq!(o.label_count(), 3);
}

// Test 4 – add_image_overlay len == 1
#[test]
fn test_overlay_state_add_image_overlay_len() {
    let mut s = OverlayState::new();
    s.add_image_overlay(ImageOverlay::new("img", vec![0.0f32; 8], [2, 2, 2]));
    assert_eq!(s.image_overlays.len(), 1);
}

// Test 5 – remove returns true/false
#[test]
fn test_overlay_state_remove_image_overlay() {
    let mut s = OverlayState::new();
    s.add_image_overlay(ImageOverlay::new("alpha", vec![0.0f32; 1], [1, 1, 1]));
    s.add_image_overlay(ImageOverlay::new("beta", vec![0.0f32; 1], [1, 1, 1]));
    assert!(s.remove_image_overlay("alpha"));
    assert_eq!(s.image_overlays.len(), 1);
    assert_eq!(s.image_overlays[0].name, "beta");
    assert!(!s.remove_image_overlay("nonexistent"));
    assert_eq!(s.image_overlays.len(), 1);
}

// Test 6 – visible_image_overlays filters hidden
#[test]
fn test_overlay_state_visible_image_overlays() {
    let mut s = OverlayState::new();
    s.add_image_overlay(
        ImageOverlay::new("vis", vec![1.0f32; 4], [1, 2, 2]).with_visible(Visibility::Visible),
    );
    s.add_image_overlay(
        ImageOverlay::new("hid", vec![1.0f32; 4], [1, 2, 2]).with_visible(Visibility::Hidden),
    );
    let r = s.visible_image_overlays();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].name, "vis");
}

// Test 7 – Colormap default
#[test]
fn test_colormap_default_is_grayscale() {
    assert_eq!(Colormap::default(), Colormap::Grayscale);
}

// Test 8 – JSON round-trip
#[test]
fn test_overlay_state_serde_round_trip() {
    let mut s = OverlayState::new();
    let mut img = ImageOverlay::new("img", vec![0.5f32, 0.75], [1, 1, 2]);
    img.opacity = Opacity::new(0.8);
    img.colormap = Colormap::Hot;
    s.add_image_overlay(img);
    let mut cnt = ContourOverlay::new("cnt", 42);
    cnt.add_contour(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .expect("infallible: validated precondition");
    s.add_contour_overlay(cnt);
    let msk = MaskOverlay::new("msk", vec![0u32, 1, 2, 0], [1, 2, 2]).with_opacity(0.6);
    s.add_mask_overlay(msk);
    let json = serde_json::to_string(&s).expect("infallible: validated precondition");
    let r: OverlayState = serde_json::from_str(&json).expect("infallible: validated precondition");
    assert_eq!(r.image_overlays.len(), 1);
    assert_eq!(r.image_overlays[0].name, "img");
    assert!((r.image_overlays[0].opacity.get() - 0.8).abs() < 1e-6);
    assert_eq!(r.image_overlays[0].colormap, Colormap::Hot);
    assert_eq!(r.image_overlays[0].data, vec![0.5f32, 0.75]);
    assert_eq!(r.contour_overlays.len(), 1);
    assert_eq!(r.contour_overlays[0].label_id, 42);
    assert_eq!(r.contour_overlays[0].contours[0].len(), 2);
    assert_eq!(r.mask_overlays.len(), 1);
    assert_eq!(r.mask_overlays[0].name, "msk");
    assert!((r.mask_overlays[0].opacity.get() - 0.6).abs() < 1e-6);
    assert_eq!(r.mask_overlays[0].data, vec![0u32, 1, 2, 0]);
}

// ── LabelId regression tests ──────────────────────────────────────────

/// LabelId serializes as a JSON integer (not an object), matching the
/// `#[repr(transparent)]` layout. This is the wire format that external
/// consumers (viewer, Python bindings) depend on.
#[test]
fn test_label_id_serde_json_is_integer() {
    let id = LabelId(42);
    let json = serde_json::to_string(&id).expect("infallible: validated precondition");
    assert_eq!(json, "42");
    let round_tripped: LabelId = serde_json::from_str(&json).expect("infallible: validated precondition");
    assert_eq!(round_tripped, LabelId(42));
}

/// LabelId::BACKGROUND serialises as 0 and round-trips.
#[test]
fn test_label_id_background_serde_round_trip() {
    let id = LabelId::BACKGROUND;
    let json = serde_json::to_string(&id).expect("infallible: validated precondition");
    assert_eq!(json, "0");
    let rt: LabelId = serde_json::from_str(&json).expect("infallible: validated precondition");
    assert_eq!(rt, LabelId::BACKGROUND);
}

/// LabelId at the u32 boundary (u32::MAX) round-trips through JSON
/// without truncation or overflow.
#[test]
fn test_label_id_u32_max_serde_round_trip() {
    let id = LabelId(u32::MAX);
    let json = serde_json::to_string(&id).expect("infallible: validated precondition");
    assert_eq!(json, u32::MAX.to_string());
    let rt: LabelId = serde_json::from_str(&json).expect("infallible: validated precondition");
    assert_eq!(rt, LabelId(u32::MAX));
}

/// ContourOverlay.label_id must survive a full JSON round-trip with its
/// exact LabelId value. This guards against the field being serialised
/// as a different type (e.g. string or object) after the u32 → LabelId
/// migration.
#[test]
fn test_contour_overlay_label_id_serde_exact() {
    let mut c = ContourOverlay::new("cnt", LabelId(77));
    c.add_contour(vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        .expect("infallible: validated precondition");
    let json = serde_json::to_string(&c).expect("infallible: validated precondition");
    let rt: ContourOverlay = serde_json::from_str(&json).expect("infallible: validated precondition");
    assert_eq!(rt.label_id, LabelId(77));
    assert_eq!(u32::from(rt.label_id), 77);
}

/// ContourOverlay.label_id must survive deserialisation from an existing
/// JSON payload that was written when the field was a bare `u32`.
/// Serialisation format is `#[repr(transparent)]` → integer, so
/// backward compatibility is automatic, but this test makes it explicit.
#[test]
fn test_contour_overlay_label_id_deser_from_u32_json() {
    let json = r#"{"name":"legacy","label_id":99,"contours":[],"color":[1.0,1.0,1.0,1.0],"line_width":1.0,"visible":"Visible"}"#;
    let c: ContourOverlay = serde_json::from_str(json).expect("infallible: validated precondition");
    assert_eq!(c.label_id, LabelId(99));
    assert_eq!(c.name, "legacy");
}
