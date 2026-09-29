use super::*;

#[test]
fn native_oblique_patient_length_clicks_use_the_rendered_plane() {
    let mut session = oblique_session();
    let initial_axis = session.app.axis;
    let plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("validated oblique plane");
    let [width, height] = plane.dimensions();
    assert!(
        width > 2 && height > 2,
        "fixture plane provides interior measurement points"
    );
    let focus_pixel = [
        f64::from(u32::try_from(width - 1).expect("width")) * 0.5,
        f64::from(u32::try_from(height - 1).expect("height")) * 0.5,
    ];
    let (focus_x, focus_y) = screen_point(&session, focus_pixel);
    click(&mut session, focus_x, focus_y);
    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::ui::tool_shortcuts::VIRTUAL_KEY_MEASURE_LENGTH,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select length tool while oblique panel is focused");
    assert_eq!(session.app.active_tool, ToolKind::MeasureLength);
    let points = [
        [0.0, 0.0],
        [
            f64::from(u32::try_from(width - 1).expect("width")),
            f64::from(u32::try_from(height - 1).expect("height")),
        ],
    ];
    let viewport = session.oblique_viewport.expect("oblique mapper");
    let mut expected_patients = Vec::new();
    let mut expected_voxels = Vec::new();
    for pixel in points {
        let (x, y) = screen_point(&session, pixel);
        let pixel = viewport
            .map(crate::presentation::ViewportPoint::new(
                f64::from(x),
                f64::from(y),
            ))
            .expect("native pointer maps to the rendered plane");
        let sample = plane
            .sample_pixel(session.app.loaded.as_ref().expect("loaded fixture"), pixel)
            .expect("rendered pixel samples source volume");
        expected_patients.push(sample.patient());
        expected_voxels.push(sample.nearest_voxel());
        click(&mut session, x, y);
    }

    assert_eq!(session.app.axis, initial_axis);
    let cursor = session
        .app
        .linked_cursor
        .expect("oblique click updates cursor");
    assert_eq!(cursor.voxel(), *expected_voxels.last().expect("two clicks"));
    assert_eq!(session.app.viewer_state.slice_index, cursor.voxel()[0]);
    assert_eq!(session.app.coronal_slice, cursor.voxel()[1]);
    assert_eq!(session.app.sagittal_slice, cursor.voxel()[2]);
    let Some(Annotation::PatientLength(length)) = session.app.annotations.last() else {
        panic!("native oblique clicks create a patient-space length")
    };
    assert_eq!(length.start_mm().coordinates(), expected_patients[0]);
    assert_eq!(length.end_mm().coordinates(), expected_patients[1]);
    let expected_length = (0..3)
        .map(|axis| {
            let delta = expected_patients[1][axis] - expected_patients[0][axis];
            delta * delta
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        (length.length_mm() - expected_length).abs() <= 6.0 * f64::EPSILON * expected_length,
        "three-dimensional Euclidean distance matches the independently summed squared displacement"
    );

    let display = super::super::layout::patient_measurement_overlay(
        &session.app.annotations,
        &session.app.tool_state,
        session.oblique_viewport.expect("current mapper"),
        session
            .oblique
            .as_ref()
            .and_then(|oblique| oblique.plane)
            .as_ref()
            .expect("current plane"),
    )
    .expect("patient-space overlay");
    assert!(display.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawText { text, .. } if text == &format!("{:.1} mm", length.length_mm())
    )));
    assert!(display.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawLine { color, .. }
            if *color == metis_platform::Color::rgba(255, 235, 59, 255)
    )));
    assert_ne!(session.app.linked_cursor, None);
}
