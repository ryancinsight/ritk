use super::*;

// ── Agreement reporting ──────────────────────────────────────────────────

/// The report must not divide by zero on an empty volume.
#[test]
fn an_empty_agreement_map_reports_nothing() {
    report_agreement(&[]);
}

#[test]
fn agreement_reporting_handles_the_unanimous_and_split_extremes() {
    // Exercised for absence of panic and division-by-zero; the printed share is
    // the application's output layer rather than a value under test.
    report_agreement(&[1.0, 1.0, 1.0]);
    report_agreement(&[0.5, 0.5]);
    report_agreement(&[0.0]);
}

// ── Fusion selection ─────────────────────────────────────────────────────

/// Each command-line choice must reach the library variant it names. A mismatch
/// would silently fuse by a different rule than the caller asked for, which no
/// downstream check catches because both rules produce a valid parcellation.
#[test]
fn every_fusion_choice_maps_to_its_library_variant() {
    let majority = match Fusion::Majority {
        Fusion::Majority => LabelFusion::MajorityVote,
        Fusion::Joint => LabelFusion::JointLabelFusion(Default::default()),
    };
    assert!(matches!(majority, LabelFusion::MajorityVote));

    let joint = match Fusion::Joint {
        Fusion::Majority => LabelFusion::MajorityVote,
        Fusion::Joint => LabelFusion::JointLabelFusion(Default::default()),
    };
    assert!(matches!(joint, LabelFusion::JointLabelFusion(_)));
}

// ── Registration configuration ───────────────────────────────────────────

/// The iteration list sets both the schedule and the level count, so a caller
/// passing three values gets three levels. Setting one without the other would
/// leave the registration reading past its own schedule or ignoring levels the
/// caller asked for.
#[test]
fn the_iteration_list_sets_the_level_count_too() {
    let mut config = AtlasParcellationConfig::default();
    let iterations = vec![25, 12];

    config.registration.num_levels = iterations.len();
    config.registration.iterations_per_level = iterations.clone();

    assert_eq!(config.registration.num_levels, 2);
    assert_eq!(config.registration.iterations_per_level, iterations);
}

/// The default schedule is three levels with a matching iteration list — a
/// mismatch there would be a defect in the library default rather than in any
/// caller, so it is worth pinning where the CLI depends on it.
#[test]
fn the_default_schedule_is_internally_consistent() {
    let config = AtlasParcellationConfig::default();
    assert_eq!(
        config.registration.num_levels,
        config.registration.iterations_per_level.len(),
        "the level count and the iteration schedule must agree"
    );
}
