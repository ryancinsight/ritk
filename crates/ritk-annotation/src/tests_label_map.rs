use super::*;

fn empty_table() -> LabelTable {
    LabelTable::new()
}

#[test]
fn test_label_map_new_all_background() {
    let lm = LabelMap::new([2, 3, 4], empty_table());
    assert!(
        lm.as_slice().iter().all(|&v| v == 0),
        "all voxels must be 0 (background) on construction"
    );
    assert_eq!(lm.num_voxels(), 24);
}

#[test]
fn test_label_map_set_and_get() {
    let mut lm = LabelMap::new([4, 5, 6], empty_table());
    lm.set_label_at([1, 2, 3], 5);
    assert_eq!(lm.label_at([1, 2, 3]), 5);
    assert_eq!(lm.label_at([0, 0, 0]), 0);
    assert_eq!(lm.label_at([3, 4, 5]), 0);
}

#[test]
fn test_label_map_from_data_valid() {
    let data: Vec<u32> = (0u32..60).collect();
    let lm = LabelMap::from_data([3, 4, 5], data.clone(), empty_table()).expect("infallible: validated precondition");
    assert_eq!(lm.as_slice(), data.as_slice());
}

#[test]
fn test_label_map_from_data_wrong_len() {
    let data = vec![0u32; 10];
    let result = LabelMap::from_data([3, 4, 5], data, empty_table());
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(msg.contains("10") && msg.contains("60"), "{}", msg);
}

#[test]
fn test_label_map_mask_for_label() {
    let mut lm = LabelMap::new([2, 2, 2], empty_table());
    lm.set_label_at([0, 0, 1], 2);
    lm.set_label_at([1, 1, 0], 2);
    let mask = lm.mask_for_label(2);
    assert_eq!(mask.len(), 8);
    // flat([0,0,1]) = 0*4 + 0*2 + 1 = 1
    // flat([1,1,0]) = 1*4 + 1*2 + 0 = 6
    assert!(mask[1], "voxel [0,0,1] must be true");
    assert!(mask[6], "voxel [1,1,0] must be true");
    assert_eq!(mask.iter().filter(|&&b| b).count(), 2);
}

#[test]
fn test_label_map_count_label() {
    let mut lm = LabelMap::new([3, 3, 3], empty_table());
    for pos in [[0, 0, 0], [0, 0, 1], [1, 1, 1], [2, 2, 2]] {
        lm.set_label_at(pos, 7);
    }
    assert_eq!(lm.count_label(7), 4);
    assert_eq!(lm.count_label(0), 27 - 4);
}

#[test]
fn test_label_map_present_labels() {
    let mut lm = LabelMap::new([2, 2, 2], empty_table());
    lm.set_label_at([0, 0, 0], 1);
    lm.set_label_at([0, 0, 1], 3);
    lm.set_label_at([1, 0, 0], 7);
    let present = lm.present_labels();
    assert_eq!(present, vec![0u32, 1, 3, 7]);
}

#[test]
fn test_label_map_num_voxels() {
    let lm = LabelMap::new([4, 5, 6], empty_table());
    assert_eq!(lm.num_voxels(), 120);
}
