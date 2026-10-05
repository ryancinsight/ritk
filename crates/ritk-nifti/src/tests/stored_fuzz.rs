use proptest::prelude::*;
use ritk_image_io::{ImageReadBudget, StoredVolume};

use crate::header::{
    write_single_file_bytes, HeaderDims, HeaderSpatial, NiftiDatatype, NiftiHeader,
};
use crate::read_nifti_stored_from_bytes;

const INPUT_BYTE_LIMIT: usize = 1024;

fn budget() -> ImageReadBudget {
    let limit = u64::try_from(INPUT_BYTE_LIMIT).expect("test limit fits u64");
    ImageReadBudget::new(limit, limit, 1).expect("test limits are positive")
}

fn valid_fixture() -> Vec<u8> {
    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 2,
            ny: 2,
            nz: 2,
        },
        NiftiDatatype::Uint16,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )
    .expect("fixture header is valid");
    let payload = [0_u16; 8].map(u16::to_le_bytes).concat();
    write_single_file_bytes(&header, &payload)
}

fn assert_valid_result(bytes: &[u8]) -> Result<(), TestCaseError> {
    if let Ok(volume) = read_nifti_stored_from_bytes(bytes, budget()) {
        assert_volume_invariants(&volume)?;
    }
    Ok(())
}

fn assert_volume_invariants(volume: &StoredVolume) -> Result<(), TestCaseError> {
    let shape = volume.shape();
    prop_assert!(shape.into_iter().all(|axis| axis > 0));
    let voxel_count = shape.into_iter().try_fold(1_usize, usize::checked_mul);
    prop_assert_eq!(Some(volume.samples().len()), voxel_count);
    let decoded_bytes = voxel_count
        .and_then(|count| count.checked_mul(volume.samples().sample_type().byte_width()));
    prop_assert!(decoded_bytes.is_some_and(|count| count <= INPUT_BYTE_LIMIT));
    Ok(())
}

proptest! {
    #[test]
    fn arbitrary_bounded_bytes_are_rejected_or_form_a_valid_volume(
        bytes in prop::collection::vec(any::<u8>(), 0..=INPUT_BYTE_LIMIT)
    ) {
        assert_valid_result(&bytes)?;
    }

    #[test]
    fn single_byte_header_mutations_are_rejected_or_form_a_valid_volume(
        offset in 0_usize..352,
        replacement in any::<u8>()
    ) {
        let mut bytes = valid_fixture();
        let slot = bytes.get_mut(offset).expect("header offset is in bounds");
        *slot = replacement;
        assert_valid_result(&bytes)?;
    }
}
