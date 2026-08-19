use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use ritk_connectome::Parcellation;

pub struct HumanDataset {
    pub dwi: PathBuf,
    pub bvals: PathBuf,
    pub bvecs: PathBuf,
    pub labels: PathBuf,
    pub label_info: PathBuf,
}

impl HumanDataset {
    pub fn locate() -> Option<Self> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/diffusion/stanford_hardi");
        let dataset = Self {
            dwi: root.join("dwi.nii.gz"),
            bvals: root.join("dwi.bvals"),
            bvecs: root.join("dwi.bvecs"),
            labels: root.join("aparc-reduced.nii.gz"),
            label_info: root.join("label_info.txt"),
        };
        let volume_is_present = std::fs::metadata(&dataset.dwi)
            .map(|metadata| metadata.len() > 1_000_000)
            .unwrap_or(false);
        (volume_is_present
            && dataset.bvals.exists()
            && dataset.bvecs.exists()
            && dataset.labels.exists()
            && dataset.label_info.exists())
        .then_some(dataset)
    }
}

pub struct HumanAtlas {
    pub labels: Box<[u32]>,
    pub shape_zyx: [usize; 3],
    pub parcellation: Parcellation,
    region_names: Vec<(u32, String)>,
}

impl HumanAtlas {
    pub fn read(dataset: &HumanDataset) -> Result<Self> {
        let (labels, shape_zyx) = ritk_io::read_nifti_labels(&dataset.labels)
            .with_context(|| format!("reading labels from {}", dataset.labels.display()))?;
        let declared_region_names =
            parse_region_names(&std::fs::read_to_string(&dataset.label_info)?)?;
        let region_names = names_for_present_labels(&labels, declared_region_names)?;
        let connectome_labels = labels
            .iter()
            .map(|label| if *label <= 2 { 0 } else { *label })
            .collect::<Vec<_>>();
        let [depth, rows, columns] = shape_zyx;
        let parcellation = Parcellation::new(
            connectome_labels.into_boxed_slice(),
            [columns, rows, depth],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            region_names.clone(),
        )
        .context("constructing the grey-matter endpoint parcellation")?;
        anyhow::ensure!(
            parcellation.region_count() == region_names.len(),
            "label volume contains {} grey-matter regions but label_info declares {}",
            parcellation.region_count(),
            region_names.len()
        );
        Ok(Self {
            labels: labels.into_boxed_slice(),
            shape_zyx,
            parcellation,
            region_names,
        })
    }

    pub fn name(&self, label: u32) -> String {
        self.region_names
            .binary_search_by_key(&label, |(candidate, _)| *candidate)
            .ok()
            .map(|index| self.region_names[index].1.clone())
            .unwrap_or_else(|| format!("label {label}"))
    }
}

fn parse_region_names(contents: &str) -> Result<Vec<(u32, String)>> {
    let mut names = BTreeMap::new();
    for (line_number, line) in contents.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.splitn(3, ',');
        let label = fields
            .next()
            .context("missing reduced label")?
            .trim()
            .parse::<u32>()
            .with_context(|| format!("invalid reduced label on line {}", line_number + 1))?;
        let _freesurfer_label = fields.next().context("missing FreeSurfer label")?;
        let name = fields
            .next()
            .context("missing FreeSurfer region name")?
            .trim()
            .trim_matches('"');
        if label > 2 {
            names.entry(label).or_insert_with(|| name.to_owned());
        }
    }
    anyhow::ensure!(
        !names.is_empty(),
        "label_info contains no grey-matter regions"
    );
    Ok(names.into_iter().collect())
}

fn names_for_present_labels(
    labels: &[u32],
    declared_names: Vec<(u32, String)>,
) -> Result<Vec<(u32, String)>> {
    let present_labels = labels
        .iter()
        .copied()
        .filter(|label| *label > 2)
        .collect::<BTreeSet<_>>();
    let region_names = declared_names
        .into_iter()
        .filter(|(label, _)| present_labels.contains(label))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        region_names.len() == present_labels.len(),
        "label_info names {} of {} grey-matter labels present in the image",
        region_names.len(),
        present_labels.len()
    );
    Ok(region_names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_label_parser_excludes_white_matter_and_deduplicates() -> Result<()> {
        let text = "new label, freesurfer label, freesurfer name\n\
                    1, 2, \"Left-Cerebral-White-Matter\"\n\
                    1, 41, \"Right-Cerebral-White-Matter\"\n\
                    3, 1032, \"ctx-lh-frontalpole\"\n\
                    46, 2032, \"ctx-rh-frontalpole\"\n\
                    \n";
        assert_eq!(
            parse_region_names(text)?,
            vec![
                (3, "ctx-lh-frontalpole".to_owned()),
                (46, "ctx-rh-frontalpole".to_owned())
            ]
        );
        Ok(())
    }

    #[test]
    fn region_names_follow_labels_present_in_the_image() -> Result<()> {
        let names = names_for_present_labels(
            &[0, 1, 3, 3, 46],
            vec![
                (3, "left".to_owned()),
                (4, "declared but absent".to_owned()),
                (46, "right".to_owned()),
            ],
        )?;
        assert_eq!(
            names,
            vec![(3, "left".to_owned()), (46, "right".to_owned())]
        );
        Ok(())
    }

    #[test]
    fn every_present_region_requires_a_name() {
        let error = names_for_present_labels(&[3, 46], vec![(3, "left".to_owned())])
            .expect_err("label 46 is present but unnamed");
        assert_eq!(
            error.to_string(),
            "label_info names 1 of 2 grey-matter labels present in the image"
        );
    }
}
