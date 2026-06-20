use super::Dataset;

/// Example brain MRI data from OpenNeuro (ds000102 - Flanker task)
/// Small subset for testing
pub struct OpenNeuroDataset;

impl OpenNeuroDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for OpenNeuroDataset {
    fn name(&self) -> &'static str {
        "openneuro"
    }

    fn description(&self) -> &'static str {
        "OpenNeuro ds000102 - Sample fMRI dataset for testing"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // OpenNeuro has S3 buckets with public access
        // Using a small test NIfTI file
        vec![(
            "https://s3.amazonaws.com/openneuro.org/ds000102/sub-01/anat/sub-01_T1w.nii.gz",
            "",
        )]
    }
}

/// ANTs example data (small test brain)
pub struct AntsExampleDataset;

impl AntsExampleDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for AntsExampleDataset {
    fn name(&self) -> &'static str {
        "ants_example"
    }

    fn description(&self) -> &'static str {
        "ANTs example brain data for registration testing"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // Using niivue-demo-images which are reliable
        vec![
            // MNI152 template (Standard space)
            (
                "https://github.com/niivue/niivue-demo-images/raw/main/mni152.nii.gz",
                "",
            ),
            // Visible Human (Another brain to register)
            (
                "https://github.com/niivue/niivue-demo-images/raw/main/visiblehuman.nii.gz",
                "",
            ),
        ]
    }
}

/// BrainWeb simulated brain data
pub struct BrainWebDataset;

impl BrainWebDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for BrainWebDataset {
    fn name(&self) -> &'static str {
        "brainweb"
    }

    fn description(&self) -> &'static str {
        "BrainWeb simulated brain MRI (requires manual download from brainweb.bic.mni.mcgill.ca)"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // BrainWeb requires form submission
        vec![]
    }
}

/// OASIS Brains Dataset (requires registration)
pub struct OasisDataset;

impl OasisDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for OasisDataset {
    fn name(&self) -> &'static str {
        "oasis"
    }

    fn description(&self) -> &'static str {
        "OASIS Brains Dataset (416 MR sessions) - Requires registration at https://www.oasis-brains.org/"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // OASIS requires registration
        vec![]
    }
}

/// IXI Dataset (small subset)
pub struct IxiDataset;

impl IxiDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for IxiDataset {
    fn name(&self) -> &'static str {
        "ixi"
    }

    fn description(&self) -> &'static str {
        "IXI Dataset (~600 MR brain images) - Download from https://brain-development.org/ixi-dataset/"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // IXI has direct download links for sample data
        vec![]
    }
}

/// Learn2Reg Challenge Dataset
pub struct Learn2RegDataset;

impl Learn2RegDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for Learn2RegDataset {
    fn name(&self) -> &'static str {
        "learn2reg"
    }

    fn description(&self) -> &'static str {
        "Learn2Reg Challenge datasets - Available at https://learn2reg.grand-challenge.org/"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // Learn2Reg datasets are hosted on Zenodo
        vec![]
    }
}

/// SynthStrip brain MRI data (freely available)
pub struct SynthStripDataset;

impl SynthStripDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Dataset for SynthStripDataset {
    fn name(&self) -> &'static str {
        "synthstrip"
    }

    fn description(&self) -> &'static str {
        "SynthStrip test brain MRI data - FreeSurfer project"
    }

    fn urls(&self) -> Vec<(&'static str, &'static str)> {
        // SynthStrip has some test data available
        vec![]
    }
}
