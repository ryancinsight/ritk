mod catalog;
#[cfg(test)]
mod tests;

pub use catalog::{
    AntsExampleDataset, BrainWebDataset, Dataset, DatasetManager, IxiDataset, Learn2RegDataset,
    OasisDataset, OpenNeuroDataset, SynthStripDataset,
};
