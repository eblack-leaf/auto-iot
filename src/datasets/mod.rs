pub mod nab_machine;
pub mod nab_taxi;
pub mod normalizer;
pub mod synthetic;

pub use nab_machine::NabMachine;
pub use nab_taxi::NabTaxi;
pub use synthetic::Synthetic;

use anyhow::Result;

/// A single sample ready for the model.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Flattened feature vector (already normalised).
    pub features: Vec<f32>,
    /// 1 = anomaly, 0 = normal, None = unlabelled.
    pub label: Option<u8>,
}

/// Common interface implemented by every dataset.
pub trait AnomalyDataset: Send + Sync {
    fn name(&self) -> &str;
    fn seq_len(&self) -> usize;

    fn train_samples(&self) -> &[Sample];
    fn val_samples(&self) -> &[Sample];
    fn test_samples(&self) -> &[Sample];

    /// True if the test split carries ground-truth labels.
    fn has_labels(&self) -> bool {
        self.test_samples().iter().any(|s| s.label.is_some())
    }
}

/// Download a dataset by name into `output_dir`.
pub fn fetch(dataset: &str, output_dir: &str) -> Result<()> {
    match dataset {
        "nab-machine" => nab_machine::fetch(output_dir),
        "nab-taxi" => nab_taxi::fetch(output_dir),
        "all" => {
            nab_machine::fetch(output_dir)?;
            nab_taxi::fetch(output_dir)?;
            Ok(())
        }
        other => anyhow::bail!(
            "Unknown dataset '{}'. Valid fetch targets: nab-machine, nab-taxi, all",
            other
        ),
    }
}

/// Load a dataset (already downloaded) and return a boxed trait object.
pub fn load(name: &str, data_dir: &str, window: usize, val_split: f32) -> Result<Box<dyn AnomalyDataset>> {
    match name {
        "nab-machine" => Ok(Box::new(NabMachine::load(data_dir, window, val_split)?)),
        "nab-taxi" => Ok(Box::new(NabTaxi::load(data_dir, window, val_split)?)),
        "synthetic" => Ok(Box::new(Synthetic::generate(window, 3000, 0.05, val_split))),
        other => anyhow::bail!(
            "Unknown dataset '{}'. Valid values: nab-machine, nab-taxi, synthetic",
            other
        ),
    }
}
