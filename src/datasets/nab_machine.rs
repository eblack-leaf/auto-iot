//! NAB Machine Temperature dataset — Numenta Anomaly Benchmark.
//!
//! Temperature readings (5-minute intervals, ~79 days) from an internal
//! component of a large industrial machine.  Two labelled anomaly windows
//! correspond to known system failures.
//!
//! Raw CSV: timestamp, value columns.
//! Anomaly labels: the same NAB `combined_labels.json` used by nab-taxi.
//!
//! We convert the univariate time series into sliding windows of `window`
//! length.  A window is anomalous if any of its timestamps appears in the
//! combined labels list.
//!
//! Sources:
//!   Data:   https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv
//!   Labels: https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

use super::normalizer::MinMaxNormalizer;
use super::{AnomalyDataset, Sample};

const DATA_URL: &str =
    "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv";
const LABELS_URL: &str =
    "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json";
const LABEL_KEY: &str = "realKnownCause/machine_temperature_system_failure.csv";

pub struct NabMachine {
    train: Vec<Sample>,
    val: Vec<Sample>,
    test: Vec<Sample>,
    window: usize,
}

impl NabMachine {
    pub fn fetch(data_dir: &str) -> Result<()> {
        let dir = PathBuf::from(data_dir).join("nab_machine");
        if dir.join("machine_temperature.csv").exists() && dir.join("labels.json").exists() {
            println!("[nab-machine] Already cached at {}", dir.display());
            return Ok(());
        }
        fs::create_dir_all(&dir)?;
        download_file(DATA_URL, &dir.join("machine_temperature.csv"), "machine_temperature.csv")?;
        download_file(LABELS_URL, &dir.join("labels.json"), "labels.json")?;
        println!("[nab-machine] Ready at {}", dir.display());
        Ok(())
    }

    pub fn load(data_dir: &str, window: usize, val_split: f32) -> Result<Self> {
        let dir = PathBuf::from(data_dir).join("nab_machine");

        let csv_path = dir.join("machine_temperature.csv");
        if !csv_path.exists() {
            anyhow::bail!(
                "nab-machine not downloaded — run `auto-iot fetch --dataset nab-machine`"
            );
        }

        let mut timestamps: Vec<String> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        let mut rdr = csv::Reader::from_path(&csv_path)?;
        for result in rdr.records() {
            let rec = result?;
            timestamps.push(rec[0].to_string());
            values.push(rec[1].trim().parse::<f32>().unwrap_or(0.0));
        }

        // Load anomaly timestamps
        let labels_path = dir.join("labels.json");
        let anomaly_set: HashSet<String> = if labels_path.exists() {
            let json_str = fs::read_to_string(&labels_path)?;
            let json: Value = serde_json::from_str(&json_str)?;
            json[LABEL_KEY]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            HashSet::new()
        };

        // Build sliding windows
        let n = values.len();
        if n < window {
            anyhow::bail!("NAB machine series length {} < window {}", n, window);
        }

        let raw_windows: Vec<Vec<f32>> = (0..=n - window)
            .map(|i| values[i..i + window].to_vec())
            .collect();

        let labels: Vec<u8> = (0..=n - window)
            .map(|i| {
                let is_anomaly = (i..i + window).any(|j| {
                    j < timestamps.len() && anomaly_set.contains(&timestamps[j])
                });
                if is_anomaly { 1 } else { 0 }
            })
            .collect();

        // Keep last 15% as test; split the rest into train/val by val_split.
        let split_val = (raw_windows.len() as f32 * 0.85) as usize;
        let split_train = (split_val as f32 * (1.0 - val_split)) as usize;

        let norm = MinMaxNormalizer::fit(&raw_windows[..split_train]);

        let to_samples = |windows: &[Vec<f32>], lbls: &[u8]| -> Vec<Sample> {
            windows
                .iter()
                .zip(lbls.iter())
                .map(|(w, &l)| Sample {
                    features: norm.transform(w),
                    label: Some(l),
                })
                .collect()
        };

        let train: Vec<Sample> = raw_windows[..split_train]
            .iter()
            .zip(labels[..split_train].iter())
            .map(|(w, &l)| Sample { features: norm.transform(w), label: Some(l) })
            .collect();

        let val = to_samples(
            &raw_windows[split_train..split_val],
            &labels[split_train..split_val],
        );

        let test = to_samples(&raw_windows[split_val..], &labels[split_val..]);

        Ok(NabMachine { train, val, test, window })
    }
}

impl AnomalyDataset for NabMachine {
    fn name(&self) -> &str {
        "nab-machine"
    }
    fn seq_len(&self) -> usize {
        self.window
    }
    fn train_samples(&self) -> &[Sample] {
        &self.train
    }
    fn val_samples(&self) -> &[Sample] {
        &self.val
    }
    fn test_samples(&self) -> &[Sample] {
        &self.test
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn download_file(url: &str, dest: &PathBuf, label: &str) -> Result<()> {
    println!("[nab-machine] Downloading {} …", label);
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("Fetching {label}…"));
    let bytes = reqwest::blocking::get(url)
        .with_context(|| format!("GET {url}"))?
        .bytes()
        .with_context(|| format!("Reading body from {url}"))?;
    pb.finish_with_message(format!("{label}: {} bytes", bytes.len()));
    fs::write(dest, &bytes)?;
    Ok(())
}

pub fn fetch(data_dir: &str) -> Result<()> {
    NabMachine::fetch(data_dir)
}
