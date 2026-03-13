//! ECG5000 dataset from the UCR Time Series Archive.
//!
//! Format: whitespace-delimited text files.  First column is the class label
//! (1 = normal, 2–5 = anomaly variants).  The remaining 140 columns are the
//! ECG time series.
//!
//! Source: <http://www.timeseriesclassification.com/Downloads/ECG5000.zip>

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use super::normalizer::MinMaxNormalizer;
use super::{AnomalyDataset, Sample};

const DOWNLOAD_URL: &str =
    "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip";

pub const SEQ_LEN: usize = 140;

pub struct Ecg5000 {
    train: Vec<Sample>,
    val: Vec<Sample>,
    test: Vec<Sample>,
}

impl Ecg5000 {
    /// Download and cache the dataset.
    pub fn fetch(data_dir: &str) -> Result<()> {
        let dir = PathBuf::from(data_dir).join("ecg5000");
        if dir.join("ECG5000_TRAIN.txt").exists() {
            println!("[ecg5000] Already cached at {}", dir.display());
            return Ok(());
        }
        fs::create_dir_all(&dir)?;

        let zip_path = dir.join("ECG5000.zip");
        download_file(DOWNLOAD_URL, &zip_path)?;
        extract_zip(&zip_path, &dir)?;
        println!("[ecg5000] Ready at {}", dir.display());
        Ok(())
    }

    /// Load from disk and split into train / val / test.
    /// ECG5000_TRAIN (500 rows) → 80 % train, 20 % val.
    /// ECG5000_TEST  (4500 rows) → test.
    pub fn load(data_dir: &str, val_split: f32) -> Result<Self> {
        let dir = PathBuf::from(data_dir).join("ecg5000");
        let train_path = find_file(&dir, "TRAIN")?;
        let test_path = find_file(&dir, "TEST")?;

        let all_train_raw = parse_file(&train_path)?;
        let all_test_raw = parse_file(&test_path)?;

        // Fit normaliser on raw training features only.
        let train_features: Vec<Vec<f32>> = all_train_raw.iter().map(|(_, f)| f.clone()).collect();
        let norm = MinMaxNormalizer::fit(&train_features);

        let all_train: Vec<(u8, Vec<f32>)> = all_train_raw
            .into_iter()
            .map(|(l, f)| (l, norm.transform(&f)))
            .collect();

        // Remap to binary 0=normal 1=anomaly everywhere for consistency with other datasets.
        let to_sample = |(l, f): &(u8, Vec<f32>)| Sample {
            features: f.clone(),
            label: Some(if *l == 1 { 0 } else { 1 }),
        };

        let split = (all_train.len() as f32 * (1.0 - val_split)) as usize;
        let train_portion = &all_train[..split];
        let val_portion = &all_train[split..];

        // Normal samples from the train portion → training set.
        // Anomaly samples from the train portion → recycled into validation so early
        // stopping sees both classes and val MSE is a better proxy for AUROC.
        let train: Vec<Sample> = train_portion
            .iter()
            .filter(|(l, _)| *l == 1)
            .map(to_sample)
            .collect();

        let mut val: Vec<Sample> = val_portion.iter().map(to_sample).collect();
        val.extend(
            train_portion
                .iter()
                .filter(|(l, _)| *l != 1)
                .map(to_sample),
        );

        let test: Vec<Sample> = all_test_raw
            .into_iter()
            .map(|(l, f)| Sample {
                features: norm.transform(&f),
                // label 1 = normal, 2-5 = anomaly → remap to binary
                label: Some(if l == 1 { 0 } else { 1 }),
            })
            .collect();

        Ok(Ecg5000 { train, val, test })
    }
}

impl AnomalyDataset for Ecg5000 {
    fn name(&self) -> &str {
        "ecg5000"
    }
    fn seq_len(&self) -> usize {
        SEQ_LEN
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

/// Parse a whitespace-delimited ECG file: (label, features).
fn parse_file(path: &Path) -> Result<Vec<(u8, Vec<f32>)>> {
    let file = fs::File::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();
        if cols.len() < 2 {
            continue;
        }
        let label = cols[0] as u8;
        let features = cols[1..].to_vec();
        rows.push((label, features));
    }
    Ok(rows)
}

fn find_file(dir: &Path, suffix: &str) -> Result<PathBuf> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.to_uppercase().contains(suffix) && name.ends_with(".txt") {
                return Ok(path);
            }
        }
    }
    anyhow::bail!(
        "Could not find *{}*.txt in {} — run `auto-iot fetch --dataset ecg5000`",
        suffix,
        dir.display()
    )
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    println!("[ecg5000] Downloading {} …", url);
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.set_message("Connecting…");

    let bytes = reqwest::blocking::get(url)
        .with_context(|| format!("GET {url}"))?
        .bytes()
        .with_context(|| format!("Reading body from {url}"))?;

    if bytes.len() < 1024 {
        anyhow::bail!(
            "Download from {url} returned only {} bytes — likely an error page, not the archive.",
            bytes.len()
        );
    }
    pb.finish_with_message(format!("Downloaded {} bytes", bytes.len()));
    fs::write(dest, &bytes)?;
    Ok(())
}

fn extract_zip(zip_path: &Path, dest: &Path) -> Result<()> {
    use std::io;
    println!("[ecg5000] Extracting {} …", zip_path.display());
    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let out = dest.join(entry.name());
        if entry.is_dir() {
            fs::create_dir_all(&out)?;
        } else {
            if let Some(p) = out.parent() {
                fs::create_dir_all(p)?;
            }
            let mut out_file = fs::File::create(&out)?;
            io::copy(&mut entry, &mut out_file)?;
        }
    }
    Ok(())
}

// Public re-export so datasets/mod.rs can call fetch without knowing the type.
pub fn fetch(data_dir: &str) -> Result<()> {
    Ecg5000::fetch(data_dir)
}
