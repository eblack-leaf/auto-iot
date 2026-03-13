//! Descriptive statistics over dataset samples.

use crate::datasets::Sample;

pub fn print_stats(samples: &[Sample], seq_len: usize) {
    if samples.is_empty() {
        println!("  (no samples)");
        return;
    }

    let n = samples.len();
    let n_anomaly = samples.iter().filter(|s| s.label == Some(1)).count();
    let n_normal = samples.iter().filter(|s| s.label == Some(0)).count();
    let n_unlabelled = samples.iter().filter(|s| s.label.is_none()).count();

    println!("  Samples   : {}", n);
    println!("  Seq length: {}", seq_len);
    if n_unlabelled < n {
        println!(
            "  Labels    : {} normal, {} anomaly, {} unlabelled",
            n_normal, n_anomaly, n_unlabelled
        );
        println!(
            "  Anomaly % : {:.2}%",
            n_anomaly as f64 / n as f64 * 100.0
        );
    } else {
        println!("  Labels    : unlabelled");
    }

    // Per-feature statistics (averaged across features for compactness)
    let all_values: Vec<f32> = samples.iter().flat_map(|s| s.features.iter().copied()).collect();
    let global_stats = compute_stats(&all_values);

    println!("  ── Global feature statistics (all dims aggregated) ──");
    println!("    Mean   : {:.4}", global_stats.mean);
    println!("    Std    : {:.4}", global_stats.std);
    println!("    Min    : {:.4}", global_stats.min);
    println!("    Max    : {:.4}", global_stats.max);
    println!("    Median : {:.4}", global_stats.median);
    println!("    Kurtosis (excess): {:.4}", global_stats.excess_kurtosis);
    println!("    Skewness         : {:.4}", global_stats.skewness);
}

#[derive(Debug)]
pub struct Stats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub excess_kurtosis: f64,
    pub skewness: f64,
}

pub fn compute_stats(values: &[f32]) -> Stats {
    if values.is_empty() {
        return Stats {
            mean: 0.0, std: 0.0, min: 0.0, max: 0.0,
            median: 0.0, excess_kurtosis: 0.0, skewness: 0.0,
        };
    }

    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    let mut sorted: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let skewness = if std > 1e-10 {
        values.iter().map(|&v| ((v as f64 - mean) / std).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };

    let excess_kurtosis = if std > 1e-10 {
        values.iter().map(|&v| ((v as f64 - mean) / std).powi(4)).sum::<f64>() / n - 3.0
    } else {
        0.0
    };

    Stats {
        mean,
        std,
        min: sorted[0],
        max: *sorted.last().unwrap(),
        median,
        excess_kurtosis,
        skewness,
    }
}
