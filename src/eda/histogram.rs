//! Terminal ASCII histogram.

pub fn print_histogram(values: &[f64], n_bins: usize) {
    if values.is_empty() {
        println!("  (no data)");
        return;
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-10 {
        println!("  All values identical: {:.4}", min);
        return;
    }

    let bin_width = (max - min) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];

    for &v in values {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);
    let bar_width = 40;

    println!(
        "  [{:.3}  …  {:.3}]  n={} bins={}",
        min, max, values.len(), n_bins
    );
    println!();

    for (i, &count) in counts.iter().enumerate() {
        let lo = min + i as f64 * bin_width;
        let hi = lo + bin_width;
        let bar_len = (count as f64 / max_count as f64 * bar_width as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "  [{:>7.3}, {:>7.3})  {:>5}  {}",
            lo, hi, count, bar
        );
    }
}

/// Print a histogram of reconstruction errors, highlighting the threshold.
pub fn print_error_histogram(errors: &[f32], threshold: f32, n_bins: usize) {
    let values: Vec<f64> = errors.iter().map(|&e| e as f64).collect();
    print_histogram(&values, n_bins);
    println!(
        "  threshold = {:.6}  (shown as vertical mark if above min)",
        threshold
    );
}
