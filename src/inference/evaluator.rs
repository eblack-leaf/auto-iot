//! Evaluation metrics for anomaly detection:
//! AUROC (trapezoidal rule), best-F1, precision, recall.

/// Reconstruction error per sample — index mirrors the test set order.
pub type AnomalyScore = f32;

/// Compute the threshold at the given percentile of reconstruction errors.
pub fn percentile_threshold(scores: &[f32], pct: f64) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Compute and print AUROC + best-threshold F1 / precision / recall.
pub fn evaluate(scores: &[f32], labels: &[u8], threshold: f32) {
    assert_eq!(scores.len(), labels.len(), "scores and labels must match");

    let n = scores.len();
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = n - n_pos;

    if n_pos == 0 {
        println!("  No positive (anomaly) samples in labels — AUROC undefined.");
        return;
    }

    // ── AUROC ─────────────────────────────────────────────────────────────
    let mut pairs: Vec<(f32, u8)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // descending score

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut prev_tpr = 0.0_f64;
    let mut prev_fpr = 0.0_f64;
    let mut auc = 0.0_f64;

    for (_, label) in &pairs {
        if *label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        let tpr = tp as f64 / n_pos as f64;
        let fpr = fp as f64 / n_neg as f64;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }

    // ── Threshold metrics ─────────────────────────────────────────────────
    let predicted: Vec<u8> = scores.iter().map(|&s| if s >= threshold { 1 } else { 0 }).collect();
    let tp_f1 = labels
        .iter()
        .zip(predicted.iter())
        .filter(|(&l, &p)| l == 1 && p == 1)
        .count() as f64;
    let fp_f1 = labels
        .iter()
        .zip(predicted.iter())
        .filter(|(&l, &p)| l == 0 && p == 1)
        .count() as f64;
    let fn_f1 = labels
        .iter()
        .zip(predicted.iter())
        .filter(|(&l, &p)| l == 1 && p == 0)
        .count() as f64;

    let precision = if tp_f1 + fp_f1 > 0.0 {
        tp_f1 / (tp_f1 + fp_f1)
    } else {
        0.0
    };
    let recall = if tp_f1 + fn_f1 > 0.0 {
        tp_f1 / (tp_f1 + fn_f1)
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("\n  ┌── Evaluation Results ──────────────────");
    println!("  │  Samples  : {} ({} anomalies, {} normal)", n, n_pos, n_neg);
    println!("  │  AUROC    : {:.4}", auc);
    println!("  │  Threshold: {:.6}", threshold);
    println!("  │  Precision: {:.4}", precision);
    println!("  │  Recall   : {:.4}", recall);
    println!("  │  F1       : {:.4}", f1);
    println!("  └────────────────────────────────────────");
}
