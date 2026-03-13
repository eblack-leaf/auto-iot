//! t-SNE on latent vectors exported by latent_export.rs.
//!
//! Reads `latent_vectors.csv`, runs Barnes-Hut t-SNE via the `bhtsne` crate
//! (Euclidean distance in latent space, 2-D embedding), and writes `tsne_embedding.csv`.
//!
//! bhtsne API: `tSNE<T, U>` where U is the sample type (Vec<f32>) and T is the float type (f32).
//! The metric closure receives `&Vec<f32>` for each pair of samples.
//!
//! # Plotting the result
//! ```python
//! import pandas as pd, matplotlib.pyplot as plt
//! df = pd.read_csv("eda_output/tsne_embedding.csv")
//! plt.scatter(df.tsne_x, df.tsne_y, c=df.label.astype(float), cmap="RdYlGn", alpha=0.6)
//! plt.savefig("tsne.png")
//! ```

use anyhow::Result;

pub fn run_tsne(
    latent_csv: &str,
    out_csv: &str,
    perplexity: f32,
    iterations: usize,
) -> Result<()> {
    // ── Load latent vectors ────────────────────────────────────────────────
    let mut rdr = csv::Reader::from_path(latent_csv)?;
    let headers = rdr.headers()?.clone();
    let label_col = headers.len() - 1;
    let latent_dim = label_col; // z0 … z{L-1}

    let mut samples: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    for record in rdr.records() {
        let rec = record?;
        let sample: Vec<f32> = (0..latent_dim)
            .map(|i| rec[i].trim().parse::<f32>().unwrap_or(0.0))
            .collect();
        samples.push(sample);
        labels.push(rec[label_col].to_string());
    }

    if samples.is_empty() {
        anyhow::bail!("No data found in {}", latent_csv);
    }

    println!(
        "  Running t-SNE: {} samples × {} dims  perplexity={} iters={}",
        samples.len(),
        latent_dim,
        perplexity,
        iterations
    );

    // ── Run Barnes-Hut t-SNE ───────────────────────────────────────────────
    // bhtsne::tSNE<T=f32, U=Vec<f32>> — metric_f computes Euclidean distance.
    let embedding: Vec<f32> = bhtsne::tSNE::new(&samples)
        .embedding_dim(2)
        .perplexity(perplexity)
        .epochs(iterations)
        .barnes_hut(0.5, |a: &Vec<f32>, b: &Vec<f32>| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .embedding();

    // ── Write output CSV ───────────────────────────────────────────────────
    let mut wtr = csv::Writer::from_path(out_csv)?;
    wtr.write_record(["tsne_x", "tsne_y", "label"])?;

    for (i, label) in labels.iter().enumerate() {
        let x = embedding[i * 2];
        let y = embedding[i * 2 + 1];
        wtr.write_record(&[format!("{:.6}", x), format!("{:.6}", y), label.clone()])?;
    }
    wtr.flush()?;

    Ok(())
}
