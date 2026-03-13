# auto-iot

Lightweight autoencoder-based anomaly detection for IoT and edge devices.
Part of a suite of local-training ML projects targeting **smallest intelligence that gets the job done**.

Built with **Burn 0.20.1** + **Rust** + **WGPU** (GPU-accelerated, no CUDA required).

---

## Project goals

- Train autoencoders on real sensor / time-series datasets entirely on your local machine
- Find the smallest model that achieves good anomaly detection via grid search
- Save and reload models for repeatable inference
- Provide EDA tooling to understand training data and the learned latent space

---

## Architectures

| Name     | Encoder                         | Decoder                         | When to use              |
|----------|---------------------------------|---------------------------------|--------------------------|
| Shallow  | input → hidden → latent         | latent → hidden → output        | Fastest, fewest params   |
| Deep     | input → H → H/2 → latent        | latent → H/2 → H → output       | Better capacity, slower  |

Activation: **ReLU** in hidden layers, **Sigmoid** on the output (matches MinMax-normalised inputs).

---

## Datasets

All labels use the same convention: `0 = normal, 1 = anomaly`.

| Name       | Domain              | Seq len             | Labels | Source |
|------------|---------------------|---------------------|--------|--------|
| `nab-machine` | IoT machine temperature | configurable window | Yes    | [Numenta NAB (GitHub)](https://github.com/numenta/NAB) |
| `nab-taxi` | NYC taxi passengers | configurable window | Yes    | [Numenta NAB (GitHub)](https://github.com/numenta/NAB) |
| `synthetic`| Gaussian + outliers | configurable window | Yes    | Generated in-memory |

---

## Quickstart

### 1. Prerequisites

- Rust stable (≥ 1.75): `curl https://sh.rustup.rs -sSf | sh`
- A GPU with Vulkan / Metal / DX12 support (WGPU). Falls back to software rendering.

### 2. Clone and build

```bash
git clone <repo-url> auto-iot
cd auto-iot
cargo build --release
```

### 3. Fetch datasets

```bash
cargo run --release -- fetch --dataset all

# Or individually
cargo run --release -- fetch --dataset nab-machine
cargo run --release -- fetch --dataset nab-taxi
```

### 4. Train a single model

```bash
cargo run --release -- train \
  --dataset nab-machine \
  --arch shallow \
  --epochs 100 \
  --latent-dim 8 \
  --hidden-dim 64 \
  --lr 0.001 \
  --clean-train
```

`--clean-train` filters the training split to normal samples only (`label=0`), which is the
standard approach for reconstruction-based anomaly detection. Without it the model sees
anomalies during training, learns to reconstruct them, and reconstruction error loses its
discriminative power.

The best-validation checkpoint is saved to `artifacts/`.

### 5. Grid search

```bash
# 36 combinations: 2 archs × 2 lrs × 3 latent × 3 hidden
cargo run --release -- train --dataset nab-machine --grid-search --clean-train \
  --epochs 400 --batch-size 512 --patience 5
```

Results are ranked by a **composite score** (AUROC×0.6 + val_loss×0.25 + params×0.15) and saved to `artifacts/grid_results.csv`.

Default grid:

| Axis          | Values        |
|---------------|---------------|
| Architecture  | shallow, deep |
| Learning rate | 1e-3, 5e-4    |
| Latent dim    | 4, 8, 16      |
| Hidden dim    | 32, 64, 128   |

> **Note:** `lr=1e-4` is excluded — on nab-machine it consistently fails to converge within 400 epochs.

> **Latent dim guidance:** for anomaly detection the bottleneck must be tight enough that
> the model generalises the normal manifold rather than memorising everything. A good
> starting point is 3–10% of input dim (latent ∈ {4, 8} for a window-64 input).
> Large latent dims produce low val loss but near-random AUROC — the grid search makes
> this trade-off visible.

### Results — nab-machine (400 epochs, batch 512, patience 5)

**With `--clean-train`** (trains on normal samples only):

| arch | lr | latent | hidden | params | auroc | val_loss | score |
|---|---|---|---|---|---|---|---|
| shallow | 5e-4 | 16 | 32 | 5 264 | **0.9976** | 0.001071 | **0.9551** |
| shallow | 1e-3 | 16 | 32 | 5 264 | 0.9949 | 0.000543 | 0.9328 |
| shallow | 1e-3 | 8  | 64 | 9 416 | 0.9932 | 0.000720 | 0.8889 |

**Without `--clean-train`** (anomaly prevalence <2% in train — both approaches work):

| arch | lr | latent | hidden | params | auroc | val_loss | score |
|---|---|---|---|---|---|---|---|
| deep    | 5e-4 | 16 | 64 | 13 584 | **0.9995** | 0.000548 | **0.9393** |
| shallow | 1e-3 | 16 | 32 | 5 264  | 0.9968 | 0.000526 | 0.9198 |
| shallow | 5e-4 | 4  | 32 | 4 484  | 0.9952 | 0.000905 | 0.8810 |

**Recommended model for edge deployment:** `shallow, latent=16, hidden=32` (5 264 params, ~21 KB f32) — top composite score with `--clean-train`, converges in ~200–350 epochs.

### 6. Inference

```bash
cargo run --release -- infer \
  --model artifacts/nab-machine_shallow_ld8_hd64_lr1e-3 \
  --dataset nab-machine \
  --arch shallow \
  --hidden-dim 64 \
  --latent-dim 8 \
  --threshold-pct 95
```

Note: omit the `.mpk` extension — Burn appends it automatically.

Prints reconstruction error histogram, AUROC, precision, recall, and F1 (when labels are available).

### 7. Exploratory data analysis

```bash
# Basic stats + histogram
cargo run --release -- eda --dataset nab-machine

# With latent-space export (requires a saved model)
cargo run --release -- eda \
  --dataset nab-machine \
  --model artifacts/nab-machine_shallow_ld8_hd64_lr1e-3 \
  --arch shallow \
  --hidden-dim 64 \
  --latent-dim 8 \
  --output-dir eda_output

# With t-SNE (slow on large datasets)
cargo run --release -- eda \
  --dataset nab-machine \
  --model artifacts/nab-machine_shallow_ld8_hd64_lr1e-3 \
  --arch shallow --tsne
```

EDA outputs:
- Terminal: descriptive statistics, ASCII amplitude histogram
- `eda_output/latent_vectors.csv`: latent coords + labels for all test samples
- `eda_output/tsne_embedding.csv` (if `--tsne`): 2D embedding for plotting

Quick plot of t-SNE:
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("eda_output/tsne_embedding.csv")
plt.scatter(df.tsne_x, df.tsne_y, c=df.label.astype(float), cmap="RdYlGn", alpha=0.6)
plt.title("Latent space t-SNE"); plt.savefig("tsne.png")
```

---

## Parameter size tracking

Every training run prints parameter count and RAM footprint:

```
  shallow │ 18.57K parameters
  RAM footprint (f32): 72.5 KB
```

Grid search results include a `params` column so you can compare model size against AUROC —
the core trade-off for edge deployment.

Approximate footprints (f32, window=64 input):

| arch    | hidden | latent | params | RAM     |
|---------|--------|--------|--------|---------|
| shallow | 32     | 4      | ~9 K   | ~36 KB  |
| shallow | 64     | 8      | ~18 K  | ~72 KB  |
| shallow | 128    | 16     | ~54 K  | ~216 KB |
| deep    | 64     | 8      | ~30 K  | ~120 KB |
| deep    | 128    | 16     | ~95 K  | ~380 KB |

---

## Metrics

| Metric              | Where              | Notes |
|---------------------|--------------------|-------|
| MSE loss            | Train / val        | Per epoch; drives early stopping |
| AUROC               | End of training + infer | Computed on test set after each run; used to rank grid results |
| Precision / Recall / F1 | Infer          | At chosen percentile threshold |
| Parameter count     | Train / grid       | Via Burn's `num_params()` |
| RAM footprint       | Train / grid       | params × 4 bytes (f32) |
| Training time       | Grid results       | Seconds per run |

---

## Project structure

```
src/
  main.rs                  Entry point — WGPU backend init + CLI dispatch
  cli.rs                   Clap argument definitions
  config.rs                Shared config types (TrainConfig, GridConfig, TrainResult)
  datasets/
    mod.rs                 AnomalyDataset trait, fetch/load dispatch
    nab_machine.rs         NAB Machine Temperature download + windowing
    nab_taxi.rs            NAB NYC Taxi download + windowing
    synthetic.rs           In-memory Gaussian + outlier generator
    normalizer.rs          MinMaxNormalizer (fit on train, applied to all splits)
  models/
    mod.rs
    shallow_ae.rs          One-hidden-layer autoencoder
    deep_ae.rs             Two-hidden-layer autoencoder
    metrics.rs             Parameter counting, RAM footprint
  training/
    mod.rs
    trainer.rs             Training loop (Adam + optional cosine LR, early stopping, checkpoint, post-train AUROC)
    early_stopping.rs      Patience-based early stopping
    grid_search.rs         Cartesian-product sweep + composite-score-ranked table + CSV export
    metrics.rs             mse_loss(), reconstruction_errors() — shared with inference
  inference/
    mod.rs
    scorer.rs              Batched reconstruction scoring + latent encoding
    evaluator.rs           compute_auroc(), evaluate() (F1, precision, recall)
  eda/
    mod.rs
    stats.rs               Descriptive statistics
    histogram.rs           ASCII histogram
    latent_export.rs       Encode test set → latent_vectors.csv
    tsne.rs                bhtsne wrapper → tsne_embedding.csv
```

---

## Reproducing results

```bash
# 1. Clone
git clone <repo-url> && cd auto-iot

# 2. Download datasets
cargo run --release -- fetch --dataset all

# 3. Grid search (clean training recommended)
cargo run --release -- train --dataset nab-machine --grid-search --clean-train

# 4. Check artifacts/grid_results.csv — sorted by composite score (auroc×0.6 + val×0.25 + params×0.15)

# 5. Evaluate the best model
cargo run --release -- infer \
  --model <artifact_path from CSV> \
  --dataset nab-machine --arch <arch> \
  --hidden-dim <hd> --latent-dim <ld>

# 6. Inspect the latent space
cargo run --release -- eda --dataset nab-machine \
  --model <artifact_path from CSV> --arch <arch> \
  --hidden-dim <hd> --latent-dim <ld> --tsne
```

All randomness is seeded (`StdRng::seed_from_u64(42)`) for the synthetic dataset.
nab-machine and nab-taxi use deterministic splits. Results are fully reproducible across
machines with the same Burn version.

---

## Dependencies

| Crate              | Version | Purpose |
|--------------------|---------|---------|
| burn               | 0.20.1  | ML framework (WGPU + training) |
| clap               | 4       | CLI argument parsing |
| indicatif          | 0.17    | Progress bars |
| reqwest            | 0.12    | Dataset HTTP downloads (rustls, no OpenSSL) |
| bhtsne             | 0.5     | t-SNE dimensionality reduction |
| csv                | 1       | CSV parsing and export |
| serde / serde_json | 1       | Config serialisation |
| rand / rand_distr  | 0.8     | Synthetic data generation |
| zip / flate2       | 2 / 1   | Archive extraction |
| anyhow / thiserror | 1       | Error handling |

---

## License

MIT
