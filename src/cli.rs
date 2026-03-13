use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "auto-iot",
    about = "Autoencoder anomaly detection for IoT / edge devices",
    long_about = "Train lightweight autoencoders on time-series and tabular datasets.\n\
                  Supports grid search over hyperparameters, model saving, and EDA.",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Download and cache dataset(s) locally
    Fetch(FetchArgs),
    /// Train an autoencoder (add --grid-search for a full hyperparameter sweep)
    Train(TrainArgs),
    /// Load a saved model and score a test set for anomalies
    Infer(InferArgs),
    /// Exploratory data analysis: statistics, histograms, latent-space export, t-SNE
    Eda(EdaArgs),
}

// ── Fetch ────────────────────────────────────────────────────────────────────

#[derive(Args, Clone, Debug)]
pub struct FetchArgs {
    /// Dataset to download: ecg5000 | nab-taxi | all
    #[arg(short, long, default_value = "all")]
    pub dataset: String,

    /// Directory to save raw files
    #[arg(short, long, default_value = "data")]
    pub output_dir: String,
}

// ── Train ────────────────────────────────────────────────────────────────────

#[derive(Args, Clone, Debug)]
pub struct TrainArgs {
    /// Dataset: ecg5000 | nab-taxi | synthetic
    #[arg(short, long, default_value = "ecg5000")]
    pub dataset: String,

    /// Architecture: shallow | deep
    #[arg(short, long, default_value = "shallow")]
    pub arch: String,

    /// Run the full hyperparameter grid (overrides per-flag values)
    #[arg(long)]
    pub grid_search: bool,

    /// Maximum training epochs per run
    #[arg(long, default_value = "100")]
    pub epochs: usize,

    /// Latent-space dimension
    #[arg(long, default_value = "8")]
    pub latent_dim: usize,

    /// Hidden-layer width (first hidden layer; deep arch adds a second at half this)
    #[arg(long, default_value = "64")]
    pub hidden_dim: usize,

    /// Adam learning rate
    #[arg(long, default_value = "0.001")]
    pub lr: f64,

    /// Mini-batch size
    #[arg(long, default_value = "64")]
    pub batch_size: usize,

    /// Early-stopping patience (epochs without validation improvement)
    #[arg(long, default_value = "3")]
    pub patience: usize,

    /// Where to write model checkpoints and the results CSV
    #[arg(long, default_value = "artifacts")]
    pub artifact_dir: String,

    /// Where downloaded data lives
    #[arg(long, default_value = "data")]
    pub data_dir: String,

    /// Sliding-window length for NAB / synthetic datasets
    #[arg(long, default_value = "64")]
    pub window: usize,

    /// Train only on samples labelled normal (label=0). Recommended for anomaly detection.
    #[arg(long)]
    pub clean_train: bool,

    /// Fraction of the training file held out for validation (0.0–1.0).
    #[arg(long, default_value = "0.1")]
    pub val_split: f32,
}

// ── Infer ────────────────────────────────────────────────────────────────────

#[derive(Args, Clone, Debug)]
pub struct InferArgs {
    /// Path to the saved model checkpoint (.mpk)
    #[arg(short, long)]
    pub model: String,

    /// Dataset to score: ecg5000 | nab-taxi | synthetic
    #[arg(short, long, default_value = "ecg5000")]
    pub dataset: String,

    /// Architecture the checkpoint was trained with: shallow | deep
    #[arg(short, long, default_value = "shallow")]
    pub arch: String,

    /// Hidden-dim used at training time (needed to rebuild the model)
    #[arg(long, default_value = "64")]
    pub hidden_dim: usize,

    /// Latent-dim used at training time
    #[arg(long, default_value = "8")]
    pub latent_dim: usize,

    /// Data directory
    #[arg(long, default_value = "data")]
    pub data_dir: String,

    /// Anomaly threshold: percentile of training reconstruction errors (0–100)
    #[arg(long, default_value = "95")]
    pub threshold_pct: f64,

    /// Window size (must match training)
    #[arg(long, default_value = "64")]
    pub window: usize,
}

// ── EDA ──────────────────────────────────────────────────────────────────────

#[derive(Args, Clone, Debug)]
pub struct EdaArgs {
    /// Dataset to inspect: ecg5000 | nab-taxi | synthetic
    #[arg(short, long, default_value = "ecg5000")]
    pub dataset: String,

    /// Data directory
    #[arg(long, default_value = "data")]
    pub data_dir: String,

    /// Path to saved model checkpoint (enables latent-space analysis)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Architecture to reconstruct (required when --model is supplied)
    #[arg(short, long, default_value = "shallow")]
    pub arch: String,

    /// Hidden-dim to reconstruct (required when --model is supplied)
    #[arg(long, default_value = "64")]
    pub hidden_dim: usize,

    /// Latent-dim to reconstruct
    #[arg(long, default_value = "8")]
    pub latent_dim: usize,

    /// Run Barnes-Hut t-SNE on the latent vectors (slow on large datasets)
    #[arg(long)]
    pub tsne: bool,

    /// Directory for CSV exports
    #[arg(short, long, default_value = "eda_output")]
    pub output_dir: String,

    /// Window size (must match training for NAB / synthetic)
    #[arg(long, default_value = "64")]
    pub window: usize,
}
