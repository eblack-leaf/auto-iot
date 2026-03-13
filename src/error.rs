use thiserror::Error;

#[derive(Debug, Error)]
pub enum AutoIotError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP fetch error for {url}: {source}")]
    Fetch {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Zip extraction error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("Dataset '{0}' not found — run `auto-iot fetch --dataset {0}` first")]
    DatasetMissing(String),

    #[error("Unknown dataset '{0}'. Valid values: ecg5000, nab-taxi, synthetic")]
    UnknownDataset(String),

    #[error("Unknown architecture '{0}'. Valid values: shallow, deep")]
    UnknownArch(String),

    #[error("Model file not found: {0}")]
    ModelMissing(String),

    #[error("Training failed: {0}")]
    Training(String),

    #[error("Empty dataset after loading")]
    EmptyDataset,
}
