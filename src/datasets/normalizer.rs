/// Trait for fit-then-transform normalisation.
pub trait Normalizer: Send + Sync {
    fn fit(data: &[Vec<f32>]) -> Self
    where
        Self: Sized;
    fn transform(&self, sample: &[f32]) -> Vec<f32>;
    fn inverse_transform(&self, sample: &[f32]) -> Vec<f32>;
}

/// Scale each feature independently to [0, 1].
#[derive(Debug, Clone)]
pub struct MinMaxNormalizer {
    min: Vec<f32>,
    max: Vec<f32>,
}

impl Normalizer for MinMaxNormalizer {
    fn fit(data: &[Vec<f32>]) -> Self {
        assert!(!data.is_empty(), "Cannot fit normaliser on empty data");
        let dim = data[0].len();
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for sample in data {
            for (i, &v) in sample.iter().enumerate() {
                if v < min[i] {
                    min[i] = v;
                }
                if v > max[i] {
                    max[i] = v;
                }
            }
        }
        MinMaxNormalizer { min, max }
    }

    fn transform(&self, sample: &[f32]) -> Vec<f32> {
        sample
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let range = self.max[i] - self.min[i];
                if range.abs() < 1e-8 {
                    0.0
                } else {
                    (v - self.min[i]) / range
                }
            })
            .collect()
    }

    fn inverse_transform(&self, sample: &[f32]) -> Vec<f32> {
        sample
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (self.max[i] - self.min[i]) + self.min[i])
            .collect()
    }
}

/// Zero-mean, unit-variance normalisation.
#[derive(Debug, Clone)]
pub struct ZScoreNormalizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalizer for ZScoreNormalizer {
    fn fit(data: &[Vec<f32>]) -> Self {
        assert!(!data.is_empty());
        let n = data.len() as f32;
        let dim = data[0].len();

        let mean: Vec<f32> = (0..dim)
            .map(|i| data.iter().map(|s| s[i]).sum::<f32>() / n)
            .collect();

        let std: Vec<f32> = (0..dim)
            .map(|i| {
                let var = data.iter().map(|s| (s[i] - mean[i]).powi(2)).sum::<f32>() / n;
                var.sqrt().max(1e-8)
            })
            .collect();

        ZScoreNormalizer { mean, std }
    }

    fn transform(&self, sample: &[f32]) -> Vec<f32> {
        sample
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.mean[i]) / self.std[i])
            .collect()
    }

    fn inverse_transform(&self, sample: &[f32]) -> Vec<f32> {
        sample
            .iter()
            .enumerate()
            .map(|(i, &v)| v * self.std[i] + self.mean[i])
            .collect()
    }
}
