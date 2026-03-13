/// Scale each feature independently to [0, 1].
#[derive(Debug, Clone)]
pub struct MinMaxNormalizer {
    min: Vec<f32>,
    max: Vec<f32>,
}

impl MinMaxNormalizer {
    pub fn fit(data: &[Vec<f32>]) -> Self {
        assert!(!data.is_empty(), "Cannot fit normaliser on empty data");
        let dim = data[0].len();
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for sample in data {
            for (i, &v) in sample.iter().enumerate() {
                if v < min[i] { min[i] = v; }
                if v > max[i] { max[i] = v; }
            }
        }
        MinMaxNormalizer { min, max }
    }

    pub fn transform(&self, sample: &[f32]) -> Vec<f32> {
        sample
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let range = self.max[i] - self.min[i];
                if range.abs() < 1e-8 { 0.0 } else { (v - self.min[i]) / range }
            })
            .collect()
    }
}
