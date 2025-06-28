///struct zum Speichern der einzelnen zeiten des benchmarks
#[derive(Debug, Clone, Copy)]
pub struct TimingResult {
    pub initialization_us: u128,
    pub h2d_us: u128,
    pub raw_multiply_us: u128,
    pub d2h_us: u128,
    pub cleanup_us: u128,
    pub total_us: u128,
}

impl TimingResult {
    // Helper to create a zero-initialized result
    pub fn zero() -> Self {
        TimingResult {
            initialization_us: 0,
            h2d_us: 0,
            raw_multiply_us: 0,
            d2h_us: 0,
            cleanup_us: 0,
            total_us: 0,
        }
    }
    pub fn max_values() -> Self {
        TimingResult {
            initialization_us: u128::MAX,
            h2d_us: u128::MAX,
            raw_multiply_us: u128::MAX,
            d2h_us: u128::MAX,
            cleanup_us: u128::MAX,
            total_us: u128::MAX,
        }
    }
}
