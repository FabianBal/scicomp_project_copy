use matrix_base::Dense;
use rayon::prelude::*;

pub trait DenseProd {
    fn product_dense_par(&self, other: &Dense) -> Dense;
}

impl DenseProd for Dense {
    fn product_dense_par(&self, other: &Dense) -> Dense {
        let m = self.shape.0;
        let n = other.shape.1;
        let p = self.shape.1;
        assert_eq!(
            p, other.shape.0,
            "Matrix dimensions do not match for multiplication"
        );

        let result: Vec<Vec<f64>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; n];
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..p {
                        sum += self.get(i, k) * other.get(k, j);
                    }
                    row[j] = sum;
                }
                row
            })
            .collect();

        let mut final_result = Dense::new_zeros((m, n));
        for (i, row) in result.into_iter().enumerate() {
            for (j, value) in row.into_iter().enumerate() {
                final_result.set(i, j, value);
            }
        }

        final_result
    }
}
