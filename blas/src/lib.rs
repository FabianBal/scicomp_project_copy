extern crate blas;
extern crate openblas_src;

use blas::*;
use matrix_base::{Dense, COO};

pub struct BlasDense {
    pub data: Vec<f64>,    // matrix [[1,2],[3,4]] is stored as [1,3,2,4]
    pub shape: (i32, i32), // i32 is needed for dgemm
}

impl BlasDense {
    // create new BlasDense matrix with given data (flattened collumn major oder) and shape
    pub fn new_with_data(data: Vec<f64>, shape: (i32, i32)) -> Self {
        BlasDense { data, shape }
    }

    // create BlasDense matrix from COO format
    pub fn from_coo(matrix: &COO) -> Self {
        let mut mat = Dense::new_zeros((matrix.shape.1, matrix.shape.0));
        for (i, j, x) in &matrix.data {
            mat.set(*j, *i, *x);
        }
        let shape = (matrix.shape.0 as i32, matrix.shape.1 as i32);
        BlasDense {
            data: mat.data,
            shape,
        }
    }

    // compute product
    pub fn prod(&self, other: &BlasDense) -> BlasDense {
        assert_eq!(self.shape.1, other.shape.0, "matrix dimension mismatch");
        let (m, n, k) = (self.shape.0, other.shape.1, self.shape.1); // m - rowa of A (and product), n - collumns of B (and prodtct, k - collumns of A, rows of B
        let (a, b) = (&self.data, &other.data);
        let mut c = vec![0.; m as usize * n as usize];
        unsafe {
            dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m);
        }
        BlasDense::new_with_data(c, (m, n))
    }
}
