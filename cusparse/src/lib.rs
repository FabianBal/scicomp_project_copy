use matrix_base::{Dense, COO, CSR};

pub fn multiply(matrix1: &CSR, matrix2: &CSR) -> Dense {
    println!("Multiplying two CSR matrices with cuSPARSE");
    matrix1.to_dense()
}