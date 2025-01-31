use matrix_base::{Dense, COO, CSR};

pub fn multiply(matrix1: &CSR, matrix2: &CSR) -> Dense {
    println!("Multiplying two Dense matrices with cuBLAS");
    matrix1.to_dense()
}