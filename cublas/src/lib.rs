use matrix_base::Dense;

pub fn multiply(matrix1: &Dense, matrix2: &Dense) -> Dense {
    println!("Multiplying two Dense matrices with cuBLAS");
    Dense::new_zeros((matrix1.shape.0, matrix2.shape.1))
}