// cublas/src/lib.rs

use cublas_sys::{cublasCreate_v2, cublasDestroy_v2, cublasGetStream_v2, cublasHandle_t, cublasSgemm_v2, Struct_CUstream_st};
use cust::error::CudaResult;
use cust::memory::*;
use cust::sys::cuStreamSynchronize;
use matrix_base::Dense;
use std::ptr;


pub fn multiply(matrix1: &Dense, matrix2: &Dense) -> CudaResult<(Dense, u128, u128, u128, u128)> { // NEU
    let time_raw_multiply: u128;
    let time_h2d: u128;
    let time_d2h: u128;
    let time_total: u128;

    // Ensure the matrices can be multiplied
    assert_eq!(matrix1.shape.1, matrix2.shape.0);

    let start_total = std::time::Instant::now(); // Starte die Gesamtzeitmessung frühzeitig

    // Initialize CUDA context (kostet einmalig Zeit pro Prozess, sollte nur 1x passieren)
    cust::init(cust::CudaFlags::empty())?;

    // Create cuBLAS handle
    let mut handle: cublasHandle_t = ptr::null_mut();
    unsafe { cublasCreate_v2(&mut handle) };

    // Convert matrices for cuBLAS (f64 to f32, and RowMajor to ColumnMajor)
    // Diese Konvertierungen auf der CPU sind Teil des Overheads, aber nicht H2D/D2H.
    // Wir können sie zum `initialization_us` in main.rs hinzufügen.
    let matrix1_col_major = matrix1.as_column_major();
    let matrix2_col_major = matrix2.as_column_major();

    let a: Vec<f32> = matrix1_col_major.data.iter().map(|&x| x as f32).collect();
    let b: Vec<f32> = matrix2_col_major.data.iter().map(|&x| x as f32).collect();
    let mut c: Vec<f32> = vec![0.0; matrix1_col_major.shape.0 * matrix2_col_major.shape.1];

    // NEU: H2D Zeitmessung beginnt
    let start_h2d_measure = std::time::Instant::now();
    // Allocate device memory and copy data from host (H2D)
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_c = DeviceBuffer::from_slice(&c)?; // Ergebnisbuffer auch H2D initialisieren
    time_h2d = start_h2d_measure.elapsed().as_micros(); // NEU: H2D Zeitmessung endet

    // Set cuBLAS parameters
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = matrix1_col_major.shape.0 as i32;
    let k = matrix1_col_major.shape.1 as i32;
    let n = matrix2_col_major.shape.1 as i32;

    // Get cuBLAS stream (optional, but good practice if using async operations)
    let mut stream: *mut Struct_CUstream_st = ptr::null_mut();
    unsafe { cublasGetStream_v2(handle, &mut stream) };

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    let start_raw = std::time::Instant::now(); // NEU: Starte RAW Messung hier
    unsafe {
        cublasSgemm_v2(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // No transpose A
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // No transpose B
            m, // rows of A and C
            n, // cols of B and C
            k, // cols of A and rows of B
            &alpha,
            d_a.as_device_ptr().as_ptr(),
            m, // leading dimension of A (rows of A)
            d_b.as_device_ptr().as_ptr(),
            k, // leading dimension of B (rows of B - since B is ColMajor, it's k)
            &beta,
            d_c.as_device_ptr().as_mut_ptr(),
            m, // leading dimension of C (rows of C)
        );
        // Synchronize stream to wait for the multiplication to finish
        cuStreamSynchronize(stream as *mut cust::sys::CUstream_st);
    }
    time_raw_multiply = start_raw.elapsed().as_micros(); // NEU: Beende RAW Messung hier

    // NEU: D2H Zeitmessung beginnt
    let start_d2h_measure = std::time::Instant::now();
    // Copy result back to host (D2H)
    d_c.copy_to(&mut c)?;
    time_d2h = start_d2h_measure.elapsed().as_micros(); // NEU: D2H Zeitmessung endet

    // Destroy cuBLAS handle
    unsafe { cublasDestroy_v2(handle) };

    time_total = start_total.elapsed().as_micros();

    // Convert result back to Dense (f32 to f64, and ColumnMajor to RowMajor)
    let result_data_f64: Vec<f64> = c.iter().map(|&x| x as f64).collect();
    let result_dense_col_major = Dense {
        data: result_data_f64,
        shape: (matrix1_col_major.shape.0, matrix2_col_major.shape.1),
    };
    let result_dense_row_major = result_dense_col_major.as_row_major();

    // Return the result matrix, raw multiply time, total time, H2D time, D2H time
    Ok((result_dense_row_major, time_raw_multiply, time_total, time_h2d, time_d2h))
}