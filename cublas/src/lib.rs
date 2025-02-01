use matrix_base::Dense;
use cust::memory::*;
use cust::error::CudaResult;
use std::ptr;
use cublas_sys::{cublasCreate_v2, cublasDestroy_v2, cublasSgemm_v2, cublasHandle_t};

pub fn multiply(matrix1: &Dense, matrix2: &Dense) -> CudaResult<Dense> {
    // Ensure the matrices can be multiplied
    assert_eq!(matrix1.shape.1, matrix2.shape.0);

    // Initialize CUDA context
    cust::init(cust::CudaFlags::empty())?;

    // Create cuBLAS handle
    let mut handle: cublasHandle_t = ptr::null_mut();
    unsafe { cublasCreate_v2(&mut handle) };

    // Flatten the matrices
    let a: Vec<f32> = matrix1.data.iter().map(|&x| x as f32).collect();
    let b: Vec<f32> = matrix2.data.iter().map(|&x| x as f32).collect();
    let mut c: Vec<f32> = vec![0.0; matrix1.shape.0 * matrix2.shape.1];

    // Allocate device memory
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_c = DeviceBuffer::from_slice(&c)?;

    // Set cuBLAS parameters
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = matrix1.shape.0 as i32;
    let k = matrix1.shape.1 as i32;
    let n = matrix2.shape.1 as i32;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    unsafe {
        cublasSgemm_v2(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // No transpose A
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // No transpose B
            m, n, k,                                    // Matrix dimensions not sure which one is which
            &alpha, 
            d_a.as_device_ptr().as_ptr(), m, 
            d_b.as_device_ptr().as_ptr(), k, 
            &beta, 
            d_c.as_device_ptr().as_mut_ptr(), m,
        );
    }

    // Copy result back to host
    d_c.copy_to(&mut c)?;

    // Destroy cuBLAS handle
    unsafe { cublasDestroy_v2(handle) };

    // Convert result back to Dense
    let result_data: Vec<f64> = c.iter().map(|&x| x as f64).collect();
    let result = Dense {
        data: result_data,
        shape: (matrix1.shape.0, matrix2.shape.1),
    };

    Ok(result)
}