// use cust::device::Device;
use cust::memory::*;
// use cust::context::*;
use cust::error::CudaResult;
use std::ptr;

use cublas_sys::{cublasCreate_v2, cublasDestroy_v2, cublasSgemm_v2, cublasHandle_t};

// Matrix size (NxN)
const N: i32 = 2;

fn main() -> CudaResult<()> {
    // Initialize CUDA context
    // let device = Device::get_device(0)?;
    // let _ctx = Context::new(device)?;
    cust::init(cust::CudaFlags::empty())?;
    // let device = Device::get_device(0)?;
    // let context = Context::new(device)?;
    // let _version = context.get_api_version()?;


    // Create cuBLAS handle
    let mut handle: cublasHandle_t = ptr::null_mut();
    unsafe { cublasCreate_v2(&mut handle) };

    // Define matrices A, B, and C
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // Row-major (2x2)
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0]; // Row-major (2x2)
    let mut c: Vec<f32> = vec![0.0; 4];         // Result matrix (2x2)

    // Allocate device memory
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_c = DeviceBuffer::from_slice(&c)?;

    // Set cuBLAS parameters
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    unsafe {
        cublasSgemm_v2(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // No transpose A
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // No transpose B
            N, N, N,                                    // Matrix dimensions
            &alpha, 
            d_a.as_device_ptr().as_ptr(), N, 
            d_b.as_device_ptr().as_ptr(), N, 
            &beta, 
            d_c.as_device_ptr().as_mut_ptr(), N,
        );
    }

    // Copy result back to host
    d_c.copy_to(&mut c)?;

    // Destroy cuBLAS handle
    unsafe { cublasDestroy_v2(handle) };

    // Print the result
    println!("Result matrix C: {:?}", c);

    Ok(())
}
