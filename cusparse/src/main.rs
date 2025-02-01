use std::ffi::c_void;
use std::ptr;

#[allow(warnings)]
mod bindings; // Generated bindings
use bindings::*;

use bindings::cusparseHandle_t;

use cust::memory::*;
// use cust::context::*;
use cust::error::CudaResult;

// Define the sparse matrix in CSR format
const ROWS: i64 = 3;
const COLS: i64 = 3;
const NNZ: i64 = 4; // Number of non-zero elements

fn main() -> CudaResult<()>{
    // Example of using a CUDA function via bindings
    // unsafe {
    //     // Call CUDA functions directly via FFI (Foreign Function Interface)
    //     let mut cusparse_handle: cusparseHandle_t = ptr::null_mut();
    //     let result = bindings::cusparseCreate(&mut cusparse_handle);
    //     if result != 0 {
    //         eprintln!("CUDA function call failed!");
    //     }
    // }
    // Initialize CUDA
    let _ctx = cust::quick_init()?;

    // Create cuSPARSE handle
    let mut cusparse_handle: cusparseHandle_t = ptr::null_mut();
    unsafe {
        bindings::cusparseCreate(&mut cusparse_handle);
    }

    // Define sparse matrix in CSR format
    let h_csr_row_ptr: [i32; 4] = [0, 1, 3, 4];
    let h_csr_col_ind: [i32; 4] = [0, 1, 2, 2];
    let h_csr_values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    // Define dense matrix
    let h_dense_matrix: [f32; 9] = [1.0, 2.0, 3.0,
                                    4.0, 5.0, 6.0,
                                    7.0, 8.0, 9.0];

    // Allocate device memory
    let d_csr_row_ptr = DeviceBuffer::from_slice(&h_csr_row_ptr)?;
    let d_csr_col_ind = DeviceBuffer::from_slice(&h_csr_col_ind)?;
    let d_csr_values = DeviceBuffer::from_slice(&h_csr_values)?;
    let d_dense_matrix = DeviceBuffer::from_slice(&h_dense_matrix)?;
    let mut d_result_matrix = DeviceBuffer::zeroed(ROWS as usize * COLS as usize)?;

    // Create cuSPARSE matrix descriptors
    let mut sparse_mat: cusparseSpMatDescr_t = ptr::null_mut();
    unsafe {
        cusparseCreateCsr(
            &mut sparse_mat,
            ROWS,
            COLS,
            NNZ,
            d_csr_row_ptr.as_device_ptr().as_mut_ptr() as *mut c_void,
            d_csr_col_ind.as_device_ptr().as_mut_ptr() as *mut c_void,
            d_csr_values.as_device_ptr().as_mut_ptr() as *mut c_void,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexBase_t_CUSPARSE_INDEX_BASE_ZERO,
            cudaDataType_t_CUDA_R_32F,
        );
    }

    let mut dense_mat: cusparseDnMatDescr_t = ptr::null_mut();
    unsafe {
        cusparseCreateDnMat(
            &mut dense_mat,
            ROWS,
            COLS,
            COLS, // Leading dimension
            d_dense_matrix.as_device_ptr().as_mut_ptr() as *mut c_void,
            cudaDataType_t_CUDA_R_32F,
            cusparseOrder_t_CUSPARSE_ORDER_ROW,
        );
    }

    let mut result_mat: cusparseDnMatDescr_t = ptr::null_mut();
    unsafe {
        cusparseCreateDnMat(
            &mut result_mat,
            ROWS,
            COLS,
            COLS,
            d_result_matrix.as_device_ptr().as_mut_ptr() as *mut c_void,
            cudaDataType_t_CUDA_R_32F,
            cusparseOrder_t_CUSPARSE_ORDER_ROW,
        );
    }

    // Perform Sparse Matrix * Dense Matrix (SpMM)
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let mut buffer_size: usize = 0;
    let mut d_buffer: DeviceBuffer<u8>;

    unsafe {
        cusparseSpMM_bufferSize(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat,
            dense_mat,
            &beta as *const f32 as *const _,
            result_mat,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpMMAlg_t_CUSPARSE_SPMM_ALG_DEFAULT,
            &mut buffer_size as *mut usize,
        );

        d_buffer = DeviceBuffer::zeroed(buffer_size)?;

        cusparseSpMM(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat,
            dense_mat,
            &beta as *const f32 as *const _,
            result_mat,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpMMAlg_t_CUSPARSE_SPMM_ALG_DEFAULT,
            d_buffer.as_device_ptr().as_mut_ptr() as *mut c_void,
        );
    }

    // Copy result back
    let mut h_result = vec![0.0f32; (ROWS * COLS) as usize];
    d_result_matrix.copy_to(&mut h_result)?;

    println!("Result Matrix: {:?}", h_result);

    // Clean up
    unsafe {
        cusparseDestroySpMat(sparse_mat);
        cusparseDestroyDnMat(dense_mat);
        cusparseDestroyDnMat(result_mat);
        cusparseDestroy(cusparse_handle);
    }

    Ok(())
}