use matrix_base::CSR;
use std::ffi::c_void;
use std::ptr;
use cust::memory::*;
use cust::error::CudaResult;

#[allow(warnings)]
mod bindings; // Generated bindings
use bindings::*;

/// Multiply two CSR matrices using cuSPARSE
/// helpful: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm/spgemm_example.c

pub fn multiply(matrix1: &CSR, matrix2: &CSR) -> CudaResult<(CSR, u128)> {
    let time_raw_multiply: u128;
    // Ensure the matrices can be multiplied
    assert_eq!(matrix1.shape.1, matrix2.shape.0);

    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Create cuSPARSE handle
    let mut cusparse_handle: cusparseHandle_t = ptr::null_mut();
    unsafe {
        cusparseCreate(&mut cusparse_handle);
    }

    // Matrix dimensions and non-zero counts
    let rows1 = matrix1.shape.0 as i64;
    let cols1 = matrix1.shape.1 as i64;
    let rows2 = matrix2.shape.0 as i64;
    let cols2 = matrix2.shape.1 as i64;
    let nnz1 = matrix1.values.len() as i64;
    let nnz2 = matrix2.values.len() as i64;
    let mut rows_result: i64 = rows1;
    let mut cols_result: i64 = 0;
    let mut nnz_result: i64 = 0;

    // Copy matrix data to device
    let d_csr_row_ptr1 = DeviceBuffer::from_slice(&matrix1.row_pos.iter().map(|value| *value as i32).collect::<Vec<i32>>())?;
    let d_csr_col_ind1 = DeviceBuffer::from_slice(&matrix1.col_pos.iter().map(|value| *value as i32).collect::<Vec<i32>>())?;
    let d_csr_values1 = DeviceBuffer::from_slice(&matrix1.values.iter().map(|value| *value as f32).collect::<Vec<f32>>())?;

    let d_csr_row_ptr2 = DeviceBuffer::from_slice(&matrix2.row_pos.iter().map(|value| *value as i32).collect::<Vec<i32>>())?;
    let d_csr_col_ind2 = DeviceBuffer::from_slice(&matrix2.col_pos.iter().map(|value| *value as i32).collect::<Vec<i32>>())?;
    let d_csr_values2 = DeviceBuffer::from_slice(&matrix2.values.iter().map(|value| *value as f32).collect::<Vec<f32>>())?;
    
    // Allocate memory for result matrix row pointers
    let d_result_row_ptr: DeviceBuffer<i32> = DeviceBuffer::zeroed((rows_result + 1) as usize)?;
    let d_result_col_ind: DeviceBuffer<i32>;
    let d_result_values: DeviceBuffer<f32>;
    
    // Create cuSPARSE matrix descriptors
    let mut sparse_mat1: cusparseSpMatDescr_t = ptr::null_mut();
    let mut sparse_mat2: cusparseSpMatDescr_t = ptr::null_mut();
    let mut sparse_result: cusparseSpMatDescr_t = ptr::null_mut();
    
    unsafe {
        // Create CSR matrix descriptor for matrix1
        cusparseCreateCsr(
            &mut sparse_mat1,
            rows1,
            cols1,
            nnz1,
            d_csr_row_ptr1.as_device_ptr().as_mut_ptr() as *mut c_void,
            d_csr_col_ind1.as_device_ptr().as_mut_ptr() as *mut c_void,
            d_csr_values1.as_device_ptr().as_mut_ptr() as *mut c_void,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexBase_t_CUSPARSE_INDEX_BASE_ZERO,
            cudaDataType_t_CUDA_R_32F,
        );
        
        // Create CSR matrix descriptor for matrix2
        cusparseCreateCsr(
            &mut sparse_mat2,
            rows2,
            cols2,
            nnz2,
            d_csr_row_ptr2.as_device_ptr().as_mut_ptr() as *mut c_void,
            d_csr_col_ind2.as_device_ptr().as_mut_ptr() as *mut c_void,
            d_csr_values2.as_device_ptr().as_mut_ptr() as *mut c_void,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexBase_t_CUSPARSE_INDEX_BASE_ZERO,
            cudaDataType_t_CUDA_R_32F,
        );
        
        // Create CSR matrix descriptor for result matrix
        cusparseCreateCsr(
            &mut sparse_result,
            rows1,
            cols2,
            0,
            d_result_row_ptr.as_device_ptr().as_mut_ptr() as *mut c_void,
            ptr::null_mut(),
            ptr::null_mut(),
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexType_t_CUSPARSE_INDEX_32I,
            cusparseIndexBase_t_CUSPARSE_INDEX_BASE_ZERO,
            cudaDataType_t_CUDA_R_32F,
        );
    }
    
    // Scalars for matrix multiplication C = alpha * A * B + beta * C
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let mut buffer_size1: usize = 0;
    let mut buffer_size2: usize = 0;
    let d_buffer1: DeviceBuffer<u8>;
    let d_buffer2: DeviceBuffer<u8>;

    unsafe {
        // Create SpGEMM descriptor
        let mut spgemm_descr: cusparseSpGEMMDescr_t = ptr::null_mut();
        cusparseSpGEMM_createDescr(&mut spgemm_descr);
        
        // Estimate buffer size
        cusparseSpGEMM_workEstimation(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat1,
            sparse_mat2,
            &beta as *const f32 as *const _,
            sparse_result,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpGEMMAlg_t_CUSPARSE_SPGEMM_DEFAULT,
            spgemm_descr,
            &mut buffer_size1 as *mut usize,
            ptr::null_mut(),
        );
        
        // Allocate buffer for work estimation
        d_buffer1 = DeviceBuffer::zeroed(buffer_size1)?;

        // compute an upper bound for the intermediate products
        cusparseSpGEMM_workEstimation(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat1,
            sparse_mat2,
            &beta as *const f32 as *const _,
            sparse_result,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpGEMMAlg_t_CUSPARSE_SPGEMM_DEFAULT,
            spgemm_descr,
            &mut buffer_size1,
            d_buffer1.as_device_ptr().as_mut_ptr() as *mut c_void,
        );

        // Estimate buffer size for SpGEMM compute
        cusparseSpGEMM_compute(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat1,
            sparse_mat2,
            &beta as *const f32 as *const _,
            sparse_result,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpGEMMAlg_t_CUSPARSE_SPGEMM_DEFAULT,
            spgemm_descr,
            &mut buffer_size2,
            ptr::null_mut(),
        );

        // Allocate buffer for SpGEMM compute
        d_buffer2 = DeviceBuffer::zeroed(buffer_size2)?;

        let start = std::time::Instant::now();
        // computes the structure of the output matrix and its values. It stores the matrix in temporary buffers
        cusparseSpGEMM_compute(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat1,
            sparse_mat2,
            &beta as *const f32 as *const _,
            sparse_result,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpGEMMAlg_t_CUSPARSE_SPGEMM_DEFAULT,
            spgemm_descr,
            &mut buffer_size2,
            d_buffer2.as_device_ptr().as_mut_ptr() as *mut c_void,
        );
        time_raw_multiply = start.elapsed().as_micros();

        // Get the size of the result matrix
        cusparseSpMatGetSize(sparse_result, &mut rows_result, &mut cols_result, &mut nnz_result);
        d_result_col_ind = DeviceBuffer::zeroed((nnz_result) as usize)?;
        d_result_values = DeviceBuffer::zeroed((nnz_result) as usize)?;
        
        // Set the pointers for the result matrix
        cusparseCsrSetPointers(sparse_result, d_result_row_ptr.as_device_ptr().as_mut_ptr() as *mut c_void, d_result_col_ind.as_device_ptr().as_mut_ptr() as *mut c_void, d_result_values.as_device_ptr().as_mut_ptr() as *mut c_void);
        
        // Copies the offsets, column indices, and values from the temporary buffers to the output matrix
        cusparseSpGEMM_copy(
            cusparse_handle,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparseOperation_t_CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            sparse_mat1,
            sparse_mat2,
            &beta as *const f32 as *const _,
            sparse_result,
            cudaDataType_t_CUDA_R_32F,
            cusparseSpGEMMAlg_t_CUSPARSE_SPGEMM_DEFAULT,
            spgemm_descr
        );
    }

    // Copy result data from device to host
    let mut h_result_row_ptr = vec![0i32; (rows_result + 1) as usize];
    let mut h_result_col_ind = vec![0i32; (nnz_result) as usize];
    let mut h_result_values = vec![0.0f32; (nnz_result) as usize];
    
    d_result_row_ptr.copy_to(&mut h_result_row_ptr)?;
    d_result_col_ind.copy_to(&mut h_result_col_ind)?;
    d_result_values.copy_to(&mut h_result_values)?;

    // Truncate the result vectors to the correct size not sure if this is necessary
    h_result_col_ind.truncate(nnz_result as usize);
    h_result_values.truncate(nnz_result as usize);
    
    // Destroy cuSPARSE matrix descriptors and handle
    unsafe {
        cusparseDestroySpMat(sparse_mat1);
        cusparseDestroySpMat(sparse_mat2);
        cusparseDestroySpMat(sparse_result);
        cusparseDestroy(cusparse_handle);
    }
    
    // Return the result matrix in CSR format
    Ok((CSR {
        row_pos: h_result_row_ptr.into_iter().map(|x| x as usize).chain(std::iter::once(nnz_result as usize)).collect(),
        col_pos: h_result_col_ind.into_iter().map(|x| x as usize).collect(),
        values: h_result_values.iter().map(|&x| x as f64).collect(),
        shape: (rows1 as usize, cols2 as usize),
    }, time_raw_multiply))
}