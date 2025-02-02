use std::{env, fs, path::{Path, PathBuf}, vec};
use matrix_base::{COO, CSR, Dense};
use fakscpu::sparse::SparseProd;

/// Benchmark matrix multiplication using different libraries
/// load all matrices from provided folder path or default and benchmark all possible combinations
fn main() {
    // Default values
    let mut repeat_count: usize = 10;
    let mut folder_path = "./matrix_instances";

    // Get command-line arguments if provided
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        repeat_count = args[1].parse().expect("Failed to parse repeat count");
    }
    if args.len() >= 3 {
        folder_path = &args[2];
    }

    // search matrices in the folder
    let matrix_paths = get_matrix_paths(folder_path);
    println!("Found {} matrices in '{}'", matrix_paths.len(), folder_path);
    
    // Generate table headers
    println!("\nTotal Times:");
    println!("{:<20} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}", 
    "Matrix 1", "Matrix 2", "cuBlas (µs)", "cuSparse (µs)", "gpuDense (µs)", "gpuSparse (µs)", "Blas (µs)", "cpuDense (µs)", "cpuSparse (µs)", "cpuSparseParallel (µs)");

    let mut buffer_table = String::from("Buffer Times:\n");
    buffer_table += &format!("{:<20} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}", 
    "Matrix 1", "Matrix 2", "cuBlas (µs)", "cuSparse (µs)", "gpuDense (µs)", "gpuSparse (µs)", "Blas (µs)", "cpuDense (µs)", "cpuSparse (µs)", "cpuSparseParallel (µs)");

    let mut multiplication_table = String::from("Raw Multiplication Times:\n");
    multiplication_table += &format!("{:<20} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}", 
    "Matrix 1", "Matrix 2", "cuBlas (µs)", "cuSparse (µs)", "gpuDense (µs)", "gpuSparse (µs)", "Blas (µs)", "cpuDense (µs)", "cpuSparse (µs)", "cpuSparseParallel (µs)");
    
    // Benchmark all possible combinations of matrices
    for matrix1_path in &matrix_paths {
        for matrix2_path in &matrix_paths {
            // Make sure that the matrices are compatible
            if COO::read_mtx(&matrix1_path, false).unwrap().shape.1 == COO::read_mtx(&matrix2_path, false).unwrap().shape.0 {
                let matrix1_name = matrix1_path.file_name().unwrap().to_str().unwrap();
                let matrix2_name = matrix2_path.file_name().unwrap().to_str().unwrap();
                // println!("\nBenchmarking {} X {}", matrix1_name, matrix2_name);
                let avg_times = benchmark_matrix(matrix1_path, matrix2_path, repeat_count);

                // generate table rows
                buffer_table += &format!("\n{:<20} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}", 
                         matrix1_name, matrix2_name, avg_times[0].0, avg_times[1].0, avg_times[2].0, avg_times[3].0, avg_times[4].0, avg_times[5].0, avg_times[6].0, avg_times[7].0);
                multiplication_table += &format!("\n{:<20} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}", 
                         matrix1_name, matrix2_name, avg_times[0].1, avg_times[1].1, avg_times[2].1, avg_times[3].1, avg_times[4].1, avg_times[5].1, avg_times[6].1, avg_times[7].1);
                println!("{:<20} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}", 
                matrix1_name, matrix2_name, avg_times[0].2, avg_times[1].2, avg_times[2].2, avg_times[3].2, avg_times[4].2, avg_times[5].2, avg_times[6].2, avg_times[7].2);
            }
        }
    }
    println!("\n\n{}\n\n{}", buffer_table, multiplication_table);
}

fn import_matrix(matrix_path: &Path) -> (Dense, CSR) {
    let matrix_coo = COO::read_mtx(matrix_path, false).expect(format!("failed reading matrix at {}", matrix_path.display()).as_str());
    let matrix_dense = matrix_coo.to_dense();
    let matrix_csr = CSR::from_coo(matrix_coo);
    (matrix_dense, matrix_csr)
}

// Benchmark matrix multiplication
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path, repeat_count: usize) -> Vec<(u128, u128, u128)> {
    let (matrix1_dense, matrix1_csr) = import_matrix(matrix1_path);
    let (matrix2_dense, matrix2_csr) = import_matrix(matrix2_path);

    // save times for each library in format (buffer_time, multiply_time, total_time)
    let mut times_cpu_dense = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse_parallel = Vec::with_capacity(repeat_count);
    let mut times_cublas = Vec::with_capacity(repeat_count);
    let mut times_cusparse = Vec::with_capacity(repeat_count);
    let mut times_gpu_dense = Vec::with_capacity(repeat_count);
    let mut times_gpu_sparse = Vec::with_capacity(repeat_count);
    let mut times_blas = Vec::with_capacity(repeat_count);

    // run benchmark for each library
    for _ in 1..=repeat_count {
        //cuBLAS
        let start = std::time::Instant::now();
        let (_matrix, time_raw_multiply) = cublas::multiply(&matrix1_dense, &matrix2_dense).unwrap();
        let time_total = start.elapsed().as_micros();
        times_cublas.push((time_raw_multiply, time_total - time_raw_multiply, time_total));
        
        //cuSPARSE
        let start = std::time::Instant::now();
        let (_matrix, time_raw_multiply) = cusparse::multiply(&matrix1_csr, &matrix2_csr).unwrap();
        let time_total = start.elapsed().as_micros();
        times_cusparse.push((time_raw_multiply, time_total - time_raw_multiply, time_total));

        //GPU Dense
        let start = std::time::Instant::now();
        let (_matrix, time_raw_multiply) = (&matrix1_dense, 0);//ToDo: call GPU Dense multiply
        let time_total = start.elapsed().as_micros();
        times_gpu_dense.push((time_raw_multiply, time_total - time_raw_multiply, time_total));

        //GPU Sparse
        let batch_size = 18;//not sure about this value
        let start_total = std::time::Instant::now();
        let mut time_raw_multiply= 0;

        //-----------------------------------------broken part without this it works xD
        let multiply_future = async {
            let mut gpusm = gpu::GPUSparseMultiplyer::new(&matrix1_csr, &matrix2_csr, batch_size).await;
            gpusm.create_and_load_buffer();
            let start_raw_multiply = std::time::Instant::now();
            let _res = gpusm.doit().await;
            time_raw_multiply = start_raw_multiply.elapsed().as_micros();
        };
        tokio::runtime::Runtime::new().unwrap().block_on(multiply_future);
        //-----------------------------------------end broken part
        let time_total = start_total.elapsed().as_micros();
        times_gpu_sparse.push((time_raw_multiply, time_total - time_raw_multiply, time_total));

        //BLAS
        let start = std::time::Instant::now();
        //ToDo: call Blas multiply
        let time_total = start.elapsed().as_micros();
        times_blas.push((0, time_total - 0, time_total));

        //CPU Dense
        let start = std::time::Instant::now();
        // matrix1_dense.multiply(&matrix2_dense);
        let time_total = start.elapsed().as_micros();
        times_cpu_dense.push((0, time_total - 0, time_total));

        //CPU Sparse
        let start = std::time::Instant::now();
        matrix1_csr.product_sparse(&matrix2_csr);
        let time_total = start.elapsed().as_micros();
        times_cpu_sparse.push((0, time_total - 0, time_total));

        //CPU Sparse Parallel
        let start = std::time::Instant::now();
        matrix1_csr.product_sparse_par(&matrix2_csr);
        let time_total = start.elapsed().as_micros();
        times_cpu_sparse_parallel.push((0, time_total - 0, time_total));        
    }
    
    // Calculate average times
    let times_vec = vec![times_cublas, times_cusparse, times_gpu_dense, times_gpu_sparse, times_blas, times_cpu_dense, times_cpu_sparse, times_cpu_sparse_parallel];
    let sum_times: Vec<(u128, u128, u128)> = times_vec.into_iter().map(|times| times.iter().fold((0, 0, 0), |acc, time| (acc.0 + time.0, acc.1 + time.1, acc.2 + time.2))).collect();
    let avg_times = sum_times.iter().map(|sum_time| (sum_time.0 / repeat_count as u128, sum_time.1 / repeat_count as u128, sum_time.2 / repeat_count as u128)).collect();
    avg_times
}

// Returns all paths to matrices inside folder_path
fn get_matrix_paths(folder_path: &str) -> Vec<PathBuf> {
    let path = Path::new(folder_path);
    let mut matrix_paths: Vec<PathBuf> = Vec::new();
    if !path.is_dir() {
        eprintln!("Error: '{}' is not a valid directory.", folder_path);
        return matrix_paths;
    }

    // Iterate over directory entries and add .mtx file paths to matrix_paths
    let entries = fs::read_dir(path).expect("Failed to read directory");
    for entry in entries.flatten() {
        let file_path = entry.path();
        if file_path.is_file() && file_path.extension().unwrap_or_default() == "mtx" {
            matrix_paths.push(file_path);
        }
    }
    matrix_paths
}