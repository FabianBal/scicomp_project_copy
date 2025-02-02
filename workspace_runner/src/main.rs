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
    println!("{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
    "Matrix 1", "Matrix 2", "cpuDense (µs)", "cpuSparse (µs)", "cpuSparseParallel (µs)", "cuBLAS (µs)", "cuSPARSE (µs)");

    let mut buffer_table = String::from("Buffer Times:\n");
    buffer_table += &format!("{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
    "Matrix 1", "Matrix 2", "cpuDense (µs)", "cpuSparse (µs)", "cpuSparseParallel (µs)", "cuBLAS (µs)", "cuSPARSE (µs)");

    let mut multiplication_table = String::from("Raw Multiplication Times:\n");
    multiplication_table += &format!("{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
    "Matrix 1", "Matrix 2", "cpuDense (µs)", "cpuSparse (µs)", "cpuSparseParallel (µs)", "cuBLAS (µs)", "cuSPARSE (µs)");
    
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
                buffer_table += &format!("\n{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
                         matrix1_name, matrix2_name, avg_times[0].0, avg_times[1].0, avg_times[2].0, avg_times[3].0, avg_times[4].0);
                multiplication_table += &format!("\n{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
                         matrix1_name, matrix2_name, avg_times[0].1, avg_times[1].1, avg_times[2].1, avg_times[3].1, avg_times[4].1);
                println!("{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
                matrix1_name, matrix2_name, avg_times[0].2, avg_times[1].2, avg_times[2].2, avg_times[3].2, avg_times[4].2);
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
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path, repeat_count: usize) -> Vec<(f64, f64, f64)> {
    let (matrix1_dense, matrix1_csr) = import_matrix(matrix1_path);
    let (matrix2_dense, matrix2_csr) = import_matrix(matrix2_path);

    // save times for each library in format (buffer_time, multiply_time, total_time)
    let mut times_cpu_dense = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse_parallel = Vec::with_capacity(repeat_count);
    let mut times_cublas = Vec::with_capacity(repeat_count);
    let mut times_cusparse = Vec::with_capacity(repeat_count);

    // run benchmark for each library
    for _ in 1..=repeat_count {
        let start = std::time::Instant::now();
        // matrix1_dense.multiply(&matrix2_dense);
        let time = start.elapsed().as_micros();
        times_cpu_dense.push((0, time, time));

        let start = std::time::Instant::now();
        matrix1_csr.product_sparse(&matrix2_csr);
        let time = start.elapsed().as_micros();
        times_cpu_sparse.push((0, time, time));

        let start = std::time::Instant::now();
        matrix1_csr.product_sparse_par(&matrix2_csr);
        let time = start.elapsed().as_micros();
        times_cpu_sparse_parallel.push((0, time, time));

        let start = std::time::Instant::now();
        let _ = cublas::multiply(&matrix1_dense, &matrix2_dense);
        let time_buffer = 0;//ToDo: get buffer time
        let time_multiply = start.elapsed().as_micros();
        times_cublas.push((time_buffer, time_multiply, time_buffer + time_multiply));
        
        let start = std::time::Instant::now();
        let _ = cusparse::multiply(&matrix1_csr, &matrix2_csr);
        let time_buffer = 0;//ToDo: get buffer time
        let time_multiply = start.elapsed().as_micros();
        times_cusparse.push((time_buffer, time_multiply, time_buffer + time_multiply));
        
    }
    
    // Calculate average times
    let times_vec = vec![times_cpu_dense, times_cpu_sparse, times_cpu_sparse_parallel, times_cublas, times_cusparse];
    let sum_times: Vec<(u128, u128, u128)> = times_vec.into_iter().map(|times| times.iter().fold((0, 0, 0), |acc, time| (acc.0 + time.0, acc.1 + time.1, acc.2 + time.2))).collect();
    let avg_times = sum_times.iter().map(|sum_time| (sum_time.0 as f64 / repeat_count as f64, sum_time.1 as f64 / repeat_count as f64, sum_time.2 as f64 / repeat_count as f64)).collect();
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