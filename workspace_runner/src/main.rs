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
        folder_path = &args[1];
    }

    // search matrices in the folder
    let matrix_paths = get_matrix_paths(folder_path);
    println!("Found {} matrices in '{}'", matrix_paths.len(), folder_path);
    
    // Print table header
    println!("{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
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
                println!("{:<20} {:<20} {:<20} {:<20} {:<30} {:<20} {:<20}", 
                         matrix1_name, matrix2_name, avg_times[0], avg_times[1], avg_times[2], avg_times[3], avg_times[4]);
            }
        }
    }
}

fn import_matrix(matrix_path: &Path) -> (Dense, CSR) {
    let matrix_coo = COO::read_mtx(matrix_path, false).expect(format!("failed reading matrix at {}", matrix_path.display()).as_str());
    let matrix_dense = matrix_coo.to_dense();
    let matrix_csr = CSR::from_coo(matrix_coo);
    (matrix_dense, matrix_csr)
}

// Benchmark matrix multiplication
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path, repeat_count: usize) -> Vec<f64>{
    let (matrix1_dense, matrix1_csr) = import_matrix(matrix1_path);
    let (matrix2_dense, matrix2_csr) = import_matrix(matrix2_path);

    // save times for each library
    let mut times_cpu_dense = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse_parallel = Vec::with_capacity(repeat_count);
    let mut times_cublas = Vec::with_capacity(repeat_count);
    let mut times_cusparse = Vec::with_capacity(repeat_count);

    // run benchmark for each library
    for _ in 1..=repeat_count {
        let start = std::time::Instant::now();
        // matrix1_dense.multiply(&matrix2_dense);
        times_cpu_dense.push(start.elapsed().as_micros());

        let start = std::time::Instant::now();
        matrix1_csr.product_sparse(&matrix2_csr);
        times_cpu_sparse.push(start.elapsed().as_micros());

        let start = std::time::Instant::now();
        matrix1_csr.product_sparse_par(&matrix2_csr);
        times_cpu_sparse_parallel.push(start.elapsed().as_micros());

        let start = std::time::Instant::now();
        matrix1_csr.product_sparse(&matrix2_csr);
        times_cpu_sparse.push(start.elapsed().as_micros());

        let start = std::time::Instant::now();
        let _ = cublas::multiply(&matrix1_dense, &matrix2_dense);
        times_cublas.push(start.elapsed().as_micros());
        
        let start = std::time::Instant::now();
        let _ = cusparse::multiply(&matrix1_csr, &matrix2_csr);
        times_cusparse.push(start.elapsed().as_micros());
        
    }
    
    // Calculate average times
    let times = vec![times_cpu_dense, times_cpu_sparse, times_cpu_sparse_parallel, times_cublas, times_cusparse];
    let avg_times: Vec<f64> = times.into_iter().map(|time| time.iter().sum::<u128>() as f64 / repeat_count as f64).collect();
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