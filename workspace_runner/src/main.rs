use std::{env, fs, path::{Path, PathBuf}, vec};
use matrix_base::{COO, CSR, Dense};

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
    
    //start benchmarking
    for matrix1_path in &matrix_paths {
        for matrix2_path in &matrix_paths {
            //ToDo: need to make sure that the matrices are compatible
            println!("\nBenchmarking {} X {}", matrix1_path.file_name().unwrap().to_str().unwrap(), matrix2_path.file_name().unwrap().to_str().unwrap());
            benchmark_matrix(matrix1_path, matrix2_path, repeat_count);
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
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path, repeat_count: usize) {
    let (matrix1_dense, matrix1_csr) = import_matrix(matrix1_path);
    let (matrix2_dense, matrix2_csr) = import_matrix(matrix2_path);

    let mut times_cublas = Vec::with_capacity(repeat_count);
    let mut times_cusparse = Vec::with_capacity(repeat_count);
    for _ in 1..=repeat_count {
        let start = std::time::Instant::now();
        cublas::multiply(&matrix1_dense, &matrix2_dense);
        times_cublas.push(start.elapsed().as_nanos());
        
        let start = std::time::Instant::now();
        cusparse::multiply(&matrix1_csr, &matrix2_csr);
        times_cusparse.push(start.elapsed().as_nanos());
        
    }
    
    // Calculate average time
    let times = vec![&times_cublas, &times_cusparse];
    let avg_time: Vec<f64> = times.into_iter().map(|time| time.iter().sum::<u128>() as f64 / repeat_count as f64).collect();
    println!("cuBLAS: {:.6} ns \t| cuSPARSE: {:.6} ns", avg_time[0], avg_time[1]);
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