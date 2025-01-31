use std::{env, fs, path::{Path, PathBuf}};

use matrix_base::{COO, CSR, Dense};
fn main() {
    let mut folder_path = "./matrix_instances";

    // Get the folder path from command-line arguments if provided
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        folder_path = &args[1];

    }
    let matrix_paths = get_matrix_paths(folder_path);
    println!("Found {} matrices in '{}'", matrix_paths.len(), folder_path);
    
    //start benchmarking
    for matrix1_path in &matrix_paths {
        for matrix2_path in &matrix_paths {
            //ToDo: needs to make sure that the matrices are compatible
            println!("\nBenchmarking {} X {}", matrix1_path.file_name().unwrap().to_str().unwrap(), matrix2_path.file_name().unwrap().to_str().unwrap());
            benchmark_matrix(matrix1_path, matrix2_path);
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
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path) {
    let (matrix1_dense, matrix1_csr) = import_matrix(matrix1_path);
    let (matrix2_dense, matrix2_csr) = import_matrix(matrix2_path);

    let start = std::time::Instant::now();
    cublas::multiply(&matrix1_dense, &matrix2_dense);
    let cublas_time = start.elapsed().as_nanos();

    let start = std::time::Instant::now();
    cusparse::multiply(&matrix1_csr, &matrix2_csr);
    let cusparse_time = start.elapsed().as_nanos();

    println!("cuBLAS: {:.6} ns \t| cuSPARSE: {:.6} ns", cublas_time, cusparse_time);
}

// Returns all paths to matrices inside folder_path
fn get_matrix_paths(folder_path: &str) -> Vec<PathBuf> {
    let mut matrix_paths: Vec<PathBuf> = Vec::new();

    let path = Path::new(folder_path);
    if !path.is_dir() {
        eprintln!("Error: '{}' is not a valid directory.", folder_path);
        return matrix_paths;
    }

    // Iterate over directory entries
    let entries = fs::read_dir(path).expect("Failed to read directory");
    for entry in entries.flatten() {
        let file_path = entry.path();
        if file_path.is_file() && file_path.extension().unwrap_or_default() == "mtx" {
            matrix_paths.push(file_path);
        }
    }
    matrix_paths
}