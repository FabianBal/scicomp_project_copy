use std::{cmp::{max, min}, env, fs::{self, File}, io::{stdout, BufRead, Write}, path::{Path, PathBuf}};
use matrix_base::{COO, CSR, Dense};
use fakscpu::{dense::DenseProd, sparse::SparseProd};
use gpu::WgpuTask;

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
    let mut results = vec![Vec::new(), Vec::new(), Vec::new()];
    results[0].push("Matrix1,Matrix2,cuBlas (µs),cuSparse (µs),gpuDense (µs),gpuSparse (µs),Blas (µs),cpuSparseParallel (µs),cpuDenseParallel (µs)".to_string());
    results[1].push("Matrix1,Matrix2,cuBlas (µs),cuSparse (µs),gpuDense (µs),gpuSparse (µs),Blas (µs),cpuSparseParallel (µs),cpuDenseParallel (µs)".to_string());
    results[2].push("Matrix1,Matrix2,cuBlas (µs),cuSparse (µs),gpuDense (µs),gpuSparse (µs),Blas (µs),cpuSparseParallel (µs),cpuDenseParallel (µs)".to_string());


    // search matrices in the folder
    let matrix_paths = get_matrix_paths(folder_path);

    // Generate table headers
    let table_head = &format!("{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
    "Matrix 1", "Matrix 2", "cuBlas (µs)", "cuSparse (µs)", "gpuDense (µs)", "gpuSparse (µs)", "Blas (µs)", "cpuSparseParallel (µs)", "cpuDenseParallel (µs)");

    println!("\nTotal Times:");
    println!("{}", table_head);

    let mut total_table = String::from("Total Times:\n");
    total_table += table_head;

    let mut overhead_table = String::from("Overhead Times:\n");
    overhead_table += table_head;

    let mut multiplication_table = String::from("Raw Multiplication Times:\n");
    multiplication_table += table_head;
    
    // Benchmark all possible combinations of matrices
    for matrix1_path in &matrix_paths {
        for matrix2_path in &matrix_paths {
            // do not multiply matrix with itself
            if matrix1_path == matrix2_path{
                continue;
            }

            // // do only multiply matrices with the same sparsity
            // if matrix1_path.file_name().unwrap().to_str().unwrap().split("_").collect::<Vec<&str>>()[1] != matrix2_path.file_name().unwrap().to_str().unwrap().split("_").collect::<Vec<&str>>()[1] {
            //     continue;
            // }

            // Make sure that the matrices are compatible
            if get_matrix_shape(matrix1_path).1 == get_matrix_shape(matrix2_path).0 {
                let matrix1_name = matrix1_path.file_name().unwrap().to_str().unwrap();
                let matrix2_name = matrix2_path.file_name().unwrap().to_str().unwrap();
                let avg_times = benchmark_matrix(matrix1_path, matrix2_path, repeat_count);

                // generate table rows
                multiplication_table += &format!("\n{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                matrix1_name, matrix2_name, avg_times[0].0, avg_times[1].0, avg_times[2].0, avg_times[3].0, avg_times[4].0, avg_times[5].0, avg_times[6].0);
                overhead_table += &format!("\n{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                matrix1_name, matrix2_name, avg_times[0].1, avg_times[1].1, avg_times[2].1, avg_times[3].1, avg_times[4].1, avg_times[5].1, avg_times[6].1);
                total_table += &format!("\n{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                matrix1_name, matrix2_name, avg_times[0].2, avg_times[1].2, avg_times[2].2, avg_times[3].2, avg_times[4].2, avg_times[5].2, avg_times[6].2);

                // generate output rows
                results[0].push(format!("{},{},{}", matrix1_name, matrix2_name, avg_times.iter().map(|&(x, _, _)| x.to_string()).collect::<Vec<String>>().join(",")));
                results[1].push(format!("{},{},{}", matrix1_name, matrix2_name, avg_times.iter().map(|&(_, x, _)| x.to_string()).collect::<Vec<String>>().join(",")));
                results[2].push(format!("{},{},{}", matrix1_name, matrix2_name, avg_times.iter().map(|&(_, _, x)| x.to_string()).collect::<Vec<String>>().join(",")));
            }
        }
    }
    //print tables
    println!("\n\n{}\n\n{}", overhead_table, multiplication_table);

    //generate output files
    let output_filename_raw_multiplication = format!("./output/data/{}_result_times_raw_multiplication_repeat_count_{}.csv", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"), repeat_count);
    let output_filename_overhead = format!("./output/data/{}_result_times_overhead_repeat_count_{}.csv", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"), repeat_count);
    let output_filename_total = format!("./output/data/{}_result_times_total_repeat_count_{}.csv", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"), repeat_count);
    let mut file_raw_multiplication = File::create(&output_filename_raw_multiplication).expect("Failed to create output file");
    let mut file_overhead = File::create(&output_filename_overhead).expect("Failed to create output file");
    let mut file_total = File::create(&output_filename_total).expect("Failed to create output file");
    for index in 0..results[0].len() {
        writeln!(file_raw_multiplication, "{}", results[0][index]).expect("Failed to write to raw multiplication output file");
        writeln!(file_overhead, "{}", results[1][index]).expect("Failed to write to overhead output file");
        writeln!(file_total, "{}", results[2][index]).expect("Failed to write to total output file");
    }
    println!("exported tables to {}, {}, {}", output_filename_overhead, output_filename_raw_multiplication, output_filename_total);
}

fn import_matrix(matrix_path: &Path) -> (Dense, CSR, COO) {
    let matrix_coo = COO::read_mtx(matrix_path, false).expect(format!("failed reading matrix at {}", matrix_path.display()).as_str());
    let matrix_dense = matrix_coo.to_dense();
    let matrix_csr = CSR::from_coo(&matrix_coo);
    (matrix_dense, matrix_csr, matrix_coo)
}

// Benchmark matrix multiplication
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path, repeat_count: usize) -> Vec<(u128, u128, u128)> {
    // Import matrices
    let (matrix1_dense, matrix1_csr, matrix1_coo) = import_matrix(matrix1_path);
    print!("{:<20}", matrix1_path.file_name().unwrap().to_str().unwrap().chars().take(20-1).collect::<String>());
    stdout().flush().unwrap();
    let (matrix2_dense, matrix2_csr, matrix2_coo) = import_matrix(matrix2_path);
    print!("{:<20}", matrix2_path.file_name().unwrap().to_str().unwrap().chars().take(20-1).collect::<String>());
    stdout().flush().unwrap();

    // save times for each library in format (multiply_time, overhead_time, total_time)
    let mut times_cpu_dense_parallel = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse_parallel = Vec::with_capacity(repeat_count);
    let mut times_cublas = Vec::with_capacity(repeat_count);
    let mut times_cusparse = Vec::with_capacity(repeat_count);
    let mut times_gpu_dense = Vec::with_capacity(repeat_count);
    let mut times_gpu_sparse = Vec::with_capacity(repeat_count);
    let mut times_blas = Vec::with_capacity(repeat_count);

    // run benchmark for each library
    // cuBLAS (Dense)
    for _ in 1..=repeat_count {
        let (_matrix, time_raw_multiply, time_total) = cublas::multiply(&matrix1_dense, &matrix2_dense).unwrap();
        times_cublas.push((time_raw_multiply, time_total - time_raw_multiply, time_total));
    }
    print!("{:<15}", times_cublas.iter().map(|&(_, _, total)| total).min().unwrap_or(0));
    stdout().flush().unwrap();
    
    //cuSPARSE
    for _ in 1..=repeat_count {
        let (_matrix, time_raw_multiply, time_total) = cusparse::multiply(&matrix1_csr, &matrix2_csr).unwrap();
        times_cusparse.push((time_raw_multiply, time_total - time_raw_multiply, time_total));
    }
    print!("{:<15}", times_cusparse.iter().map(|&(_, _, total)| total).min().unwrap_or(0));
    stdout().flush().unwrap();
    
    //GPU Dense Parallel
    for _ in 1..=repeat_count {
        let mut time_raw_multiply = 0;
        let mut time_total = 0;
        let _matrix_dense: Vec<f32>;
        //-----------------------------------------broken part without this it works xD
        (_matrix_dense, time_raw_multiply, time_total) = gpu::dense::multiply_for_benchmark(&matrix1_dense, &matrix2_dense, 1000*1000*1000);
        //-----------------------------------------end broken part
        times_gpu_dense.push((time_raw_multiply, time_total - time_raw_multiply, time_total));
    }
    print!("{:<15}", times_gpu_dense.iter().map(|&(_, _, total)| total).min().unwrap_or(0));
    stdout().flush().unwrap();

    //GPU Sparse
    for _ in 1..=repeat_count {
        let batch_size = 256;//not sure about this value
        let start_total = std::time::Instant::now();
        let mut time_raw_multiply= 0;
        //-----------------------------------------broken part without this it works xD
        let multiply_future = async {
            let mut gpusm = gpu::GPUSparseMultiplyer::new(&matrix1_csr, &matrix2_csr, batch_size, WgpuTask::new(1000*1000*1000).await).await;
            gpusm.create_and_load_buffer();
            let start_raw_multiply = std::time::Instant::now();
            let _res = gpusm.doit().await;
            time_raw_multiply = start_raw_multiply.elapsed().as_micros();
        };
        tokio::runtime::Runtime::new().unwrap().block_on(multiply_future);
        //-----------------------------------------end broken part
        let time_total = start_total.elapsed().as_micros();
        times_gpu_sparse.push((time_raw_multiply, time_total - time_raw_multiply, time_total));
    }
    
    print!("{:<15}", times_gpu_sparse.iter().map(|&(_, _, total)| total).sum::<u128>() / times_gpu_sparse.len() as u128);
    stdout().flush().unwrap();

    //BLAS (Dense)
    for _ in 1..=repeat_count {
        let start = std::time::Instant::now();
        let a = blas_dense::BlasDense::from_coo(&matrix1_coo);
        let b = blas_dense::BlasDense::from_coo(&matrix2_coo);
        let start_raw_multiply = std::time::Instant::now();
        let _matrix = a.prod(&b);
        let time_raw_multiply = start_raw_multiply.elapsed().as_micros();
        let time_total = start.elapsed().as_micros();
        times_blas.push((time_raw_multiply, time_total - time_raw_multiply, time_total));
    }
    print!("{:<15}", times_blas.iter().map(|&(_, _, total)| total).min().unwrap_or(0));
    stdout().flush().unwrap();
    
    //CPU Sparse Parallel
    for _ in 1..=repeat_count {
        let start = std::time::Instant::now();
        matrix1_csr.product_sparse_to_coo_par(&matrix2_csr);
        let time_total = start.elapsed().as_micros();
        times_cpu_sparse_parallel.push((time_total - 0, 0, time_total));
    }
    print!("{:<25}", times_cpu_sparse_parallel.iter().map(|&(_, _, total)| total).min().unwrap_or(0));
    stdout().flush().unwrap();
    
    //CPU Dense Parallel
    for _ in 1..=repeat_count {
        let start = std::time::Instant::now();
        matrix1_dense.product_dense_par(&matrix2_dense);
        let time_total = start.elapsed().as_micros();
        times_cpu_dense_parallel.push((time_total - 0, 0, time_total));
    }
    print!("{:<25}", times_cpu_dense_parallel.iter().map(|&(_, _, total)| total).min().unwrap_or(0));
    stdout().flush().unwrap();
    println!();

    // Calculate average times
    let times_vec: Vec<Vec<(u128, u128, u128)>> = vec![times_cublas, times_cusparse, times_gpu_dense, times_gpu_sparse, times_blas, times_cpu_sparse_parallel, times_cpu_dense_parallel];
    let min_times: Vec<(u128, u128, u128)> = times_vec.into_iter().map(|times| times.iter().fold((u128::max_value(), u128::max_value(), u128::max_value()), |acc, time| (max(1, min(acc.0, time.0)), max(1,min(acc.1, time.1)), max(1,min(acc.2, time.2))))).collect();
    min_times
    // let sum_times: Vec<(u128, u128, u128)> = times_vec.into_iter().map(|times| times.iter().fold((0, 0, 0), |acc, time| (acc.0 + time.0, acc.1 + time.1, acc.2 + time.2))).collect();
    // let avg_times: Vec<(u128, u128, u128)> = sum_times.iter().map(|sum_time| (sum_time.0 / repeat_count as u128, sum_time.1 / repeat_count as u128, sum_time.2 / repeat_count as u128)).collect();
    // avg_times
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
    matrix_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    matrix_paths
}

fn get_matrix_shape(file_name: &PathBuf) -> (usize, usize) {
    let file = File::open(file_name).expect("Error parsing file.");
    let file = std::io::BufReader::new(file);

    let mut liter = file.lines();

    // "Header" / comments
    // Ignore all comments / type codes (assume 'MatrixMarket matrix coordinate real general format')
    for line in liter.by_ref() {
        let line = String::from(line.map_err(|_| ("Error parsing line")).unwrap().trim());
        if line.starts_with("%"){
            continue;
        } else {
            let mut hspl = line.split(' ');
            let m = hspl.next().unwrap().parse().unwrap();
            let n = hspl.next().unwrap().parse().unwrap();
            let _l: usize = hspl.next().unwrap().parse().unwrap();
            return (m, n);
        }            
    }
    panic!("{}", format!("no shape found in file {}", file_name.file_name().unwrap().to_str().unwrap()))
}