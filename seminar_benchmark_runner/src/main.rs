use std::{
    cmp::{min},
    env,
    fs::{self, File},
    io::{stdout, BufRead, Write},
    path::{Path, PathBuf},
};
use matrix_base::{Dense, COO, CSR};
use fakscpu::{dense::DenseProd, sparse::SparseProd};
use gpu::WgpuTask; // WgpuTask von gpu-Crate
use tools::TimingResult; // TimingResult von tools-Crate

/// Benchmark matrix multiplication using different libraries
/// load all matrices from provided folder path or default and benchmark all possible combinations
fn main() {
    // Default values
    let mut repeat_count: usize = 10;
    let mut folder_path = "./matrix_instances/generated/dense";

    // Get command-line arguments if provided
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        repeat_count = args[1].parse().expect("Failed to parse repeat count");
    }
    if args.len() >= 3 {
        folder_path = &args[2];
    }

    // NEU: Initialisierung der Ergebnis-Vektoren für detaillierte CSVs
    // results[0] -> Raw Multiplication
    // results[1] -> H2D
    // results[2] -> D2H
    // results[3] -> Initialization + Cleanup (reine API-Overheads ohne Kopien für WGPU; mit Kopien für cuBLAS/cuSPARSE)
    // results[4] -> Total
    // results[5] -> Combined Overhead (initialization + h2d + d2h + cleanup)
    let mut results: Vec<Vec<String>> = vec![Vec::new(); 6];

    // Header für die CSV-Dateien generieren
    let common_header_components = "Matrix1,Matrix2";
    let libraries = [
        "cuBlas", "cuSparse", "gpuDense", "gpuSparse", "Blas", "cpuSparseParallel", "cpuDenseParallel"
    ];

    let mut raw_mult_header = common_header_components.to_string();
    let mut h2d_header = common_header_components.to_string();
    let mut d2h_header = common_header_components.to_string();
    let mut init_cleanup_header = common_header_components.to_string();
    let mut total_header = common_header_components.to_string();
    let mut combined_overhead_header = common_header_components.to_string(); // <-- NEU: Header für Combined Overhead

    for lib in &libraries {
        raw_mult_header.push_str(&format!(",{}_Raw (µs)", lib));
        h2d_header.push_str(&format!(",{}_H2D (µs)", lib));
        d2h_header.push_str(&format!(",{}_D2H (µs)", lib));
        init_cleanup_header.push_str(&format!(",{}_InitCleanup (µs)", lib));
        total_header.push_str(&format!(",{}_Total (µs)", lib));
        combined_overhead_header.push_str(&format!(",{}_CombinedOverhead (µs)", lib));
    }

    results[0].push(raw_mult_header);
    results[1].push(h2d_header);
    results[2].push(d2h_header);
    results[3].push(init_cleanup_header);
    results[4].push(total_header);
    results[5].push(combined_overhead_header);

    // search matrices in the folder
    let matrix_paths = get_matrix_paths(folder_path);

    // Generate table headers for console output (can stay simple with total times)
    let table_head = &format!("{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                              "Matrix 1", "Matrix 2", "cuBlas (µs)", "cuSparse (µs)", "gpuDense (µs)", "gpuSparse (µs)", "Blas (µs)", "cpuSparseParallel (µs)", "cpuDenseParallel (µs)");

    println!("\nTotal Times:");
    println!("{}", table_head);

    // These String buffers are for console output only, not for CSVs anymore
    let mut total_table = String::from("Total Times:\n");
    total_table += table_head;

    let mut overhead_table = String::from("Overhead Times:\n");
    overhead_table += table_head;

    let mut multiplication_table = String::from("Raw Multiplication Times:\n");
    multiplication_table += table_head;

    // Benchmark all possible combinations of matrices
    for matrix1_path in &matrix_paths {
        for matrix2_path in &matrix_paths {
            if matrix1_path == matrix2_path{
                continue;
            }
            if get_matrix_shape(matrix1_path).1 == get_matrix_shape(matrix2_path).0 {
                let matrix1_name = matrix1_path.file_name().unwrap().to_str().unwrap();
                let matrix2_name = matrix2_path.file_name().unwrap().to_str().unwrap();

                let current_matrix_type = if matrix1_name.starts_with("dense_") {
                    "dense"
                } else if matrix1_name.starts_with("sparse_") {
                    "sparse"
                } else {
                    "unknown"
                };

                // benchmark_matrix gibt Vec<TimingResult> zurück
                let min_times = benchmark_matrix(matrix1_path, matrix2_path, repeat_count, current_matrix_type);

                // generate table rows for console (using total and raw_multiply from min_times)
                multiplication_table += &format!("\n{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                                                 matrix1_name, matrix2_name, min_times[0].raw_multiply_us, min_times[1].raw_multiply_us,
                                                 min_times[2].raw_multiply_us, min_times[3].raw_multiply_us, min_times[4].raw_multiply_us,
                                                 min_times[5].raw_multiply_us, min_times[6].raw_multiply_us);

                overhead_table += &format!("\n{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                                           matrix1_name, matrix2_name,
                                           min_times[0].initialization_us + min_times[0].h2d_us + min_times[0].d2h_us + min_times[0].cleanup_us, // Summe der Overheads für cuBlas
                                           min_times[1].initialization_us + min_times[1].h2d_us + min_times[1].d2h_us + min_times[1].cleanup_us, // Summe der Overheads für cuSparse
                                           min_times[2].initialization_us + min_times[2].h2d_us + min_times[2].d2h_us + min_times[2].cleanup_us, // Summe der Overheads für gpuDense
                                           min_times[3].initialization_us + min_times[3].h2d_us + min_times[3].d2h_us + min_times[3].cleanup_us, // Summe der Overheads für gpuSparse
                                           min_times[4].initialization_us + min_times[4].h2d_us + min_times[4].d2h_us + min_times[4].cleanup_us, // Summe der Overheads für Blas
                                           min_times[5].initialization_us + min_times[5].h2d_us + min_times[5].d2h_us + min_times[5].cleanup_us, // Summe der Overheads für cpuSparseParallel
                                           min_times[6].initialization_us + min_times[6].h2d_us + min_times[6].d2h_us + min_times[6].cleanup_us  // Summe der Overheads für cpuDenseParallel
                );

                total_table += &format!("\n{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<25}{:<25}",
                                        matrix1_name, matrix2_name, min_times[0].total_us, min_times[1].total_us,
                                        min_times[2].total_us, min_times[3].total_us, min_times[4].total_us,
                                        min_times[5].total_us, min_times[6].total_us);


                let mut raw_mult_row = format!("{},{}", matrix1_name, matrix2_name);
                let mut h2d_row = format!("{},{}", matrix1_name, matrix2_name);
                let mut d2h_row = format!("{},{}", matrix1_name, matrix2_name);
                let mut init_cleanup_row = format!("{},{}", matrix1_name, matrix2_name);
                let mut total_row = format!("{},{}", matrix1_name, matrix2_name);
                let mut combined_overhead_row = format!("{},{}", matrix1_name, matrix2_name); // <-- NEU

                for res in &min_times {
                    raw_mult_row.push_str(&format!(",{}", res.raw_multiply_us));
                    h2d_row.push_str(&format!(",{}", res.h2d_us));
                    d2h_row.push_str(&format!(",{}", res.d2h_us));
                    init_cleanup_row.push_str(&format!(",{}", res.initialization_us + res.cleanup_us));
                    total_row.push_str(&format!(",{}", res.total_us));
                    // NEU: Berechnung des kombinierten Overheads
                    let current_combined_overhead = res.initialization_us + res.h2d_us + res.d2h_us + res.cleanup_us;
                    combined_overhead_row.push_str(&format!(",{}", current_combined_overhead)); // <-- NEU
                }

                results[0].push(raw_mult_row);     // Raw Multiplication
                results[1].push(h2d_row);          // H2D
                results[2].push(d2h_row);          // D2H
                results[3].push(init_cleanup_row); // Initialization + Cleanup
                results[4].push(total_row);        // Total
                results[5].push(combined_overhead_row); // <-- NEU
            }
        }
    }

    // print tables to console
    println!("\n\n{}\n\n{}", overhead_table, multiplication_table);

    // NEU: generate output files for all detailed times
    let base_filename = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let output_dir = Path::new("./output/data");
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let write_csv = |file_suffix: &str, data: &Vec<String>, result_type: &str| {
        let filename = format!("{}/{}_{}_repeat_count_{}.csv", output_dir.display(), base_filename, file_suffix, repeat_count);
        let mut file = File::create(&filename).expect(&format!("Failed to create {} output file at {}", result_type, filename));
        for line in data {
            writeln!(file, "{}", line).expect(&format!("Failed to write to {} output file", result_type));
        }
        println!("exported {} table to {}", result_type, filename);
    };

    write_csv("raw_multiplication_times", &results[0], "raw multiplication");
    write_csv("h2d_times", &results[1], "H2D");
    write_csv("d2h_times", &results[2], "D2H");
    write_csv("init_cleanup_times", &results[3], "initialization and cleanup");
    write_csv("total_times", &results[4], "total");
    write_csv("combined_overhead_times", &results[5], "combined overhead"); // <-- NEU
}

fn import_matrix(matrix_path: &Path) -> (Dense, CSR, COO) {
    let matrix_coo = COO::read_mtx(matrix_path, false)
        .expect(format!("failed reading matrix at {}", matrix_path.display()).as_str());
    let matrix_dense = matrix_coo.to_dense();
    let matrix_csr = CSR::from_coo(&matrix_coo);
    (matrix_dense, matrix_csr, matrix_coo)
}

// Benchmark matrix multiplication
fn benchmark_matrix(matrix1_path: &Path, matrix2_path: &Path, repeat_count: usize, matrix_type: &str) -> Vec<TimingResult> {
    let (matrix1_dense, matrix1_csr, matrix1_coo) = import_matrix(matrix1_path);
    print!("{:<20}", matrix1_path.file_name().unwrap().to_str().unwrap().chars().take(19).collect::<String>());
    stdout().flush().unwrap();
    let (matrix2_dense, matrix2_csr, matrix2_coo) = import_matrix(matrix2_path);
    print!("{:<20}", matrix2_path.file_name().unwrap().to_str().unwrap().chars().take(19).collect::<String>());
    stdout().flush().unwrap();

    // Initiale Vektoren für TimingResults
    let mut times_cpu_dense_parallel = Vec::with_capacity(repeat_count);
    let mut times_cpu_sparse_parallel = Vec::with_capacity(repeat_count);
    let mut times_cublas = Vec::with_capacity(repeat_count);
    let mut times_cusparse = Vec::with_capacity(repeat_count);
    let mut times_gpu_dense = Vec::with_capacity(repeat_count);
    let mut times_gpu_sparse = Vec::with_capacity(repeat_count);
    let mut times_blas = Vec::with_capacity(repeat_count);

    // **REFERENZBERECHNUNG AUF DER CPU (EINMALIG PRO MATRIX-PAAR)**
    // Diese Ergebnisse werden für die Korrektheitsprüfung verwendet.
    let reference_result_dense_cpu = matrix1_dense.product_dense_par(&matrix2_dense);
    let reference_result_sparse_coo_cpu = matrix1_csr.product_sparse_to_coo_par(&matrix2_csr);


    // cuBLAS (Dense)
    if matrix_type == "dense" {
        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();
            let (res_matrix_cublas_dense, time_raw_multiply, time_total_internal, time_h2d_cublas, time_d2h_cublas) = cublas::multiply(&matrix1_dense, &matrix2_dense).unwrap();

            let mut timing_result = TimingResult {
                initialization_us: time_total_internal - time_raw_multiply - time_h2d_cublas - time_d2h_cublas, // Overhead ohne H2D/D2H
                h2d_us: time_h2d_cublas,
                raw_multiply_us: time_raw_multiply,
                d2h_us: time_d2h_cublas,
                cleanup_us: 0,
                total_us: start_total.elapsed().as_micros(),
            };

            // Korrektheitsprüfung
            if !check_results_dense(&res_matrix_cublas_dense.data, &reference_result_dense_cpu.data) {
                eprintln!("WARNING: cuBlas result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                timing_result = TimingResult::max_values();
            }
            times_cublas.push(timing_result);
        }
    } else {
        // Nicht anwendbar für diesen Matrixtyp
        for _ in 0..repeat_count { times_cublas.push(TimingResult::zero()); }
    }
    print!("{:<15}", times_cublas.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();

    // cuSPARSE (Sparse)
    if matrix_type == "sparse" {
        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();
            // ÄNDERUNG: Neuer Rückgabetyp für cusparse::multiply
            let (res_matrix_cusparse, time_raw_multiply, time_total_internal, time_h2d_cusparse, time_d2h_cusparse) = cusparse::multiply(&matrix1_csr, &matrix2_csr).unwrap();

            let mut timing_result = TimingResult {
                initialization_us: time_total_internal - time_raw_multiply - time_h2d_cusparse - time_d2h_cusparse, // Overhead ohne H2D/D2H
                h2d_us: time_h2d_cusparse, // NEU: Gemessener H2D-Wert
                raw_multiply_us: time_raw_multiply,
                d2h_us: time_d2h_cusparse, // NEU: Gemessener D2H-Wert
                cleanup_us: 0, // In der cusparse-Implementierung nicht explizit gemessen, Teil des restlichen Overheads
                total_us: start_total.elapsed().as_micros(),
            };

            let cusparse_result_coo = CSR::to_coo(&res_matrix_cusparse);

            if !check_results_sparse_coo(&cusparse_result_coo, &reference_result_sparse_coo_cpu) {
                eprintln!("WARNING: cuSparse result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                timing_result = TimingResult::max_values();
            }
            times_cusparse.push(timing_result);
        }
    } else {
        for _ in 0..repeat_count { times_cusparse.push(TimingResult::zero()); }
    }
    print!("{:<15}", times_cusparse.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();

    // GPU Dense Parallel
    // Führe nur aus, wenn der Matrixtyp "dense" ist
    if matrix_type == "dense" {
        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();
            let (gpu_dense_result_vec, mut timing_result) = gpu::dense::multiply_for_benchmark(&matrix1_dense, &matrix2_dense, 1000*1000*1000);
            timing_result.total_us = start_total.elapsed().as_micros();

            // Korrektheitsprüfung: gpu_dense_result_vec ist Vec<f32>, muss zu Vec<f64> konvertiert werden
            let gpu_dense_result_vec_f64: Vec<f64> = gpu_dense_result_vec.iter().map(|&x| x as f64).collect();
            if !check_results_dense(&gpu_dense_result_vec_f64, &reference_result_dense_cpu.data) {
                eprintln!("WARNING: gpuDense result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                timing_result = TimingResult::max_values();
            }
            times_gpu_dense.push(timing_result);
        }
    } else {
        // Nicht anwendbar
        for _ in 0..repeat_count { times_gpu_dense.push(TimingResult::zero()); }
    }
    print!("{:<15}", times_gpu_dense.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();

    // GPU Sparse
    // Führe nur aus, wenn der Matrixtyp "sparse" ist
    if matrix_type == "sparse" {
        let wgpu_task_global_init_start = std::time::Instant::now();
        let wgpu_task_global = tokio::runtime::Runtime::new().unwrap().block_on(WgpuTask::new(1000 * 1000 * 1000));
        let wgpu_task_global_init_us = wgpu_task_global_init_start.elapsed().as_micros();

        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();

            let (mut gpusm, mut new_timing) = tokio::runtime::Runtime::new().unwrap().block_on(
                gpu::GPUSparseMultiplyer::new(
                    &matrix1_csr,
                    &matrix2_csr,
                    256,
                    wgpu_task_global.clone()
                )
            );

            new_timing.initialization_us += wgpu_task_global_init_us;

            let h2d_time_for_instance = gpusm.create_and_load_buffer();
            new_timing.h2d_us = h2d_time_for_instance;

            let (_, doit_timing) = tokio::runtime::Runtime::new().unwrap().block_on(gpusm.doit());

            let mut final_timing = TimingResult {
                initialization_us: new_timing.initialization_us + doit_timing.initialization_us,
                h2d_us: new_timing.h2d_us,
                raw_multiply_us: doit_timing.raw_multiply_us,
                d2h_us: doit_timing.d2h_us,
                cleanup_us: doit_timing.cleanup_us,
                total_us: start_total.elapsed().as_micros(),
            };

            // Korrektheitsprüfung für Sparse
            if let Some(gpu_sparse_result_coo) = gpusm.cast_result() {
                if !check_results_sparse_coo(&gpu_sparse_result_coo, &reference_result_sparse_coo_cpu) {
                    eprintln!("WARNING: gpuSparse result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                    final_timing = TimingResult::max_values();
                }
            } else {
                eprintln!("WARNING: gpuSparse result for {} x {} returned None!", matrix1_path.display(), matrix2_path.display());
                final_timing = TimingResult::max_values();
            }
            times_gpu_sparse.push(final_timing);
        }
    } else {
        // Nicht anwendbar
        for _ in 0..repeat_count { times_gpu_sparse.push(TimingResult::zero()); }
    }
    print!("{:<15}", times_gpu_sparse.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();

    // BLAS (Dense)
    // Führe nur aus, wenn der Matrixtyp "dense" ist
    if matrix_type == "dense" {
        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();
            let start_init_h2d = std::time::Instant::now();
            let a = blas_dense::BlasDense::from_coo(&matrix1_coo);
            let b = blas_dense::BlasDense::from_coo(&matrix2_coo);
            let init_h2d_us = start_init_h2d.elapsed().as_micros();

            let start_raw_multiply = std::time::Instant::now();
            let blas_result_dense = a.prod(&b); // Ergebnis von BLAS
            let raw_multiply_us = start_raw_multiply.elapsed().as_micros();

            let mut timing_result = TimingResult {
                initialization_us: init_h2d_us,
                h2d_us: 0,
                raw_multiply_us,
                d2h_us: 0,
                cleanup_us: 0,
                total_us: start_total.elapsed().as_micros(),
            };

            // Korrektheitsprüfung: blas_result_dense.data ist Vec<f64>
            if !check_results_dense(&blas_result_dense.data, &matrix_base::Dense::as_column_major(&reference_result_dense_cpu).data) {
                eprintln!("WARNING: Blas result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                timing_result = TimingResult::max_values();
            }
            times_blas.push(timing_result);
        }
    } else {
        // Nicht anwendbar
        for _ in 0..repeat_count { times_blas.push(TimingResult::zero()); }
    }
    print!("{:<15}", times_blas.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();

    // CPU Sparse Parallel
    // Führe nur aus, wenn der Matrixtyp "sparse" ist
    if matrix_type == "sparse" {
        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();
            let cpu_sparse_result_coo = matrix1_csr.product_sparse_to_coo_par(&matrix2_csr);
            let total_us = start_total.elapsed().as_micros();

            let mut timing_result = TimingResult {
                initialization_us: 0, h2d_us: 0,
                raw_multiply_us: total_us,
                d2h_us: 0, cleanup_us: 0,
                total_us,
            };

            // Korrektheitsprüfung
            if !check_results_sparse_coo(&cpu_sparse_result_coo, &reference_result_sparse_coo_cpu) {
                eprintln!("WARNING: cpuSparseParallel result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                timing_result = TimingResult::max_values();
            }
            times_cpu_sparse_parallel.push(timing_result);
        }
    } else {
        // Nicht anwendbar
        for _ in 0..repeat_count { times_cpu_sparse_parallel.push(TimingResult::zero()); }
    }
    print!("{:<25}", times_cpu_sparse_parallel.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();

    // CPU Dense Parallel
    // Führe nur aus, wenn der Matrixtyp "dense" ist
    if matrix_type == "dense" {
        for _ in 0..repeat_count {
            let start_total = std::time::Instant::now();
            let cpu_dense_result = matrix1_dense.product_dense_par(&matrix2_dense);
            let total_us = start_total.elapsed().as_micros();

            let mut timing_result = TimingResult {
                initialization_us: 0, h2d_us: 0,
                raw_multiply_us: total_us,
                d2h_us: 0, cleanup_us: 0,
                total_us,
            };

            // Korrektheitsprüfung
            // Da dies unsere Referenz ist, sollte es immer korrekt sein, aber wir prüfen es trotzdem.
            if !check_results_dense(&cpu_dense_result.data, &reference_result_dense_cpu.data) {
                eprintln!("WARNING: cpuDenseParallel result for {} x {} is INCORRECT!", matrix1_path.display(), matrix2_path.display());
                timing_result = TimingResult::max_values();
            }
            times_cpu_dense_parallel.push(timing_result);
        }
    } else {
        // Nicht anwendbar
        for _ in 0..repeat_count { times_cpu_dense_parallel.push(TimingResult::zero()); }
    }
    print!("{:<25}", times_cpu_dense_parallel.iter().map(|t| t.total_us).min().unwrap_or(0));
    stdout().flush().unwrap();
    println!();

    // Minimums über alle Läufe berechnen
    let all_times: Vec<Vec<TimingResult>> = vec![
        times_cublas,
        times_cusparse,
        times_gpu_dense,
        times_gpu_sparse,
        times_blas,
        times_cpu_sparse_parallel,
        times_cpu_dense_parallel,
    ];

    let min_results: Vec<TimingResult> = all_times.into_iter().map(|times| {
        // Findet das Minimum für jede Zeitkomponente, ignoriert dabei u128::MAX (Fehlerwerte)
        times.iter().fold(TimingResult {
            initialization_us: u128::MAX,
            h2d_us: u128::MAX,
            raw_multiply_us: u128::MAX,
            d2h_us: u128::MAX,
            cleanup_us: u128::MAX,
            total_us: u128::MAX,
        }, |mut acc, time| {
            // Nur wenn der aktuelle Zeitwert kein Fehlerwert ist, wird das Minimum aktualisiert.
            // Sonst bleibt acc auf MAX, wenn alle Werte MAX sind.
            if time.initialization_us != u128::MAX { acc.initialization_us = min(acc.initialization_us, time.initialization_us); }
            if time.h2d_us != u128::MAX { acc.h2d_us = min(acc.h2d_us, time.h2d_us); }
            if time.raw_multiply_us != u128::MAX { acc.raw_multiply_us = min(acc.raw_multiply_us, time.raw_multiply_us); }
            if time.d2h_us != u128::MAX { acc.d2h_us = min(acc.d2h_us, time.d2h_us); }
            if time.cleanup_us != u128::MAX { acc.cleanup_us = min(acc.cleanup_us, time.cleanup_us); }
            if time.total_us != u128::MAX { acc.total_us = min(acc.total_us, time.total_us); }
            acc
        })
    }).collect();

    min_results
}

/// Checks if two dense matrix results (flattened Vec<f64> vs Dense.data) are approximately equal.
/// Both inputs are expected as slices of f64 for consistent comparison.
fn check_results_dense(result_tested_data: &[f64], reference_data: &[f64]) -> bool {

    if result_tested_data.len() != reference_data.len() {
        eprintln!("Error: Result lengths mismatch: Tested {} vs Reference {}", result_tested_data.len(), reference_data.len());
        return false;
    }

    let tolerance = 1e-3; // Absolute Toleranz für f32/f64 Vergleich
    let mut errors_found = 0;
    let max_errors_to_print = 5; // Begrenze die Anzahl der ausgegebenen Fehler

    for i in 0..result_tested_data.len() {
        let diff = (result_tested_data[i] - reference_data[i]).abs();
        if diff > tolerance {
            errors_found += 1;
            if errors_found <= max_errors_to_print {
                eprintln!("Mismatch at index {}: Tested={:.2e} Reference={:.2e} Diff={:.2e}", i, result_tested_data[i], reference_data[i], diff);
            }
        }
    }
    if errors_found > max_errors_to_print {
        eprintln!("... ({} more errors not printed)", errors_found - max_errors_to_print);
    }
    if errors_found > 0 {
        eprintln!("Total mismatches: {}", errors_found);
    }


    errors_found == 0
}


/// Helper for Sparse COO results. Compares two COO matrices.
fn check_results_sparse_coo(result_gpu_coo: &COO, result_cpu_coo: &COO) -> bool {
    if result_gpu_coo.shape != result_cpu_coo.shape {
        eprintln!("Error: Sparse shapes mismatch: GPU {:?} vs CPU {:?}", result_gpu_coo.shape, result_cpu_coo.shape);
        return false;
    }

    // Für einen robusten COO-Vergleich müssen beide Listen der (i, j, x) Triplets sortiert werden,
    // da die Reihenfolge der Nicht-Null-Elemente in COO variieren kann.
    let mut gpu_data_sorted = result_gpu_coo.data.clone();
    gpu_data_sorted.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1))); // Sort by row, then by column

    let mut cpu_data_sorted = result_cpu_coo.data.clone();
    cpu_data_sorted.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1))); // Sort by row, then by column

    if gpu_data_sorted.len() != cpu_data_sorted.len() {
        eprintln!("Error: Sparse NNZ count mismatch: GPU {} vs CPU {}", gpu_data_sorted.len(), cpu_data_sorted.len());
        return false;
    }

    let tolerance = 1e-5;
    let mut errors_found = 0;
    let max_errors_to_print = 5;

    for i in 0..gpu_data_sorted.len() {
        let (r_g, c_g, v_g) = gpu_data_sorted[i];
        let (r_c, c_c, v_c) = cpu_data_sorted[i];

        // Prüfe Index und Wert
        if r_g != r_c || c_g != c_c || (v_g - v_c).abs() > tolerance {
            errors_found += 1;
            if errors_found <= max_errors_to_print {
                eprintln!("Sparse Mismatch at sorted index {}: GPU=({},{},{:.2e}) CPU=({},{},{:.2e}) Diff={:.2e}",
                          i, r_g, c_g, v_g, r_c, c_c, v_c, (v_g - v_c).abs());
            }
        }
    }
    if errors_found > max_errors_to_print {
        eprintln!("... ({} more sparse errors not printed)", errors_found - max_errors_to_print);
    }
    if errors_found > 0 {
        eprintln!("Total sparse mismatches: {}", errors_found);
    }

    errors_found == 0
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
        if line.starts_with("%") {
            continue;
        } else {
            let mut hspl = line.split(' ');
            let m = hspl.next().unwrap().parse().unwrap();
            let n = hspl.next().unwrap().parse().unwrap();
            let _l: usize = hspl.next().unwrap().parse().unwrap();
            return (m, n);
        }
    }
    panic!(
        "{}",
        format!(
            "no shape found in file {}",
            file_name.file_name().unwrap().to_str().unwrap()
        )
    )
}


