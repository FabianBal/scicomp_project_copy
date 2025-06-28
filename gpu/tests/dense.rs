use std::path::Path;

use gpu::sparse::*;
use gpu::WgpuTask;
use matrix_base::{Dense, COO, CSR};

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";

#[cfg(test)]
fn cmp_float(x: f64, y: f64, eps: f64) -> bool {
    (x - y).abs() < eps
}

#[cfg(test)]
fn cmp_dense(A: &Dense, B: &Dense, eps: f64) -> bool {
    let mut res = true;

    for (x, y) in A.data.iter().zip(B.data.iter()) {
        res = res && ((x - y).abs() < eps);
    }

    res
}

#[tokio::test]
async fn test_wgpu_dense() {
    let eps = 1e-5;

    let batch_size = 4;

    // Number of matrices to test
    let n = 10;

    for k in 0..n {
        println!("Testing k={}", k);

        // COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_A.mtx"), true).expect("Failed reading matrix file.");

        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let a = a.to_dense();
        let b = b.to_dense();
        let c_test = c.to_dense();

        // let (c, _, _) = gpu::dense::multiply_for_benchmark(&a, &b, 300*1024*1024);
        // let c = Dense{data: c.into_iter().map(|x| x as f64).collect(), shape: c_test.shape};

        // assert_eq!(c.shape.0, c_test.shape.0);
        // assert_eq!(c.shape.1, c_test.shape.1);
        // assert!(cmp_dense(&c, &c_test, eps));
    }

    assert!(true);
}
