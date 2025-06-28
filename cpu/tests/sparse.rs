use std::path::Path;

use fakscpu::sparse::SparseProd;
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
        println!("AAA {} {} {}", x, y, x - y);
        res = res && ((x - y).abs() < eps);
    }

    res
}

#[test]
fn test_product_csr() {
    let eps = 1e-7;

    // Number of matrices to test
    let n = 9;

    for k in 0..n {
        println!("Testing k={}", k);

        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let a = CSR::from_coo(&a);
        let b = CSR::from_coo(&b);
        let c = c.to_dense();

        let c_test = a.product(&b);

        c.print();
        c_test.print();

        assert!(cmp_dense(&c, &c_test, eps));
    }
}

#[test]
fn test_product_csr_sparse() {
    let eps = 1e-7;

    // Number of matrices to test
    let n = 9;

    for k in 0..n {
        println!("Testing k={}", k);

        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let a = CSR::from_coo(&a);
        let b = CSR::from_coo(&b);
        let c = c.to_dense();

        let c_test = a.product_sparse(&b).to_dense();

        // C.print();
        // C_test.print();

        assert!(cmp_dense(&c, &c_test, eps));
    }
}

#[test]
fn test_product_csr_sparse_par() {
    let eps = 1e-7;

    // Number of matrices to test
    let n = 9;

    for k in 0..n {
        println!("Testing k={}", k);

        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let a = CSR::from_coo(&a);
        let b = CSR::from_coo(&b);
        let c = c.to_dense();

        let c_test = a.product_sparse_par(&b).to_dense();

        // C.print();
        // C_test.print();

        assert!(cmp_dense(&c, &c_test, eps));
    }
}

#[test]
fn test_product_csr_sparse_to_coo_par() {
    let eps = 1e-7;

    // Number of matrices to test
    let n = 9;

    for k in 0..n {
        println!("Testing k={}", k);

        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname =
            Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let a = CSR::from_coo(&a);
        let b = CSR::from_coo(&b);
        let c = c.to_dense();

        let c_test = a.product_sparse_to_coo_par(&b).to_dense();

        // C.print();
        // C_test.print();

        assert!(cmp_dense(&c, &c_test, eps));
        // assert!(false);
    }
}
