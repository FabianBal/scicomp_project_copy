 use std::path::Path;

use fakscpu::{dense::*, sparse::SparseProd};
use matrix_base::{Dense, COO, CSR};

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";




#[cfg(test)]
fn cmp_float(x: f64, y: f64, eps: f64) -> bool {
    (x-y).abs() < eps
}

#[cfg(test)]
fn cmp_dense(A: &Dense, B: &Dense, eps: f64) -> bool {
    let mut res = true;

    for (x,y) in A.data.iter().zip(B.data.iter()) {
        println!("AAA {} {} {}", x, y, x-y);
       res = res &&  ( (x-y).abs() < eps );
    }

    res
}



#[test]
fn test_product_dense() {
    let eps = 1e-7;

    // Number of matrices to test
    let n = 15;    

    for k in 0..n {
        println!("Testing k={}", k);
        
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let A = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let B = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let C = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let A = A.to_dense();
        let B = B.to_dense();
        let C = C.to_dense();
        
        let C_test = A.product_dense_par(&B);

        C.print();
        C_test.print();


        assert!(cmp_dense(&C, &C_test, eps));
    }
    
    
}