use std::path::Path;

use cublas::multiply;
use matrix_base::{COO, Dense};

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";



#[cfg(test)]
fn cmp_dense(A: &Dense, B: &Dense, eps: f64) -> bool {
    let mut res = true;

    for (x,y) in A.data.iter().zip(B.data.iter()) {
        // println!("AAA {} {} {}", x, y, x-y);
       res = res &&  ( (x-y).abs() < eps );
       
    }

    res
}

#[test]
fn test_product_blas() {
    let eps = 1e-6;

    // Number of matrices to test
    let n = 15;    

    for k in 0..n {
        println!("Testing k={}", k);
        
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let a = a.to_dense();
        let b = b.to_dense();
        let c = c.to_dense();
        
        let (c_test, _, _) = multiply(&a, &b).unwrap();

        // C.print();
        // C_test.print();


        assert!(cmp_dense(&c, &c_test, eps));
    }
    
    
}
