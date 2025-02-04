use std::path::Path;

use blas_dense::BlasDense;
use matrix_base::COO;

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";



#[cfg(test)]
fn cmp_dense(A: &BlasDense, B: &BlasDense, eps: f64) -> bool {
    let mut res = true;

    for (x,y) in A.data.iter().zip(B.data.iter()) {
        // println!("AAA {} {} {}", x, y, x-y);
       res = res &&  ( (x-y).abs() < eps );
    }

    res
}

#[test]
fn test_product_blas() {
    let eps = 1e-7;

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

        let a = BlasDense::from_coo(&a);
        let b = BlasDense::from_coo(&b);
        let c = BlasDense::from_coo(&c);
        
        let c_test = a.prod(&b);

        // C.print();
        // C_test.print();


        assert!(cmp_dense(&c, &c_test, eps));
    }
    
    
}