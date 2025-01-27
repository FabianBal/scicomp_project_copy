use std::path::Path;

use fakscpu::cpu::*;

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";




#[cfg(test)]
fn cmp_float(x: f64, y: f64, eps: f64) -> bool {
    x-y < eps
}

#[cfg(test)]
fn cmp_dense(A: &Dense, B: &Dense, eps: f64) -> bool {
    let mut res = true;

    for (x,y) in A.data.iter().zip(B.data.iter()) {
       res = res &&  ( x-y < eps );
    }

    res
}



#[test]
fn test_read_coo() {
    let eps = 1e-10;

    // a001.mtx
    let fname1 = Path::new(DATA_PATH).join(&Path::new("a001.mtx"));
    let coo = COO::read_mtx(&fname1, false).expect("Failed reading matrix during test");

    assert_eq!(coo.shape.0, 3);
    assert_eq!(coo.shape.1, 3);
    assert_eq!(coo.data.len(), 5);

    let data_a001 = [(0,0,25.), (1,0,15.), (1,1,18.), (2,0,5.), (2,2,11.)];

    for ((i,j,x), (a,b,y)) in coo.data.iter().zip(data_a001) {
        assert_eq!(*i, a);
        assert_eq!(*j, b);
        assert!(cmp_float(*x, y, eps));
    }

    // a002.mtx
    let fname2 = Path::new(DATA_PATH).join(&Path::new("a002.mtx"));
    let coo = COO::read_mtx(&fname2, false).expect("Failed reading matrix during test");

    assert_eq!(coo.shape.0, 2);
    assert_eq!(coo.shape.1, 3);
    assert_eq!(coo.data.len(), 3);

    let data_a001 = [(0,0,25.), (1,0,15.), (1,1,18.)];

    for ((i,j,x), (a,b,y)) in coo.data.iter().zip(data_a001) {
        assert_eq!(*i, a);
        assert_eq!(*j, b);
        assert!(cmp_float(*x, y, eps));
    }
}


#[test]
fn test_read_csr() {   
    let eps = 1e-10;

    // TODO

}


#[test]
fn test_product_csr() {
    let eps = 1e-7;

    // Number of matrices to test
    let n = 2;    

    for k in 0..n {
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let A = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let B = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let C = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let A = CSR::from_coo(A);
        let B = CSR::from_coo(B);
        let C = C.to_dense();
        
        let C_test = A.product(&B);

        C.print();
        C_test.print();


        assert!(cmp_dense(&C, &C_test, eps));
    }
    
    
}

