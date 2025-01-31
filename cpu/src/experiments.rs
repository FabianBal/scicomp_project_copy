use std::path::Path;
use std::{time::Instant, env};

use fakscpu::sparse::SparseProd;
use matrix_base::{Dense, COO, CSR};




fn main() {
    let DATA_PATH: &str = "../matrix_instances";

    let n = 19;

    // s2p = sparse-to-dense
    let mut times_s2d = vec![];
    let mut times_s2s = vec![];
    let mut times_s2s_par = vec![];

    for k in 0..n {
        println!("Loading k={}", k);
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/large_{:04}_A.mtx", k)));
        let A = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/large_{:04}_B.mtx", k)));
        let B = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        // let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        // let C = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let A = CSR::from_coo(A);
        let B = CSR::from_coo(B);
        // let C = C.to_dense();
        
        // sparse-to-dense
        let start_time = Instant::now();
        let C = A.product(&B);
        let duration= start_time.elapsed();
        times_s2d.push(duration.as_micros());

        // sparse-to-sparse
        let start_time = Instant::now();
        let C = A.product_sparse(&B);
        let duration= start_time.elapsed();
        times_s2s.push(duration.as_micros());

        // sparse-to-sparse parallel
        let start_time = Instant::now();
        let C = A.product_sparse_par(&B);
        let duration= start_time.elapsed();
        times_s2s_par.push(duration.as_micros());

        // C.print();
        // C_test.print();
    }



    println!("Times s2d (us): {:?}", times_s2d);
    println!("Times s2s (us): {:?}", times_s2s);
    println!("Times s2s-par (us): {:?}", times_s2s_par);


}