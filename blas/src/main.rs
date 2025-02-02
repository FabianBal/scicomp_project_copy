use blas_dense::*;
use matrix_base::COO;
use std::path::Path;

fn main() {
    let sp_a = COO::read_mtx(Path::new("../matrix_instances/a_blas.mtx"), false)
        .expect("Failed reading file");
    let sp_b = COO::read_mtx(Path::new("../matrix_instances/b_blas.mtx"), false).expect("Failed reading file");
    let a = BlasDense::from_coo(&sp_a);
    let b = BlasDense::from_coo(&sp_b);
    let result = a.prod(&b);
    
    //assert!(c == vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0,]);
    assert!(result.data == vec![7.0, 8.0, 11.0, 5.0, 4.0, 5.0]);
}
