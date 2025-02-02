extern crate blas;
extern crate openblas_src;

use blas::*;
use matrix_base::{Dense, COO};
use std::path::Path;

fn to_dense_transpose(matrix: &COO) -> Dense {
    let mut mat = Dense::new_zeros((matrix.shape.1, matrix.shape.0));
    for (i, j, x) in &matrix.data {
        mat.set(*j, *i, *x);
    }
    mat
}

fn main() {
    let sp_a = COO::read_mtx(Path::new("../matrix_instances/a002.mtx"), false)
        .expect("Failed reading file");
    println!("{:?}", sp_a.shape);
    let b = sp_a.to_dense().data;
    let a = to_dense_transpose(&sp_a).data;
    println!("{:?}", a);

    let (m, n, k) = (
        sp_a.shape.0 as i32,
        sp_a.shape.0 as i32,
        sp_a.shape.1 as i32,
    );
    //let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    //let b = vec![ 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0, ];
    //let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

    let mut c = vec![0.0; sp_a.shape.0 * sp_a.shape.0];

    unsafe {
        dgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
    }

    //assert!(c == vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0,]);
    assert!(c == vec![625.0, 375.0, 375.0, 549.0]);
}
