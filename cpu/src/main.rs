use std::path::Path;

use fakscpu::cpu::{COO, CSR, Dense};

fn main() {
    println!("hi");

    let fname = Path::new("../matrix_instances/a001.mtx");
    let coo = COO::read_mtx(fname, false).expect(":(");
    coo.print();
    coo.to_dense().print();

    let csr = CSR::from_coo(coo);
    csr.print();

    let res = csr.product(&csr);
    res.print();

    let res2 = csr.product_sparse_par(&csr);
    let res2d = res2.to_dense();
    res2.print();
    res2d.print();

    for k in 0..csr.shape.0 {
        println!("nnz({}) = {}", k, csr.get_row_nnz(k));
    }


    // let csr_d = csr.to_dense();
    // csr_d.print();

    // for row in res {
    //     for x in row {
    //         print!("{}\t",x);
    //     }
    //     println!();
    // }

    // let d = Dense::new_zeros((3,3));
    // d.print();

    // let a002 = COO::read_mtx("matrix_instances/generated/case_0000_A.mtx").expect(":(");
    // let mut a002 = COO::read_mtx(Path::new("../matrix_instances/generated/case_0000_A.mtx"), true).expect(":(");
    // a002.to_dense().print();

    // a002.data.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    // println!("{:?}", a002.data);







    println!("************************");
    let DATA_PATH: &str = "../matrix_instances";


    let k= 9;

    let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
    let A = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
    let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
    let B = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
    let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
    let C = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

    let A = CSR::from_coo(A);
    let B = CSR::from_coo(B);
    let C = C.to_dense();
    

    C.print();
    

    let C_test = A.product_sparse_par(&B);
    // let C_test = C_test.to_dense();







}
