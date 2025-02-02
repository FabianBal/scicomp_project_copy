use std::path::Path;

use fakscpu::sparse::SparseProd;
use matrix_base::{Dense, COO, CSR};

// use matrix_base::HAHA;

// trait yea {

//     fn yo(&self);

// }

// impl yea for HAHA {
//     fn yo (&self) {
//         self.x * 4;
//     }
// }





fn main() {
    println!("hi");

    // let fname = Path::new("../matrix_instances/a001.mtx");
    // let coo = COO::read_mtx(fname, true).expect(":(");
    // coo.print();
    // coo.to_dense().print();

    // let csr = CSR::from_coo(coo);
    // csr.print();

    // let res = csr.product(&csr);
    // res.print();

    // let res2 = csr.product_sparse_par(&csr);
    // let res2d = res2.to_dense();
    // res2.print();
    // res2d.print();

    // for k in 0..csr.shape.0 {
    //     println!("nnz({}) = {}", k, csr.get_row_nnz(k));
    // }



    let fname = Path::new("../matrix_instances/generated/case_0000_A.mtx");
    let a = COO::read_mtx(fname, true).expect(":(");

    // a.to_dense().print();

    let fname = Path::new("../matrix_instances/generated/case_0000_B.mtx");
    let b = COO::read_mtx(fname, true).expect(":(");
    
    let a = CSR::from_coo(a);
    let b = CSR::from_coo(b);
    let rescoo = a.product_sparse_to_coo_par(&b);

    println!("{:?}", b.row_pos);

    // rescoo.to_dense().print();

    // println!("{:?}", rescoo.to_dense().data);

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


}
