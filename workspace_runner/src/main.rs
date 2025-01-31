use std::path::Path;

use matrix_base::{COO, CSR, Dense};
fn main() {
    println!("Hello, world! workspace_runner");
    let fname = Path::new("./matrix_instances/a001.mtx");
    let coo = COO::read_mtx(fname, false).expect("reading matrix failed");

    let csr = CSR::from_coo(coo);

    cublas::multiply(&csr, &csr);
    cusparse::multiply(&csr, &csr);
}
