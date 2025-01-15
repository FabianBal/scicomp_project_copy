use faksgpu::cpu::{COO, CSR};

fn main() {
    println!("hi");

    let fname = "data/a01.mm";
    let fname = "data/a01.mm";
    let coo = COO::read_mtx(fname).expect(":(");
    coo.print();

    let csr = CSR::from_coo(coo);
    csr.print();

    let res = csr.product(&csr);

    for row in res {
        for x in row {
            print!("{}\t",x);
        }
        println!();
    }


}
