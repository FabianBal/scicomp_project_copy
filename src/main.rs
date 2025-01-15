use faksgpu::cpu::COO;

fn main() {
    println!("hi");

    let fname = "data/a01.mm";
    let coo = COO::read_mtx(fname).expect(":(");
    coo.print();


}
