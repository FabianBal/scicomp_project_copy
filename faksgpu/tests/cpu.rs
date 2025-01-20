use std::path::Path;

use faksgpu::cpu::*;

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";




#[cfg(test)]
fn cmp_float(x: f64, y: f64, eps: f64) -> bool {
    x-y < eps
}


#[test]
fn test_read_coo() {
    let eps = 1e-10;

    // a001.mtx
    let fname1 = Path::new(DATA_PATH).join(&Path::new("a001.mtx"));
    let coo = COO::read_mtx(&fname1).expect("Failed reading matrix during test");

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
    let coo = COO::read_mtx(&fname2).expect("Failed reading matrix during test");

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
    // TODO
}

