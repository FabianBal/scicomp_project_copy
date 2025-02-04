
use std::path::Path;

use gpu::sparse::*;
use gpu::WgpuTask;
use matrix_base::{Dense, COO, CSR};

// Im Endeffekt etwas umständlich über Path joinen.
// Kann man auch mit String-Concat machen, aber
// ich wollte Fehler mit / vermeiden
const DATA_PATH: &str = "../matrix_instances";




#[cfg(test)]
fn cmp_float(x: f64, y: f64, eps: f64) -> bool {
    (x-y).abs() < eps
}

#[cfg(test)]
fn cmp_dense(A: &Dense, B: &Dense, eps: f64) -> bool {
    let mut res = true;

    for (x,y) in A.data.iter().zip(B.data.iter()) {
       res = res &&  ( (x-y).abs() < eps );
    }

    res
}



#[tokio::test]
async fn test_wgpu_sparse() {
    let eps = 1e-5;


    let batch_size = 4;

    // Number of matrices to test
    let n = 10;    


    for k in 0..n {
        println!("Testing k={}", k);


        // COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_A.mtx"), true).expect("Failed reading matrix file.");

        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_A.mtx", k)));
        let a = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_B.mtx", k)));
        let b = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");
        let fname = Path::new(DATA_PATH).join(&Path::new(&format!("generated/case_{:04}_C.mtx", k)));
        let c = COO::read_mtx(&fname, true).expect("Failed reading matrix during test");

        let A = CSR::from_coo(&a);
        let B = CSR::from_coo(&b);
        let C = c.to_dense();
        


        let mut gpusm = GPUSparseMultiplyer::new(&A, &B, batch_size, WgpuTask::new(300*1024*1024).await).await;
        gpusm.create_and_load_buffer();
        // let mut res = gpusm.doit().await;        
        gpusm.doit().await;        
        let mut res = gpusm.cast_result().expect("casting result failed");
        let C_test =res.to_dense();


        assert!(cmp_dense(&C, &C_test, eps));


    }


    assert!(true);
}