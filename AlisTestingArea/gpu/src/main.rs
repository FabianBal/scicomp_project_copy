use std::path::Path;
use std::process::Command;
use std::result;
use std::sync::Arc;

use gpu_sparse_ali::gpu_sparse_multiplyer::GPUSparseMultiplyer;
use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};
use futures_intrusive::channel::shared::oneshot_channel;

use gpu_sparse_ali::*;



use matrix_base::{COO, CSR};





#[tokio::main]
async fn main() {
    let batch_size = 18;


    let a = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_A.mtx"), true).expect("Failed reading matrix file.");
    let a = CSR::from_coo(a); 
    let b = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_B.mtx"), true).expect("Failed reading matrix file.");
    let b = CSR::from_coo(b);


    let mut gpusm = GPUSparseMultiplyer::new(a, b, batch_size).await;
    gpusm.create_and_load_buffer();
    // let (n_c_data, gd) = gpusm.doit().await;
    let res = gpusm.doit().await;
    // let res = res.to_dense();



    let c = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_C.mtx"), true).expect("Failed reading matrix file.");
    let c = c.to_dense();
    
    for (i,j,x) in res.data {
        // let gd = result[idx];
        println!("({},{}) = {} ({})", i,j,x, c.get(i as usize, j as usize));
    }


    // for idx in 0..(n_c_data as usize) {
    //     let gd = result[idx];
    //     println!("({},{}) = {} ", gd.i, gd.j, gd.x);
    // }


   
}
