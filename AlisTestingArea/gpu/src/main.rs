use std::path::Path;
use std::process::Command;
use std::result;
use std::sync::Arc;

use gpu_sparse_ali::gpu_sparse_multiplyer::GPUSparseMultiplyer;
use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};
use futures_intrusive::channel::shared::{self, oneshot_channel};

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
    let mut res = gpusm.doit().await;
    // let res = res.to_dense();
    // res.sort_data();



    let c = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_C.mtx"), true).expect("Failed reading matrix file.");

    println!("bbb {} {}", res.data.len(), c.data.len());

    let c = c.to_dense();
    
    
    // for (i,j,x) in res.data {
    //     // let gd = result[idx];
    //     // println!("({},{}) = {} ({})", i,j,x, c.get(i as usize, j as usize));
    //     if (x as f64 - c.get(i,j)).abs() > 1e-5 {
    //         println!("AAAAA  ({},{}) = {} ({})", i,j,x, c.get(i as usize, j as usize));
    //     }
        
    // }

    let res = res.to_dense();
    for i in 0..c.shape.0 {
        for j in 0..c.shape.1 {
            if  (res.get(i,j) - c.get(i, j)).abs() > 1e-5 {

                println!("UAU ({} {}): {} {}", i,j, res.get(i,j), c.get(i, j));
            }
        }
    }



    // for idx in 0..(n_c_data as usize) {
    //     let gd = result[idx];
    //     println!("({},{}) = {} ", gd.i, gd.j, gd.x);
    // }

    // let mut shader_code = std::fs::read_to_string("shader/sparse_mul.wgsl").expect("Shader file not found.");
    // shader_code = shader_code.replace("HIERDIENUMMER", 4.to_string().as_str());
    // println!("{}", shader_code);

   
}
