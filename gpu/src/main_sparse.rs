use std::path::Path;
use std::process::Command;
use std::result;
use std::sync::Arc;

use futures_intrusive::channel::shared::{self, oneshot_channel};
use gpu::gpu_sparse_multiplyer::GPUSparseMultiplyer;
use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};

use gpu::*;

use matrix_base::{COO, CSR};

#[tokio::main]
async fn main() {
    let batch_size = 4;

    // let a = COO::read_mtx(Path::new("matrix_instances/generated/case_0001_A.mtx"), true).expect("Failed reading matrix file.");
    // let a = COO::read_mtx(Path::new("matrix_instances/generated/sparse/sparse_100_A.mtx"), true).expect("Failed reading matrix file.");
    let a = COO::read_mtx(
        Path::new("matrix_instances/generated/sparse_vs_dense/sparse_vs_dense_0.1_A.mtx"),
        true,
    )
    .expect("Failed reading matrix file.");
    let a = CSR::from_coo(&a);
    // let b = COO::read_mtx(Path::new("matrix_instances/generated/case_0001_B.mtx"), true).expect("Failed reading matrix file.");
    let b = COO::read_mtx(
        Path::new("matrix_instances/generated/sparse/sparse_100_B.mtx"),
        true,
    )
    .expect("Failed reading matrix file.");
    let b = COO::read_mtx(
        Path::new("matrix_instances/generated/sparse_vs_dense/sparse_vs_dense_0.1_B.mtx"),
        true,
    )
    .expect("Failed reading matrix file.");
    let b = CSR::from_coo(&b);

    let mut gpusm =
        GPUSparseMultiplyer::new(&a, &b, batch_size, WgpuTask::new(300 * 1024 * 1024).await).await;
    gpusm.create_and_load_buffer();
    // let (n_c_data, gd) = gpusm.doit().await;
    // let mut res = gpusm.doit().await;
    let nnz_pred = gpusm.nnz_pred;
    println!("Size pred: {}", nnz_pred);
    gpusm.doit().await;
    let mut res = gpusm.cast_result().expect("Casting result failed");
    // let res = res.to_dense();
    // res.sort_data();

    let c = COO::read_mtx(
        Path::new("matrix_instances/generated/case_0001_C.mtx"),
        true,
    )
    .expect("Failed reading matrix file.");

    println!("bbb {} {}", res.data.len(), c.data.len());
    println!("ccc {:?} {:?}", res.shape, c.shape);

    let c = c.to_dense();

    // for (c1, c2) in c.data.iter().enumerate() {

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
            if (res.get(i, j) - c.get(i, j)).abs() > 1e-5 {
                println!("UAU ({} {}): {} {}", i, j, res.get(i, j), c.get(i, j));
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
