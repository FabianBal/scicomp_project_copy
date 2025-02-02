
pub mod sparse;
pub use sparse::*;
// pub use sparse::ResultBuffer;
// pub use sparse::CSRBuffer;



use wgpu::{util::DeviceExt, Adapter, Device, Instance, Queue};
use futures_intrusive::channel::shared::oneshot_channel;

use matrix_base::{COO, CSR};


pub struct WgpuTask {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue
}



pub fn size_prediction(A: &CSR, B: &CSR) -> usize {
    let m = A.shape.0;

    let mut nnzs = vec![];

    // for (i, col_pos_pos) in A.row_pos.iter().enumerate() {
    for i in 0..m {
        let i0 = A.row_pos[i];
        let i1 = A.row_pos[i+1];

        let mut nnz_i = 0;
        for k in A.col_pos[i0..i1].iter() {
            nnz_i += B.get_row_nnz(*k);
        }
        nnzs.push(nnz_i);
    }


    nnzs.iter().sum()
}






impl WgpuTask {
    pub async fn new() -> Self {

        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();


        WgpuTask{instance, adapter, device, queue}
    }
}


