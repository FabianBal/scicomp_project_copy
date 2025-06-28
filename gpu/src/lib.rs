use std::cmp::min;

pub mod sparse;
pub use sparse::*;
// pub use sparse::ResultBuffer;
// pub use sparse::CSRBuffer;

pub mod dense;
// pub use dense;

use matrix_base::{CSR};
use wgpu::{Adapter, Device, Instance, Queue};

#[derive(Clone)]
pub struct WgpuTask {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

pub fn size_prediction(a: &CSR, b: &CSR) -> usize {
    let m = a.shape.0;

    let mut nnzs = vec![];

    // for (i, col_pos_pos) in A.row_pos.iter().enumerate() {
    for i in 0..m {
        let i0 = a.row_pos[i];
        let i1 = a.row_pos[i + 1];

        let mut nnz_i = 0;
        for k in a.col_pos[i0..i1].iter() {
            nnz_i += b.get_row_nnz(*k);
        }
        nnzs.push(nnz_i);
    }

    min(nnzs.iter().sum(), m * b.shape.1)
}

impl WgpuTask {
    pub async fn new(limit: u64) -> Self {
        let limits = wgpu::Limits {
            //max_storage_buffer_binding_size: limit, // 1 GB
            max_buffer_size: limit,
            ..wgpu::Limits::default() // Andere Limits beibehalten
        };
        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        };
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&device_descriptor, None)
            .await
            .unwrap();
        //println!("device-limts: {:#?}", device.limits());

        WgpuTask {
            instance,
            adapter,
            device,
            queue,
        }
    }
}
