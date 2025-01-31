use std::path::Path;

use wgpu::{util::DeviceExt, Buffer};
use wgpu::Device;
use futures_intrusive::channel::shared::oneshot_channel;

use gpu::WgpuTask;

use matrix_base::{COO, CSR};

// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct Rueber<N: usize> {
//     x: usize,
//     y: &[f32]
// }

struct CSRBuffer {
    row_pos: Buffer,
    col_pos: Buffer,
    values: Buffer,
    shape: Buffer,
}


fn create_buffer_for_csr(device: &Device, a: &CSR, name: &str)  -> CSRBuffer {
    let row_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
        contents: bytemuck::cast_slice(&a.row_pos),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let col_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
        contents: bytemuck::cast_slice(&a.col_pos),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
        contents: bytemuck::cast_slice(&a.values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let shape = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
        contents: bytemuck::cast_slice(&[a.shape.0, a.shape.1]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    CSRBuffer{row_pos, col_pos, values, shape}
}


fn size_prediction(A: &CSR, B: &CSR) -> usize {
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



#[tokio::main]
async fn main() {
    // let instance = wgpu::Instance::default();
    // let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    // let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

    let wgpu_task = WgpuTask::new().await;
    let instance= wgpu_task.instance;
    let adapter= wgpu_task.adapter;
    let device= wgpu_task.device;
    let queue= wgpu_task.queue;


    let a = COO::read_mtx(Path::new("../matrix_instances/generated/case_0000_A.mtx"), true).expect("Failed reading matrix file.");
    let a = CSR::from_coo(a); 
    let b = COO::read_mtx(Path::new("../matrix_instances/generated/case_0000_B.mtx"), true).expect("Failed reading matrix file.");
    let b = CSR::from_coo(b);

    let buffer_a = create_buffer_for_csr(&device, &a, "A");
    let buffer_b = create_buffer_for_csr(&device, &a, "B");

    // *** Create Buffer for CSR Matrix Data ***
    // A



    // let zz = Rueber{x: 5, y: vec![1., 2., 3.]};
    // let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: Some("Matrix B"),
    //     contents: bytemuck::cast_slice(&[zz]),
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });



    let size: u32 = 4;
    let matrix_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    //let matrix_b: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let matrix_b: Vec<f32> = vec![1.0; 16];
    println!("Matrix-a: {:?}",matrix_a);
    println!("Matrix-b: {:?}",matrix_b);

    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A"),
        contents: bytemuck::cast_slice(&matrix_a),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B"),
        contents: bytemuck::cast_slice(&matrix_b),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Create buffer_c for storage, copying, and as a source for copying
    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: (size * size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,  // Add COPY_SRC
        mapped_at_creation: false,
    });

    // Create a staging buffer for reading
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (size * size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,  // Add COPY_DST for copying from buffer_c
        mapped_at_creation: false,
    });

    let shader_code = std::fs::read_to_string("shader/sparse_mul.wgsl").expect("Shader file not found.");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sparse Matrix Multiplication Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });


    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_c.as_entire_binding() },
        ],
    });




    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });



    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((size + 7) / 8, (size + 7) / 8, 1); // Angepasste Dispatch-Parameter
    }

    queue.submit(Some(encoder.finish()));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy Encoder") });
    encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, (size * size * std::mem::size_of::<f32>() as u32) as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = oneshot_channel();

    unsafe {
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    }

    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);
    println!("Ergebnis-Matrix: {:?}", result);
}
