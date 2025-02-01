use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};
use futures_intrusive::channel::shared::oneshot_channel;

use gpu_sparse_ali::WgpuTask;

use matrix_base::{COO, CSR};

// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct Rueber<N: usize> {
//     x: usize,
//     y: &[f32]
// }


enum BufferIO {
    Input,
    Output
}



struct CSRBuffer {
    row_pos: Buffer,
    col_pos: Buffer,
    values: Buffer,
    shape: Buffer,
}


impl CSRBuffer {
    fn new(device: &Device, a: &CSR, name: &str, usage: BufferUsages) -> Self {
        let row_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
            contents: bytemuck::cast_slice(&a.row_pos),
            usage: usage,
        });
        let col_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.col_pos", name).as_str()),
            contents: bytemuck::cast_slice(&a.col_pos),
            usage: usage,
        });
        let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.values", name).as_str()),
            contents: bytemuck::cast_slice(&a.values),
            usage: usage,
        });
        let shape = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.shape", name).as_str()),
            contents: bytemuck::cast_slice(&[a.shape.0, a.shape.1]),
            usage: usage,
        });
    
        CSRBuffer{row_pos, col_pos, values, shape}
    }



    fn new_input(device: &Device, a: &CSR, name: &str) -> Self {
        let row_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
            contents: bytemuck::cast_slice(&a.row_pos),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let col_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.col_pos", name).as_str()),
            contents: bytemuck::cast_slice(&a.col_pos),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.values", name).as_str()),
            contents: bytemuck::cast_slice(&a.values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let shape = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("CSR Matrix {}.shape", name).as_str()),
            contents: bytemuck::cast_slice(&[a.shape.0, a.shape.1]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    
        CSRBuffer{row_pos, col_pos, values, shape}

    }


    fn new_output(device: &Device, a: &CSR, nnz: usize, name: &str, usage: BufferUsages) -> Self {
        // let size = (nnz * 4 * 2 + nnz + 2*4 ) as u64; 


        let row_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("CSR Matrix {}.row_pos", name).as_str()),
            size: (nnz*4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });
        let col_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("CSR Matrix {}.col_pos", name).as_str()),
            size: (nnz*4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });
        let values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("CSR Matrix {}.values", name).as_str()),
            size: (nnz*4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });
        let shape = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("CSR Matrix {}.shape", name).as_str()),
            size: (2*4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });

        CSRBuffer{row_pos, col_pos, values, shape}
    }

    fn gen_bind_group_entries(&self, offset: usize, mode: BufferIO) -> Vec<wgpu::BindGroupLayoutEntry> {
        let mut entries = vec![];

        for i in offset..(offset+4) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                // If input, read-only
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: match mode { BufferIO::Input => true, BufferIO::Output => false} }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
        }

        entries
    }


    fn create_bind_group(&self, device: &Device, layout: &wgpu::BindGroupLayout) -> BindGroup {

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.row_pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.col_pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.shape.as_entire_binding() },
            ],
        });

        bind_group
    }


    fn copy_b2b(&self, target: &CSRBuffer, n: usize, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.row_pos, 0, &target.row_pos, 0, (n*4) as u64);
        encoder.copy_buffer_to_buffer(&self.col_pos, 0, &target.col_pos, 0, (n*4) as u64);
        encoder.copy_buffer_to_buffer(&self.values, 0, &target.values, 0, (n*4) as u64);
        encoder.copy_buffer_to_buffer(&self.shape, 0, &target.shape, 0, (2*4) as u64);
    }


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


    let a = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_A.mtx"), true).expect("Failed reading matrix file.");
    let a = CSR::from_coo(a); 
    let b = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_B.mtx"), true).expect("Failed reading matrix file.");
    let b = CSR::from_coo(b);


    // Create Buffer for CSR Matrix Data 
    let buffer_a = CSRBuffer::new(&device, &a, "A", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let buffer_b = CSRBuffer::new(&device, &a, "B", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);

    let nnz_pred = size_prediction(&a, &b);

    let buffer_c = CSRBuffer::new_output(&device, &a, nnz_pred, "C", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let buffer_c_staging = CSRBuffer::new_output(&device, &a, nnz_pred, "C (staging)", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,);


    // Load Shader

    let shader_code = std::fs::read_to_string("shader/sparse_mul.wgsl").expect("Shader file not found.");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sparse Matrix Multiplication Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });


    // Bind group

    let bg_a_entries = buffer_a.gen_bind_group_entries(0, BufferIO::Input);
    let bg_b_entries = buffer_b.gen_bind_group_entries(0, BufferIO::Input);
    let bg_c_entries = buffer_c.gen_bind_group_entries(0, BufferIO::Output);

    // let bg_entries = vec![bg_a, bg_b, bg_c].concat();

    let bg_a_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout: Matrix A"),
        entries: &bg_a_entries
    });  
    let bg_b_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout: Matrix B"),
        entries: &bg_b_entries
    });  
    let bg_c_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout: Matrix C"),
        entries: &bg_c_entries
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bg_a_layout, &bg_b_layout, &bg_c_layout],
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


    // let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //     label: Some("Bind Group"),
    //     layout: &bg_a_layout,
    //     entries: &[
    //         wgpu::BindGroupEntry { binding: 0, resource: buffer_a.row_pos.as_entire_binding() },
    //         wgpu::BindGroupEntry { binding: 1, resource: buffer_a.col_pos.as_entire_binding() },
    //         wgpu::BindGroupEntry { binding: 2, resource: buffer_a.values.as_entire_binding() },
    //         wgpu::BindGroupEntry { binding: 3, resource: buffer_a.shape.as_entire_binding() },
    //     ],
    // });

    let bg_a = buffer_a.create_bind_group(&device, &bg_a_layout);
    let bg_b = buffer_b.create_bind_group(&device, &bg_b_layout);
    let bg_c = buffer_c.create_bind_group(&device, &bg_c_layout);


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });



    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bg_a, &[]);
        compute_pass.set_bind_group(1, &bg_b, &[]);
        compute_pass.set_bind_group(2, &bg_c, &[]);
        compute_pass.dispatch_workgroups(8,8, 1); // Angepasste Dispatch-Parameter
    }

    queue.submit(Some(encoder.finish()));


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy Encoder") });
    // encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, (size * size * std::mem::size_of::<f32>() as u32) as u64);
    buffer_a.copy_b2b(&buffer_c_staging, nnz_pred, &mut encoder);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = buffer_c_staging.shape.slice(..);
    let (sender, receiver) = oneshot_channel();

    unsafe {
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    }

    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: &[u32] = bytemuck::cast_slice(&data);
    println!("Ergebnis-Matrix: {:?}", result);




   
}
