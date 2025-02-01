use std::path::Path;
use std::process::Command;
use std::result;
use std::sync::Arc;

use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};
use futures_intrusive::channel::shared::oneshot_channel;

use gpu_sparse_ali::*;



use matrix_base::{COO, CSR};


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
    let batch_size = 18;


    

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
    let buffer_b = CSRBuffer::new(&device, &b, "B", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);

    let nnz_pred = size_prediction(&a, &b);

    // let buffer_c = CSRBuffer::new_output(&device, nnz_pred, "C", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    // let buffer_c_staging = CSRBuffer::new_output(&device, nnz_pred, "C (staging)", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,);


    let buffer_res = ResultBuffer::new(&device, nnz_pred, b.shape.1, batch_size, "C", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let buffer_res_staging = ResultBuffer::new(&device, nnz_pred, b.shape.1, batch_size, "C", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

    // Load Shader

    let shader_code = std::fs::read_to_string("shader/sparse_mul.wgsl").expect("Shader file not found.");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sparse Matrix Multiplication Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });


    // Bind group

    let bg_a_entries = buffer_a.gen_bind_group_entries(0, true);
    let bg_b_entries = buffer_b.gen_bind_group_entries(0, true);
    // let bg_c_entries = buffer_c.gen_bind_group_entries(0, BufferIO::Output);

    let bg_res_entries = buffer_res.gen_bind_group_entries(0, false);

    // let bg_entries = vec![bg_a, bg_b, bg_c].concat();

    let bg_a_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout: Matrix A"),
        entries: &bg_a_entries
    });  
    let bg_b_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout: Matrix B"),
        entries: &bg_b_entries
    });  
    // let bg_c_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //     label: Some("Bind Group Layout: Matrix C"),
    //     entries: &bg_c_entries
    // });

    let bg_res_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout: Result"),
        entries: &bg_res_entries
    });


    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bg_a_layout, &bg_b_layout, &bg_res_layout],
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
    // let bg_c = buffer_c.create_bind_group(&device, &bg_c_layout);

    let bg_res = buffer_res.create_bind_group(&device, &bg_res_layout);


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });



    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bg_a, &[]);
        compute_pass.set_bind_group(1, &bg_b, &[]);
        // compute_pass.set_bind_group(2, &bg_c, &[]);
        compute_pass.set_bind_group(2, &bg_res, &[]);
        compute_pass.dispatch_workgroups(1,1, 1); // Angepasste Dispatch-Parameter
    }

    // queue.submit(Some(encoder.finish()));


    // let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy Encoder") });


    // encoder.copy_buffer_to_buffer(&buffer_res.shape, 0, &buffer_c_staging.shape, 0, 8 as u64);
    // encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, (size * size * std::mem::size_of::<f32>() as u32) as u64);
    buffer_res.copy_b2b(&buffer_res_staging, nnz_pred, b.shape.1, batch_size,&mut encoder);
    queue.submit(Some(encoder.finish()));

    let result_buffer_idx = buffer_res_staging.idx.slice(..);
    let result_buffer_glob_data = buffer_res_staging.glob_data.slice(..);


    let (sender, receiver_idx) = oneshot_channel();
    unsafe {
        result_buffer_idx.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    } 
    

    let (sender, receiver_data) = oneshot_channel();
    unsafe {        
        result_buffer_glob_data.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    }


    device.poll(wgpu::Maintain::Wait);
    receiver_idx.receive().await.unwrap().unwrap();
    receiver_data.receive().await.unwrap().unwrap();

    let data_result_idx = result_buffer_idx.get_mapped_range();
    let result: &[u32] = bytemuck::cast_slice(&data_result_idx);
    let n_c_data = result[0];
    println!("Ergebnis-Matrix: {:?}", n_c_data);


    let data_result_idx = result_buffer_glob_data.get_mapped_range();
    let result: &[GlobDataEntry] = bytemuck::cast_slice(&data_result_idx);

    let gd = result[0];

    // println!("Ergebnis-Matrix: {:?}", result[0]);
    // println!("Ergebnis-Matrix: {:?}", result[0].x);
    


    // println!("A.row_pos = {:?}", a.row_pos);
    // println!("A.row_pos = {:?}", a.col_pos);
    // println!("A.row_pos = {:?}", a.values);


    let c = COO::read_mtx(Path::new("../../matrix_instances/generated/case_0000_C.mtx"), true).expect("Failed reading matrix file.");
    let c = c.to_dense();
    
    for idx in 0..(n_c_data as usize) {
        let gd = result[idx];
        println!("({},{}) = {} ({})", gd.i, gd.j, gd.x, c.get(gd.i as usize, gd.j as usize));
    }


    // for idx in 0..(n_c_data as usize) {
    //     let gd = result[idx];
    //     println!("({},{}) = {} ", gd.i, gd.j, gd.x);
    // }


   
}
