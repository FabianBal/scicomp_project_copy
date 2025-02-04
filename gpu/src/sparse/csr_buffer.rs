
use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};

use bytemuck::{Pod, Zeroable};

use matrix_base::{COO, CSR};


pub struct GPUCSR {
    pub row_pos: Vec<u32>,
    pub col_pos: Vec<u32>,
    pub values: Vec<f32>,
    pub shape: (u32, u32)
}


pub struct CSRBuffer {
    pub row_pos: Buffer,
    pub col_pos: Buffer,
    pub values: Buffer,
    pub shape: Buffer,
}


impl GPUCSR {
    pub fn new (a: &CSR) -> Self  {
        let row_pos: Vec<u32> = a.row_pos.iter().map(|i| (*i as u32) ).collect();
        let col_pos: Vec<u32> = a.col_pos.iter().map(|j| (*j as u32) ).collect();
        let values: Vec<f32> = a.values.iter().map(|x| (*x as f32) ).collect();

        GPUCSR{row_pos, col_pos, values, shape: (a.shape.0 as u32, a.shape.1 as u32) }
    }
}



impl CSRBuffer {
    pub fn new(device: &Device, a: &GPUCSR, name: &str, usage: BufferUsages) -> Self {
        // let row_pos: Vec<u32> = a.row_pos.iter().map(|i| (*i as u32) ).collect();
        // let col_pos: Vec<u32> = a.col_pos.iter().map(|j| (*j as u32) ).collect();
        // let values: Vec<f32> = a.values.iter().map(|x| (*x as f32) ).collect();


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
            contents: bytemuck::cast_slice(&[a.shape.0 as u32, a.shape.1 as u32]),
            usage: usage,
        });
    
        CSRBuffer{row_pos, col_pos, values, shape}
    }


    pub fn new_output(device: &Device, nnz: usize, name: &str, usage: BufferUsages) -> Self {
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

    pub fn gen_bind_group_entries(&self, offset: usize, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
        let mut entries = vec![];

        for i in offset..(offset+4) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                // If input, read-only
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
        }

        entries
    }


    pub fn create_bind_group(&self, device: &Device, layout: &wgpu::BindGroupLayout) -> BindGroup {

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


    pub fn copy_b2b(&self, target: &CSRBuffer, n: usize, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.row_pos, 0, &target.row_pos, 0, (n*4) as u64);
        encoder.copy_buffer_to_buffer(&self.col_pos, 0, &target.col_pos, 0, (n*4) as u64);
        encoder.copy_buffer_to_buffer(&self.values, 0, &target.values, 0, (n*4) as u64);
        encoder.copy_buffer_to_buffer(&self.shape, 0, &target.shape, 0, (2*4) as u64);
    }


}