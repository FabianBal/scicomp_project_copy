
use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};

use bytemuck::{Pod, Zeroable};

use matrix_base::{COO, CSR};


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GlobDataEntry {
    pub i: u32,
    pub j: u32, 
    pub x: f32
}



pub struct ResultBuffer {
    pub idx: Buffer,
    pub glob_data: Buffer, // GlobalResult,
}


impl ResultBuffer {
    pub fn new(device: &Device, nnz: usize, n: usize, name: &str, usage: BufferUsages) -> Self {
        let idx = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Result {}.idx", name).as_str()),
            size: (4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        }); 
        let glob_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Result {}.glob_data", name).as_str()),
            size: (nnz*3*4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });

        ResultBuffer{idx, glob_data}

    }

    pub fn gen_bind_group_entries(&self, offset: usize, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
        let mut entries = vec![];

        for i in offset..(offset+2) {
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
            label: Some("Bind Group Result"),
            layout: layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.glob_data.as_entire_binding() },
            ],
        });

        bind_group
    }

    pub fn copy_b2b(&self, target: &ResultBuffer, nnz: usize, n: usize, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.idx, 0, &target.idx, 0, (4) as u64);
        encoder.copy_buffer_to_buffer(&self.glob_data, 0, &target.glob_data, 0, (nnz*3*4) as u64);
    }

}

