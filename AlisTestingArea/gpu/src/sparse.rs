
use wgpu::{util::DeviceExt, Buffer};
use wgpu::{BindGroup, BufferUsages, CommandEncoder, Device};

use bytemuck::{Pod, Zeroable};

use matrix_base::{COO, CSR};

pub enum BufferIO {
    Input,
    Output
}



pub struct CSRBuffer {
    pub row_pos: Buffer,
    pub col_pos: Buffer,
    pub values: Buffer,
    pub shape: Buffer,
}


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GlobDataEntry {
    i: u32,
    j: u32, 
    x: f32
}

// #[repr(C)]
// #[derive(Clone, Copy, Pod, Zeroable)]
// pub struct GlobalResult {
//     pub idx: u32,
//     pub data: Vec<GlobDataEntry>,
//  }

pub struct ResultBuffer {
    pub idx: Buffer,
    pub glob_data: Buffer, //GlobalResult,
    pub res_curr_row: Buffer, //Vec<f32>,
    pub nz_row_marker: Buffer, //Vec<bool> eig., aber u32 da bool nicht unterstützt
}


impl ResultBuffer {
    pub fn new(device: &Device, nnz: usize, n: usize, batch_size: usize, name: &str, usage: BufferUsages) -> Self {
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
        let res_curr_row = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Result {}.res_curr_row", name).as_str()),
            size: (n* batch_size *4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });
        let nz_row_marker = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Result {}.nz_row_marker", name).as_str()),
            size: (n* batch_size*4) as u64,
            usage: usage,  // Add COPY_SRC
            mapped_at_creation: false,
        });

        ResultBuffer{idx, glob_data, res_curr_row, nz_row_marker}

    }

    pub fn gen_bind_group_entries(&self, offset: usize, mode: BufferIO) -> Vec<wgpu::BindGroupLayoutEntry> {
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

    pub fn create_bind_group(&self, device: &Device, layout: &wgpu::BindGroupLayout) -> BindGroup {

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Result"),
            layout: layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.glob_data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.res_curr_row.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.nz_row_marker.as_entire_binding() },
            ],
        });

        bind_group
    }

    pub fn copy_b2b(&self, target: &ResultBuffer, nnz: usize, n: usize, batch_size: usize, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.idx, 0, &target.idx, 0, (4) as u64);
        encoder.copy_buffer_to_buffer(&self.glob_data, 0, &target.glob_data, 0, (nnz*3*4) as u64);
        encoder.copy_buffer_to_buffer(&self.res_curr_row, 0, &target.res_curr_row, 0, (n*batch_size*4) as u64);
        encoder.copy_buffer_to_buffer(&self.nz_row_marker, 0, &target.nz_row_marker, 0, (n*batch_size*4) as u64);
    }

}



impl CSRBuffer {
    pub fn new(device: &Device, a: &CSR, name: &str, usage: BufferUsages) -> Self {
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

    pub fn gen_bind_group_entries(&self, offset: usize, mode: BufferIO) -> Vec<wgpu::BindGroupLayoutEntry> {
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