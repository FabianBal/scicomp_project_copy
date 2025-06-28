use futures_intrusive::channel::shared::oneshot_channel;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    // read Matrix-Data
    let (row_size_a, col_size_a, matrix_a) =
        read_mtx_file(Path::new("../matrix_instances/generated/matrix_3x4_A.mtx"));
    println!("Matrix-a eingelesen");
    let (row_size_b, col_size_b, matrix_b) =
        read_mtx_file(Path::new("../matrix_instances/generated/matrix_4x2_B.mtx"));
    println!("Matrix-b eingelesen");
    let (_row_size_c, _col_size_c, matrix_c) =
        read_mtx_file(Path::new("../matrix_instances/generated/matrix_3x2_C.mtx"));
    println!("Matrix-c eingelesen");

    // create WGPU-Instanz, Adapter und Device
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    // create Buffer
    let (buffer_a, buffer_b, buffer_c, staging_buffer, matrix_a_size_buffer, matrix_b_size_buffer) =
        create_buffers(
            &device, &matrix_a, &matrix_b, row_size_a, col_size_a, row_size_b, col_size_b,
        );

    // calculate matrix/matrix
    let result = compute_matrix(
        &device,
        &queue,
        buffer_a,
        buffer_b,
        buffer_c,
        staging_buffer,
        matrix_a_size_buffer,
        matrix_b_size_buffer,
        row_size_a,
        col_size_b,
    )
    .await;

    // control result
    let are_equal = compare_matrices(&matrix_c, &result);
    if are_equal {
        println!("correct result");
    } else {
        println!("wrong result");
    }
}

fn create_buffers(
    device: &wgpu::Device,
    matrix_a: &[f32],
    matrix_b: &[f32],
    row_size_a: u32,
    col_size_a: u32,
    row_size_b: u32,
    col_size_b: u32,
) -> (
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
) {
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A"),
        contents: bytemuck::cast_slice(matrix_a),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B"),
        contents: bytemuck::cast_slice(matrix_b),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: (row_size_a * col_size_b * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (row_size_a * col_size_b * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let matrix_a_size = [row_size_a as u32, col_size_a as u32];
    let matrix_b_size = [row_size_b as u32, col_size_b as u32];

    let matrix_a_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A Size"),
        contents: bytemuck::cast_slice(&matrix_a_size),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let matrix_b_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B Size"),
        contents: bytemuck::cast_slice(&matrix_b_size),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    (
        buffer_a,
        buffer_b,
        buffer_c,
        staging_buffer,
        matrix_a_size_buffer,
        matrix_b_size_buffer,
    )
}

async fn compute_matrix(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    buffer_c: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    matrix_a_size_buffer: wgpu::Buffer,
    matrix_b_size_buffer: wgpu::Buffer,
    row_size_a: u32,
    col_size_b: u32,
) -> Vec<f32> {
    let shader_code =
        std::fs::read_to_string("./shader/matrix_mult.wgsl").expect("Shader-Datei nicht gefunden");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Matrix Multiplication Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_c.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: matrix_a_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: matrix_b_size_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroup_size = 8;
    let dispatch_x = (col_size_b + workgroup_size - 1) / workgroup_size;
    let dispatch_y = (row_size_a + workgroup_size - 1) / workgroup_size;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    queue.submit(Some(encoder.finish()));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });
    encoder.copy_buffer_to_buffer(
        &buffer_c,
        0,
        &staging_buffer,
        0,
        (row_size_a * col_size_b * std::mem::size_of::<f32>() as u32) as u64,
    );
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
    result.to_vec()
}

fn read_mtx_file<P: AsRef<Path>>(path: P) -> (u32, u32, Vec<f32>) {
    let file = File::open(path).expect("Datei konnte nicht geöffnet werden");
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    let header = lines
        .next()
        .expect("Datei ist leer")
        .expect("Fehler beim Lesen der ersten Zeile");
    let header_parts: Vec<&str> = header.split_whitespace().collect();
    assert_eq!(header_parts.len(), 3, "Header hat falsches Format");

    let rows: u32 = header_parts[0]
        .parse()
        .expect("Fehler beim Parsen der Zeilenanzahl");
    let cols: u32 = header_parts[1]
        .parse()
        .expect("Fehler beim Parsen der Spaltenanzahl");
    let _num_entries: usize = header_parts[2]
        .parse()
        .expect("Fehler beim Parsen der Eintragsanzahl");

    let mut matrix = vec![0.0; (rows * cols) as usize];

    for line in lines {
        let line = line.expect("Fehler beim Lesen einer Zeile");
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts.len(), 3, "Zeilenformat ist ungültig");

        let row: u32 = parts[0]
            .parse()
            .expect("Fehler beim Parsen der Zeilennummer");
        let col: u32 = parts[1]
            .parse()
            .expect("Fehler beim Parsen der Spaltennummer");
        let value: f32 = parts[2].parse().expect("Fehler beim Parsen des Werts");

        assert!(
            row > 0 && row <= rows,
            "Zeilenindex außerhalb des gültigen Bereichs"
        );
        assert!(
            col > 0 && col <= cols,
            "Spaltenindex außerhalb des gültigen Bereichs"
        );

        matrix[((row - 1) * cols + (col - 1)) as usize] = value;
    }

    (rows, cols, matrix)
}

fn compare_matrices(matrix_a: &[f32], matrix_b: &[f32]) -> bool {
    if matrix_a.len() != matrix_b.len() {
        return false;
    }

    for i in 0..matrix_a.len() {
        if (matrix_a[i] - matrix_b[i]).abs() > f32::EPSILON {
            return false;
        }
    }

    true
}
