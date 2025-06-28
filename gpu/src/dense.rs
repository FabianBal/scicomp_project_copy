use bytemuck::cast_slice;
use futures_intrusive::channel::shared::oneshot_channel;
use matrix_base::Dense;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use tools::TimingResult;
use wgpu::util::DeviceExt;
use pollster::block_on;

pub fn multiply_for_benchmark(
    matrix1: &Dense,
    matrix2: &Dense,
    limit: u64,
) -> (Vec<f32>, TimingResult) {
    let start_total = std::time::Instant::now();

    // 1. Initialisierung von WGPU
    let start_init = std::time::Instant::now();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
        .unwrap();
    let limits = wgpu::Limits {
        //max_storage_buffer_binding_size: limit, // 1 GB
        max_buffer_size: limit ,
        ..wgpu::Limits::default() // Andere Limits beibehalten
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
        .unwrap();
    let initialization_us = start_init.elapsed().as_micros();

    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
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
        // Binding 2 wird der GPU-Ergebnis-Buffer (STORAGE) sein
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
    ];

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &bind_group_layout_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader_module = device.create_shader_module(wgpu::include_wgsl!("../shader/matrix_mult.wgsl"));

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // 2. Host-to-Device (H2D) Datenübertragung
    let start_h2d = std::time::Instant::now();
    let size_of_matrix1 = (std::mem::size_of::<f32>() * matrix1.shape.0 * matrix1.shape.1) as u64;
    let size_of_matrix2 = (std::mem::size_of::<f32>() * matrix2.shape.0 * matrix2.shape.1) as u64;
    let size_of_result = (std::mem::size_of::<f32>() * matrix1.shape.0 * matrix2.shape.1) as u64;
    let size_of_dims = (std::mem::size_of::<u32>() * 3) as u64;

    let matrix1_data_f32: Vec<f32> = matrix1.data.iter().map(|&x| x as f32).collect();
    let matrix2_data_f32: Vec<f32> = matrix2.data.iter().map(|&x| x as f32).collect();

    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix A Buffer"), // Add labels for easier debugging
        size: size_of_matrix1,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix B Buffer"),
        size: size_of_matrix2,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // NEU: Trennung in Compute-Result-Buffer und Staging-Buffer
    let buffer_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Compute Result Buffer C"),
        size: size_of_result,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, // STORAGE für Compute, COPY_SRC zum Kopieren
        mapped_at_creation: false,
    });
    let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer for Result C"),
        size: size_of_result,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, // MAP_READ für CPU, COPY_DST als Ziel
        mapped_at_creation: false,
    });

    let buffer_dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dimensions Buffer"),
        size: size_of_dims,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer_a, 0, cast_slice(&matrix1_data_f32));
    queue.write_buffer(&buffer_b, 0, cast_slice(&matrix2_data_f32));
    queue.write_buffer(
        &buffer_dims,
        0,
        cast_slice(&[matrix1.shape.0 as u32, matrix1.shape.1 as u32, matrix2.shape.1 as u32]),
    );
    device.poll(wgpu::Maintain::Wait); // Synchronisation nach H2D
    let h2d_us = start_h2d.elapsed().as_micros();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dense Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(buffer_a.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(buffer_b.as_entire_buffer_binding()),
            },
            // Binding 2 ist jetzt der GPU-Ergebnis-Buffer (`buffer_result`)
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(buffer_result.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(buffer_dims.as_entire_buffer_binding()),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });

    // 3. Reine Multiplikation (Kernel-Ausführung)
    let start_raw_multiply = std::time::Instant::now();
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass for Dense MM"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            (matrix1.shape.0 as u32 + 7) / 8,
            (matrix2.shape.1 as u32 + 7) / 8,
            1,
        );
    }
    //Kopiere das Ergebnis vom GPU-Puffer in den Staging-Puffer
    encoder.copy_buffer_to_buffer(&buffer_result, 0, &buffer_staging, 0, size_of_result);

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait); // Synchronisation nach Submit und Kopie
    let raw_multiply_us = start_raw_multiply.elapsed().as_micros();

    // 4. Device-to-Host (D2H) Datenübertragung vom Staging-Puffer
    let start_d2h = std::time::Instant::now();
    let result_slice = buffer_staging.slice(0..size_of_result);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    result_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait); // Diese Zeile stellt sicher, dass der Callback ausgeführt wird.

    let _ = block_on(receiver.receive()).unwrap().unwrap();

    let data = result_slice.get_mapped_range();
    let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer_staging.unmap();

    let d2h_us = start_d2h.elapsed().as_micros();

    // 5. Aufräumen (minimaler Overhead)
    let start_cleanup = std::time::Instant::now();
    let cleanup_us = start_cleanup.elapsed().as_micros();

    let total_us = start_total.elapsed().as_micros();

    (result_vec, TimingResult {
        initialization_us,
        h2d_us,
        raw_multiply_us,
        d2h_us,
        cleanup_us,
        total_us,
    })
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
    let shader_code = std::fs::read_to_string("./gpu/shader/matrix_mult.wgsl")
        .expect("Shader-Datei nicht gefunden");
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

    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());


    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);
    result.to_vec()
}
#[allow(dead_code)]
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
#[allow(dead_code)]
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
