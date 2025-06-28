use std::cmp::min;

// WGPU spezifische Imports
use bytemuck::{cast_slice};
use futures_intrusive::channel::shared::oneshot_channel;
use wgpu::{BindGroup, BindGroupLayout, ShaderModule};

use crate::{CSRBuffer, GlobDataEntry, ResultBuffer, WgpuTask, GPUCSR};
use matrix_base::{COO, CSR};
use tools::TimingResult; // Matrix-Typen

pub struct GPUSparseMultiplyer {
    pub wgpu_task: WgpuTask,
    pub a: GPUCSR,
    pub b: GPUCSR,
    pub shader: ShaderModule,
    pub batch_size: usize,
    pub bind_groups: Option<(BindGroup, BindGroup, BindGroup)>,
    pub bind_group_layouts: Option<(BindGroupLayout, BindGroupLayout, BindGroupLayout)>,
    pub buffer_res: Option<ResultBuffer>,
    pub buffer_res_staging: Option<ResultBuffer>,
    pub nnz_pred: usize,
    pub n_disps: usize,
    pub result: Option<(Vec<GlobDataEntry>, usize)>,
}

impl<'a> GPUSparseMultiplyer {
    // Konstruktor: Initialisiert WGPU-Ressourcen (Shader, Pipeline). Misst reine Initialisierungszeit.
    pub async fn new(
        a: &'a CSR,
        b: &'a CSR,
        batch_size: usize,
        wgpu_task_in: WgpuTask,
    ) -> (Self, TimingResult) {
        let start_total_new = std::time::Instant::now();
        let start_init_resources = std::time::Instant::now();

        let device = &wgpu_task_in.device;

        let nnz_pred = size_prediction(&a, &b);
        let n_disps = (a.shape.0 as f64 / batch_size as f64).ceil() as usize;

        // Shader-Code laden und anpassen
        let mut shader_code = match std::fs::read_to_string("gpu/shader/sparse_mul.wgsl") {
            Ok(s) => s,
            Err(_err) => std::fs::read_to_string("shader/sparse_mul.wgsl")
                .expect("Error reading shader file!"),
        };
        shader_code = shader_code.replace("HIERWGANZ", batch_size.to_string().as_str());
        shader_code = shader_code.replace("HIERDIESPALTEN", b.shape.1.to_string().as_str());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Matrix Multiplication Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        // GPUCSR-Strukturen erstellen (konvertiert Daten zu f32/u32, aber noch nicht auf GPU)
        let gpu_a = GPUCSR::new(&a);
        let gpu_b = GPUCSR::new(&b);

        // Ende der reinen Initialisierungszeit
        let initialization_us = start_init_resources.elapsed().as_micros();

        (
            GPUSparseMultiplyer {
                wgpu_task: wgpu_task_in,
                a: gpu_a,
                b: gpu_b,
                shader,
                batch_size,
                bind_groups: None,
                bind_group_layouts: None,
                buffer_res: None,
                buffer_res_staging: None,
                nnz_pred,
                n_disps,
                result: None,
            },
            TimingResult {
                initialization_us,
                h2d_us: 0, // Wird in create_and_load_buffer gemessen
                raw_multiply_us: 0,
                d2h_us: 0,
                cleanup_us: 0,
                total_us: start_total_new.elapsed().as_micros(),
            },
        )
    }

    // Lädt Matrizen auf die GPU und erstellt Bind Groups. Misst H2D-Zeit.
    pub fn create_and_load_buffer(&mut self) -> u128 {
        let start_h2d = std::time::Instant::now(); // H2D-Messung startet hier

        let device = &self.wgpu_task.device;
        let queue = &self.wgpu_task.queue;
        let nnz_pred = self.nnz_pred;

        if nnz_pred == 0 {
            println!("WARNING: Result matrix 0. Skipping.");
            return 0;
        }

        // Buffer erstellen und Daten kopieren (H2D für CSRBuffer::new() durch create_buffer_init)
        let buffer_a = CSRBuffer::new(
            &device,
            &self.a,
            "A",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buffer_b = CSRBuffer::new(
            &device,
            &self.b,
            "B",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // ResultBuffer erstellt Buffer ohne Daten; Initialisierung des Zählers ist H2D.
        let buffer_res = ResultBuffer::new(
            &device,
            nnz_pred,
            self.b.shape.1 as usize,
            "C",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let buffer_res_staging = ResultBuffer::new(
            &device,
            nnz_pred,
            self.b.shape.1 as usize,
            "C",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Zähler des Ergebnis-Puffers auf 0 initialisieren (H2D-Operation)
        queue.write_buffer(&buffer_res.idx, 0, cast_slice(&[0u32]));
        queue.write_buffer(&buffer_res_staging.idx, 0, cast_slice(&[0u32]));

        // Bind group layouts und Bind groups erstellen
        let bg_a_entries = buffer_a.gen_bind_group_entries(0, true);
        let bg_b_entries = buffer_b.gen_bind_group_entries(0, true);
        let bg_res_entries = buffer_res.gen_bind_group_entries(0, false);

        let bg_a_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout: Matrix A"),
            entries: &bg_a_entries,
        });
        let bg_b_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout: Matrix B"),
            entries: &bg_b_entries,
        });
        let bg_res_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout: Result"),
            entries: &bg_res_entries,
        });

        let bg_a = buffer_a.create_bind_group(&device, &bg_a_layout);
        let bg_b = buffer_b.create_bind_group(&device, &bg_b_layout);
        let bg_res = buffer_res.create_bind_group(&device, &bg_res_layout);

        self.bind_groups = Some((bg_a, bg_b, bg_res));
        self.bind_group_layouts = Some((bg_a_layout, bg_b_layout, bg_res_layout));
        self.buffer_res_staging = Some(buffer_res_staging);
        self.buffer_res = Some(buffer_res);

        // Synchronisation nach allen H2D-Operationen
        self.wgpu_task.device.poll(wgpu::Maintain::Wait);
        let h2d_us = start_h2d.elapsed().as_micros();
        h2d_us
    }

    // Führt die Multiplikation auf der GPU durch (Raw Multiply & D2H-Messung).
    pub async fn doit(&mut self) -> (Vec<f32>, TimingResult) {
        let start_total_doit = std::time::Instant::now();
        // Keine `let mut ... = 0;` hier mehr!

        if self.nnz_pred == 0 || self.buffer_res.is_none() || self.bind_groups.is_none() {
            println!("WARNING: Sparse result matrix 0 or buffers not loaded. Skipping.");
            return (Vec::new(), TimingResult::zero());
        }

        let device = &self.wgpu_task.device;
        let queue = &self.wgpu_task.queue;
        let msg = "No Bind group (layout) found";

        // Pipeline Layout und Pipeline erstellen
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[
                &self.bind_group_layouts.as_ref().expect(msg).0,
                &self.bind_group_layouts.as_ref().expect(msg).1,
                &self.bind_group_layouts.as_ref().expect(msg).2,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &self.shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });

        // Zähler des Ergebnis-Puffers auf 0 zurücksetzen (H2D-Operation, Teil des Overheads pro Multiplikation)
        let start_counter_reset = std::time::Instant::now();
        queue.write_buffer(&self.buffer_res.as_ref().unwrap().idx, 0, cast_slice(&[0u32]));
        let doit_initialization_us = start_counter_reset.elapsed().as_micros(); // Jetzt direkt deklariert


        // 3. Reine Multiplikation (Kernel-Ausführung)
        let start_raw_multiply = std::time::Instant::now();
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &self.bind_groups.as_ref().expect(msg).0, &[]);
            compute_pass.set_bind_group(1, &self.bind_groups.as_ref().expect(msg).1, &[]);
            compute_pass.set_bind_group(2, &self.bind_groups.as_ref().expect(msg).2, &[]);
            compute_pass.dispatch_workgroups(self.n_disps as u32, 1, 1);
        }

        // Ergebnis-Puffer zur Staging-Buffer kopieren (D2D - Device-to-Device-Kopie)
        self.buffer_res.as_ref().expect(msg).copy_b2b(&self.buffer_res_staging.as_ref().expect(msg), self.nnz_pred, self.b.shape.1 as usize, &mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait); // Synchronisation nach Submit
        let raw_multiply_us = start_raw_multiply.elapsed().as_micros(); // Jetzt direkt deklariert

        // 4. Device-to-Host (D2H) Kopie des Ergebnisses
        let start_d2h = std::time::Instant::now();
        let buffer_res_staging = self.buffer_res_staging.as_ref().expect(msg);

        // NNZ-Zähler lesen
        let result_buffer_idx_slice = buffer_res_staging.idx.slice(..);
        let (sender_idx, receiver_idx) = oneshot_channel();
        result_buffer_idx_slice.map_async(wgpu::MapMode::Read, move |v| sender_idx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait); // Warten bis Zähler gemappt ist
        receiver_idx.receive().await.unwrap().unwrap();
        let data_result_idx = result_buffer_idx_slice.get_mapped_range();
        let n_c_data = min(cast_slice::<u8, u32>(&data_result_idx)[0] as usize, self.nnz_pred);
        drop(data_result_idx);
        buffer_res_staging.idx.unmap(); // unmap auf originalem Buffer

        // GlobData lesen (Werte, Zeilen-Indizes, Spalten-Indizes)
        let result_buffer_glob_data_slice = buffer_res_staging.glob_data.slice(0..(n_c_data * std::mem::size_of::<GlobDataEntry>()) as u64);
        let (sender_data, receiver_data) = oneshot_channel();
        result_buffer_glob_data_slice.map_async(wgpu::MapMode::Read, move |v| sender_data.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        receiver_data.receive().await.unwrap().unwrap();
        let data_result_glob_data = result_buffer_glob_data_slice.get_mapped_range();
        let result_glob_entries: Vec<GlobDataEntry> = bytemuck::cast_slice(&data_result_glob_data).to_vec();
        drop(data_result_glob_data);
        buffer_res_staging.glob_data.unmap(); // unmap auf originalem Buffer

        let d2h_us = start_d2h.elapsed().as_micros(); // Jetzt direkt deklariert

        // Ergebnis in Self speichern
        self.result = Some((result_glob_entries, n_c_data));

        // 5. Aufräumen (minimaler Overhead)
        let start_cleanup = std::time::Instant::now();
        let cleanup_us = start_cleanup.elapsed().as_micros(); // Jetzt direkt deklariert

        let total_doit_us = start_total_doit.elapsed().as_micros();

        // Ergebnis als Vec<f32> für den Main-Benchmark zurückgeben
        let res_values: Vec<f32> = self.result.as_ref().map_or_else(Vec::new, |(entries, count)| {
            entries.iter().take(*count).map(|e| e.x as f32).collect()
        });

        (res_values, TimingResult {
            initialization_us: doit_initialization_us, // Zähler-Reset Zeit hier!
            h2d_us: 0,
            raw_multiply_us,
            d2h_us,
            cleanup_us,
            total_us: total_doit_us,
        })
    }

    // Bestehende cast_result Funktion
    pub fn cast_result(&self) -> Option<COO> {
        if self.nnz_pred == 0 {
            println!("WARNING: Result matrix 0. Skipping.");
            return None;
        }
        match &self.result {
            Some((r, n_c_data)) => {
                let data_final: Vec<(usize, usize, f64)> = r[..*n_c_data]
                    .into_iter()
                    .map(|entry| (entry.i as usize, entry.j as usize, entry.x as f64))
                    .collect();
                Some(COO {
                    data: data_final,
                    shape: (self.a.shape.0 as usize, self.b.shape.1 as usize),
                })
            }
            None => None,
        }
    }
}

// Hilfsfunktion zur Vorhersage der NNZ (angenommen, sie ist bereits definiert)
pub fn size_prediction(a: &CSR, b: &CSR) -> usize {
    let m = a.shape.0;
    let mut nnzs = vec![];
    for i in 0..m {
        let i0 = a.row_pos[i];
        let i1 = a.row_pos[i + 1];
        let mut nnz_i = 0;
        // Korrektur: Zugriff auf A.col_pos[k_idx]
        for k_idx in i0..i1 {
            let k = a.col_pos[k_idx]; // KORREKTUR: A.col_pos statt A.col_indices
            nnz_i += b.get_row_nnz(k);
        }
        nnzs.push(nnz_i);
    }
    min(nnzs.iter().sum(), m * b.shape.1)
}
