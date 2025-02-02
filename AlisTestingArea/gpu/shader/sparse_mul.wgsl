struct DataEntry {
    i: u32,
    j: u32,
    x: f32,
};

// HIERDIENUMMERu

@group(0) @binding(0) var<storage, read> a_row_pos: array<u32>;
@group(0) @binding(1) var<storage, read> a_col_pos: array<u32>;
@group(0) @binding(2) var<storage, read> a_values: array<f32>;
@group(0) @binding(3) var<storage, read> a_shape: vec2<u32>;

@group(1) @binding(0) var<storage, read> b_row_pos: array<u32>;
@group(1) @binding(1) var<storage, read> b_col_pos: array<u32>;
@group(1) @binding(2) var<storage, read> b_values: array<f32>;
@group(1) @binding(3) var<storage, read> b_shape: vec2<u32>;

@group(2) @binding(0) var<storage, read_write> idx: atomic<u32>;
@group(2) @binding(1) var<storage, read_write> glob_data: array<DataEntry>;
@group(2) @binding(2) var<storage, read_write> res_curr_row: array<f32>;
@group(2) @binding(3) var<storage, read_write> nz_row_marker: array<u32>;


@compute @workgroup_size(20,1,1)
fn main(@builtin(local_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = a_shape.x;
    let n = b_shape.y;


    let wg_disp_n = HIERDIENUMMERu;
    // let i = id.x + wid.x*wg_disp_n;

    let i = id.x;

    

    for (var col_pos_pos = a_row_pos[i]; col_pos_pos < a_row_pos[i+1]; col_pos_pos++) {
        let k = a_col_pos[col_pos_pos];

        for (var other_col_pos_pos = b_row_pos[k]; other_col_pos_pos < b_row_pos[k+1]; other_col_pos_pos++) {
            let j = b_col_pos[other_col_pos_pos];

            res_curr_row[i*n + j] = res_curr_row[i*n + j] + a_values[col_pos_pos] * b_values[other_col_pos_pos];
            nz_row_marker[i*n + j] = 1;     
        }
    }

    for (var k: u32 = 0; k < n; k++) {
        let x = res_curr_row[i*n + k];
        let marker = nz_row_marker[i*n + k];
        // atomicAdd(&idx, 1u);
        if marker > 0{
            let idx = atomicAdd(&idx, 1u);
            glob_data[idx].i = i;
            glob_data[idx].j = k;
            glob_data[idx].x = x;
        }
    }
}

