struct DataEntry {
    i: u32,
    j: u32,
    x: f32,
};


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


// For algorithm see CPU sparse implementation

// The constants HIERDIESPALTEN and HIERWGANZ get replaced by the Rust code
// by the number of columns of the result, resp. the workgroup size.


@compute @workgroup_size(HIERWGANZ,1,1)
fn main(@builtin(local_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = a_shape.x;
    let n = b_shape.y;


    var res_curr_row: array<f32, HIERDIESPALTEN> = array<f32, HIERDIESPALTEN>();
    var nz_row_marker: array<u32, HIERDIESPALTEN> = array<u32, HIERDIESPALTEN>();   


    let i = id.x + wid.x * HIERWGANZu;

    
        for (var col_pos_pos = a_row_pos[i]; col_pos_pos < a_row_pos[i+1]; col_pos_pos++) {
            let k = a_col_pos[col_pos_pos];

            for (var other_col_pos_pos = b_row_pos[k]; other_col_pos_pos < b_row_pos[k+1]; other_col_pos_pos++) {
                let j = b_col_pos[other_col_pos_pos];

                // res_curr_row[i*n + j] = res_curr_row[i*n + j] + a_values[col_pos_pos] * b_values[other_col_pos_pos];
                // nz_row_marker[i*n + j] = 1;  

                res_curr_row[j] = res_curr_row[j] + a_values[col_pos_pos] * b_values[other_col_pos_pos];
                nz_row_marker[j] = 1;     
            }
        }

        for (var k: u32 = 0; k < n; k++) {
            // let x = res_curr_row[i*n + k];
            // let marker = nz_row_marker[i*n + k];

            let x = res_curr_row[k];
            let marker = nz_row_marker[k];
            // atomicAdd(&idx, 1u);
            if marker > 0{
                let write_idx = atomicAdd(&idx, 1u);
                glob_data[write_idx].i = i;
                glob_data[write_idx].j = k;
                glob_data[write_idx].x = x;
            }
        }
}

