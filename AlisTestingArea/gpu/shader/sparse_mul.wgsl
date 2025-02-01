// struct CSR {
//     row_pos: array<u32>,
//     col_pos: array<u32>,
//     values: array<f32>,
//     shape: vec2<u32>
// }

struct DataEntry {
    i: u32,
    j: u32,
    x: f32,
};


// struct GlobalResult {
//    idx: atomic<u32>,
//    data: array<DataEntry>,
// };

// @group(0) @binding(0) var<storage, read> a: CSR;

@group(0) @binding(0) var<storage, read> a_row_pos: array<u32>;
@group(0) @binding(1) var<storage, read> a_col_pos: array<u32>;
@group(0) @binding(2) var<storage, read> a_values: array<f32>;
@group(0) @binding(3) var<storage, read> a_shape: vec2<u32>;


@group(1) @binding(0) var<storage, read> b_row_pos: array<u32>;
@group(1) @binding(1) var<storage, read> b_col_pos: array<u32>;
@group(1) @binding(2) var<storage, read> b_values: array<f32>;
@group(1) @binding(3) var<storage, read> b_shape: vec2<u32>;

// @group(2) @binding(0) var<storage, read_write> c_row_pos: array<u32>;
// @group(2) @binding(1) var<storage, read_write> c_col_pos: array<u32>;
// @group(2) @binding(2) var<storage, read_write> c_values: array<f32>;
// @group(2) @binding(3) var<storage, read_write> c_shape: vec2<u32>;


@group(2) @binding(0) var<storage, read_write> idx: atomic<u32>;
@group(2) @binding(1) var<storage, read_write> glob_data: array<DataEntry>;
@group(2) @binding(2) var<storage, read_write> res_curr_row: array<f32>;
@group(2) @binding(3) var<storage, read_write> nz_row_marker: array<u32>;


@compute @workgroup_size(18,1,1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let m = a_shape.x;
    let n = b_shape.y;

    let i: u32 = id.x;

    // if i == 0 {
        // glob_data[i].i = b_row_pos[i];
        // glob_data[i].j = b_row_pos[i+1];
        // glob_data[i].x = b_values[i];

    // }
    // if i == 0 {
    //     glob_data[0].i = a_row_pos[0];
    //     glob_data[0].j = a_row_pos[1];

    // }
    

    for (var col_pos_pos = a_row_pos[i]; col_pos_pos < a_row_pos[i+1]; col_pos_pos++) {
        let k = a_col_pos[col_pos_pos];



        for (var other_col_pos_pos = b_row_pos[k]; other_col_pos_pos < b_row_pos[k+1]; other_col_pos_pos++) {
            let j = b_col_pos[other_col_pos_pos];


            // if i == 0 {
            //     glob_data[k].i = b_col_pos[other_col_pos_pos];
            // }

            res_curr_row[i*n + j] = res_curr_row[i*n + j] + a_values[col_pos_pos] * b_values[other_col_pos_pos];
            nz_row_marker[i*n + j] = 1;     

        }
    }

    for (var k: u32 = 0; k < n; k++) {
        let x = res_curr_row[i*n + k];
        let marker = nz_row_marker[i*n + k];
        // atomicAdd(&idx, 1u);
        if marker > 0{
            let idx = atomicAdd(&idx, 1u) - 1;
            glob_data[idx].i = i;
            glob_data[idx].j = k;
            glob_data[idx].x = x;
        }

    }

    // atomicAdd(&idx, res_curr_row[0]);

    // glob_data[0].i = b_shape.x;
    // glob_data[0].j = b_shape.y;
    // glob_data[0].x = b_values[0];




    // c_shape.x = 20;
    // c_shape.y = 20;
}

