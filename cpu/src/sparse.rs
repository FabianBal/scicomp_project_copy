use std::sync::{Mutex, Arc};

use rayon::prelude::*;

use matrix_base::{Dense, CSR};




pub trait SparseProd {
    fn product(&self, other: &CSR) -> Dense;
    fn product_sparse(&self, other: &CSR) -> CSR;
    fn product_sparse_par(&self, other: &CSR) -> CSR;
}



impl SparseProd for CSR {
    // Matrix/Matrix product, see seciton 3.2 from "A Systematic Survey of General Sparse Matrix-Matrix Multiplication", Gao et al.
    // https://doi.org/10.1145/3571157
    // Notation also from Paper
    // C = A*B
    // I_i(A) is the set of column indices of all non-zero entries of the i-th row of A
    // Returns dense matrix
    fn product(&self, other: &CSR) -> Dense {
        // let data = vec![];
        let m = self.shape.0;
        let n = other.shape.1;
        // let mut mat = vec![vec![0.;n];m];
        let mut mat = Dense::new_zeros((m,n));

        for i in 0..m {
            // iterate over all non-zero cols of A_{i*}
            // let cols = self.col_pos[i]..self.col_pos[i+1];
            for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
                let k = self.col_pos[col_pos_pos];

                for other_col_pos_pos in other.row_pos[k]..other.row_pos[k+1] {
                    let j = other.col_pos[other_col_pos_pos];

                    // C_{i*} = \sum_{k \in I_i (A)} a_{ik} b_{i*}
                    // a_{ik} = self.values[col_pos_pos]
                    // b_{kj} = other.values[other_col_pos_pos]
                    // mat[i][j] += self.values[col_pos_pos] * other.values[other_col_pos_pos]
                    mat.set(i,j , mat.get(i, j) +   self.values[col_pos_pos] * other.values[other_col_pos_pos]);

                }
            }
        }

        mat
    }




    // For the general algorithm see above
    // This is a modification for directly saving CSR
    // via dense intermediate results, suited for
    // parallel execution
    fn product_sparse(&self, other: &CSR) -> CSR {
        let m = self.shape.0;
        let n = other.shape.1;
        // let mut mat = Dense::new_zeros((m,n));

        let mut res_rows: Vec<Vec<f64>> = vec![];
        let mut res_col_idxs: Vec<Vec<usize>> = vec![];

        for i in 0..m {
            // Iterate over all non-zero cols of A_{i*}

            // Create a dense row for the result matrix, C_{i*}
            // and also a bool array that flags if some non-zero
            // entry for the k-th (i.e. for C_{ik}) is calculated.            
            let mut nz_row_marker = vec![false;n];
            let mut res_curr_row = vec![0.;n];


            for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
                let k = self.col_pos[col_pos_pos];

                for other_col_pos_pos in other.row_pos[k]..other.row_pos[k+1] {
                    let j = other.col_pos[other_col_pos_pos];

                    // C_{i*} = \sum_{k \in I_i (A)} a_{ik} b_{i*}
                    // a_{ik} = self.values[col_pos_pos]
                    // b_{kj} = other.values[other_col_pos_pos]
                    // mat[i][j] += self.values[col_pos_pos] * other.values[other_col_pos_pos]
                    // mat.set(i,j , mat.get(i, j) +   self.values[col_pos_pos] * other.values[other_col_pos_pos]);
                    res_curr_row[j] += self.values[col_pos_pos] * other.values[other_col_pos_pos];
                    nz_row_marker[j] = true;                    

                }
            }

            // Go through the row, which may contain 0 entries.
            // If !marker, then the k-col is 0 and can be ignored
            // Save only non-zero entries and their index
            let mut res_curr_row_final_val = vec![];
            let mut res_curr_row_final_col_idx = vec![];
            for ((k, x), marker) in res_curr_row.iter().enumerate().into_iter().zip(nz_row_marker) {
                if marker {
                    res_curr_row_final_val.push(*x);
                    res_curr_row_final_col_idx.push(k);
                }
            }
            
            // Push the final result for the current row
            res_rows.push(res_curr_row_final_val);
            res_col_idxs.push(res_curr_row_final_col_idx);
        }


        // Create the row_pos vector and flatten the other
        let mut row_pos_counter = 0;
        let mut row_pos = vec![0];

        for c in &res_col_idxs {
            row_pos_counter += c.len();
            row_pos.push(row_pos_counter);
        }

        let col_pos = res_col_idxs.concat();
        let values = res_rows.concat();

        row_pos.push(values.len());

        CSR{row_pos, col_pos, values, shape: (m,n)}

    }



    fn product_sparse_par(&self, other: &CSR) -> CSR {
        let m = self.shape.0;
        let n = other.shape.1;

        let res_rows: Arc<Mutex<Vec<(usize, Vec<f64>)>>> = Arc::new(Mutex::new(vec![]));
        let res_col_idxs: Arc<Mutex<Vec<(usize, Vec<usize>)>>> =  Arc::new(Mutex::new(vec![]));

        (0..m).into_par_iter()
        .for_each(|i| {
            // Iterate over all non-zero cols of A_{i*}

            // Create a dense row for the result matrix, C_{i*}
            // and also a bool array that flags if some non-zero
            // entry for the k-th (i.e. for C_{ik}) is calculated.            
            let mut nz_row_marker = vec![false;n];
            let mut res_curr_row = vec![0.;n];


            for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
                let k = self.col_pos[col_pos_pos];

                for other_col_pos_pos in other.row_pos[k]..other.row_pos[k+1] {
                    let j = other.col_pos[other_col_pos_pos];

                    // C_{i*} = \sum_{k \in I_i (A)} a_{ik} b_{i*}
                    // a_{ik} = self.values[col_pos_pos]
                    // b_{kj} = other.values[other_col_pos_pos]
                    // mat[i][j] += self.values[col_pos_pos] * other.values[other_col_pos_pos]
                    // mat.set(i,j , mat.get(i, j) +   self.values[col_pos_pos] * other.values[other_col_pos_pos]);
                    res_curr_row[j] += self.values[col_pos_pos] * other.values[other_col_pos_pos];
                    nz_row_marker[j] = true;                    

                }
            }

            // Go through the row, which may contain 0 entries.
            // If !marker, then the k-col is 0 and can be ignored
            // Save only non-zero entries and their index
            let mut res_curr_row_final_val = vec![];
            let mut res_curr_row_final_col_idx = vec![];
            for ((k, x), marker) in res_curr_row.iter().enumerate().into_iter().zip(nz_row_marker) {
                if marker {
                    res_curr_row_final_val.push(*x);
                    res_curr_row_final_col_idx.push(k);
                }
            }
            
            // Push the final result for the current row
            let mut rr = res_rows.lock().unwrap();
            let mut rci = res_col_idxs.lock().unwrap();

            rr.push((i,res_curr_row_final_val));
            rci.push((i,res_curr_row_final_col_idx));        
        });


        // Consume Arc Mutex
        let mut res_rows = Arc::try_unwrap(res_rows).unwrap().into_inner().unwrap();
        let mut res_col_idxs = Arc::try_unwrap(res_col_idxs).unwrap().into_inner().unwrap();

        res_rows.sort_by_key(|&(i, _)| i);
        res_col_idxs.sort_by_key(|&(i, _)| i);

        // Create the row_pos vector and flatten the other
        let mut row_pos_counter = 0;
        let mut row_pos = vec![0];

        for (_, c) in &res_col_idxs {
            row_pos_counter += c.len();
            row_pos.push(row_pos_counter);
        }

        let col_pos = res_col_idxs.into_iter().flat_map(|(_, inner_vec)| inner_vec).collect();
        let values: Vec<_> = res_rows.into_iter().flat_map(|(_, inner_vec)| inner_vec).collect();

        row_pos.push(values.len());

        CSR{row_pos, col_pos, values, shape: (m,n)}

    }


}