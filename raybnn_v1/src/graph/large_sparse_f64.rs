extern crate arrayfire;

use rayon::prelude::*;

const LOOP_THRESHOLD: usize = 6;

pub fn block_sparse_matmul_rayon(
    input_start: &Vec<i64>,
    input_end: &Vec<i64>,

    block_start: &Vec<i64>,
    block_end: &Vec<i64>,

    input: &arrayfire::Array<f64>,
    block: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    let output_vec: Vec<arrayfire::Array<f64>> = (input_start, input_end, block_start, block_end)
        .into_par_iter()
        .map(|(istart, iend, bstart, bend)| {
            //[lhs.dims()[0]  rhs.dims()[0] ]
            let mut lhs = arrayfire::rows(input, *istart, *iend);

            //[rhs.dims()[0]  rhs.dims()[1]]
            let rhs = arrayfire::slices(block, *bstart, *bend);

            let dim0 = lhs.dims()[0];
            let dim1 = rhs.dims()[1];

            let row_num = ((*bend) as u64) - ((*bstart) as u64) + 1;
            let seg = dim0 / row_num;

            //[rhs.dims()[0]  lhs.dims()[0]   ]
            lhs = arrayfire::transpose(&lhs, false);

            let dims = arrayfire::Dim4::new(&[rhs.dims()[0], seg, row_num, 1]);
            lhs = arrayfire::moddims(&lhs, dims);

            //[seg  rhs.dims()[0]   lhs.dims()[0]/(*seg) ]
            lhs = arrayfire::transpose(&lhs, false);

            //[seg  rhs.dims()[1]   lhs.dims()[0]/(*seg) ]
            let mut ret = arrayfire::matmul(
                &lhs,
                &rhs,
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::NONE,
            );
            drop(lhs);
            drop(rhs);

            //[rhs.dims()[1]   seg  lhs.dims()[0]/(*seg) ]
            ret = arrayfire::transpose(&ret, false);

            let dims = arrayfire::Dim4::new(&[dim1, dim0, 1, 1]);
            ret = arrayfire::moddims(&ret, dims);

            arrayfire::transpose(&ret, false)
        })
        .collect();

    let output_vec_iter = output_vec.par_iter().collect();

    arrayfire::join_many(0, output_vec_iter)
}

pub fn block_sparse_matmul_loop(
    input_start: &Vec<i64>,
    input_end: &Vec<i64>,

    block_start: &Vec<i64>,
    block_end: &Vec<i64>,

    input: &arrayfire::Array<f64>,
    block: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    let mut istart = 0;
    let mut iend = 0;
    let mut bstart = 0;
    let mut bend = 0;

    let outputarr_dims = arrayfire::Dim4::new(&[1, block.dims()[1], 1, 1]);

    let mut outputarr = arrayfire::constant::<f64>(0.0, outputarr_dims);
    for ii in 0..input_start.len() {
        istart = input_start[ii];
        iend = input_end[ii];
        bstart = block_start[ii];
        bend = block_end[ii];

        //[lhs.dims()[0]  rhs.dims()[0] ]
        let mut lhs = arrayfire::rows(input, istart, iend);

        //[rhs.dims()[0]  rhs.dims()[1]]
        let rhs = arrayfire::slices(block, bstart, bend);

        let dim0 = lhs.dims()[0];
        let dim1 = rhs.dims()[1];

        let row_num = ((bend) as u64) - ((bstart) as u64) + 1;
        let seg = dim0 / row_num;

        //[rhs.dims()[0]  lhs.dims()[0]   ]
        lhs = arrayfire::transpose(&lhs, false);

        let dims = arrayfire::Dim4::new(&[rhs.dims()[0], seg, row_num, 1]);
        lhs = arrayfire::moddims(&lhs, dims);

        //[seg  rhs.dims()[0]   lhs.dims()[0]/(*seg) ]
        lhs = arrayfire::transpose(&lhs, false);

        //[seg  rhs.dims()[1]   lhs.dims()[0]/(*seg) ]
        let mut ret = arrayfire::matmul(
            &lhs,
            &rhs,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        );
        drop(lhs);
        drop(rhs);

        //[rhs.dims()[1]   seg  lhs.dims()[0]/(*seg) ]
        ret = arrayfire::transpose(&ret, false);

        let dims = arrayfire::Dim4::new(&[dim1, dim0, 1, 1]);
        ret = arrayfire::moddims(&ret, dims);

        ret = arrayfire::transpose(&ret, false);

        outputarr = arrayfire::join(0, &outputarr, &ret);
    }

    outputarr = arrayfire::rows(&outputarr, 1, (outputarr.dims()[0] - 1) as i64);

    outputarr
}

pub fn block_sparse_matmul(
    input_start: &Vec<i64>,
    input_end: &Vec<i64>,

    block_start: &Vec<i64>,
    block_end: &Vec<i64>,

    input: &arrayfire::Array<f64>,
    block: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    if input_start.len() < LOOP_THRESHOLD {
        return block_sparse_matmul_loop(
            &input_start,
            &input_end,
            &block_start,
            &block_end,
            &input,
            &block,
        );
    } else {
        return block_sparse_matmul_rayon(
            &input_start,
            &input_end,
            &block_start,
            &block_end,
            &input,
            &block,
        );
    }
}

pub fn trans_block_sparse_matmul_rayon(
    pointer_start: &Vec<i64>,
    pointer_end: &Vec<i64>,

    seg_size: &Vec<u64>,

    input: &arrayfire::Array<f64>,
    block: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    let output_vec: Vec<arrayfire::Array<f64>> = (pointer_start, pointer_end, seg_size)
        .into_par_iter()
        .map(|(istart, iend, segs)| {
            //[lhs.dims()[0]  rhs.dims()[0] ]
            let mut lhs = arrayfire::cols(input, *istart, *iend);

            //[rhs.dims()[0]  rhs.dims()[1]]
            let mut rhs = arrayfire::rows(block, *istart, *iend);

            let lhs_size = lhs.dims()[1];

            let lhs_dims = arrayfire::Dim4::new(&[1, *segs, lhs_size / (*segs), 1]);
            lhs = arrayfire::moddims(&lhs, lhs_dims);

            let rhs_dim0 = rhs.dims()[0];
            let rhs_dim1 = rhs.dims()[1];

            rhs = arrayfire::transpose(&rhs, false);

            let rhs_dims = arrayfire::Dim4::new(&[rhs_dim1, *segs, rhs_dim0 / (*segs), 1]);
            rhs = arrayfire::moddims(&rhs, rhs_dims);

            rhs = arrayfire::transpose(&rhs, false);

            //[seg  rhs.dims()[1]   lhs.dims()[0]/(*seg) ]
            arrayfire::matmul(
                &lhs,
                &rhs,
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::NONE,
            )
        })
        .collect();

    let output_vec_iter = output_vec.par_iter().collect();

    let jvec = arrayfire::join_many(2, output_vec_iter);

    arrayfire::reorder_v2(&jvec, 2, 1, Some(vec![0]))
}

pub fn trans_block_sparse_matmul_loop(
    pointer_start: &Vec<i64>,
    pointer_end: &Vec<i64>,

    seg_size: &Vec<u64>,

    input: &arrayfire::Array<f64>,
    block: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    let mut istart = 0;
    let mut iend = 0;
    let mut segs = 0;

    let outputarr_dims = arrayfire::Dim4::new(&[1, block.dims()[1], 1, 1]);

    let mut outputarr = arrayfire::constant::<f64>(0.0, outputarr_dims);
    for ii in 0..pointer_start.len() {
        istart = pointer_start[ii];
        iend = pointer_end[ii];
        segs = seg_size[ii];

        //[lhs.dims()[0]  rhs.dims()[0] ]
        let mut lhs = arrayfire::cols(input, istart, iend);

        //[rhs.dims()[0]  rhs.dims()[1]]
        let mut rhs = arrayfire::rows(block, istart, iend);

        let lhs_size = lhs.dims()[1];

        let lhs_dims = arrayfire::Dim4::new(&[1, segs, lhs_size / (segs), 1]);
        lhs = arrayfire::moddims(&lhs, lhs_dims);

        let rhs_dim0 = rhs.dims()[0];
        let rhs_dim1 = rhs.dims()[1];

        rhs = arrayfire::transpose(&rhs, false);

        let rhs_dims = arrayfire::Dim4::new(&[rhs_dim1, segs, rhs_dim0 / (segs), 1]);
        rhs = arrayfire::moddims(&rhs, rhs_dims);

        rhs = arrayfire::transpose(&rhs, false);

        //[seg  rhs.dims()[1]   lhs.dims()[0]/(*seg) ]
        let ret = arrayfire::matmul(
            &lhs,
            &rhs,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        );

        outputarr = arrayfire::join(2, &outputarr, &ret);
    }

    outputarr = arrayfire::slices(&outputarr, 1, (outputarr.dims()[2] - 1) as i64);

    arrayfire::reorder_v2(&outputarr, 2, 1, Some(vec![0]))
}

pub fn trans_block_sparse_matmul(
    pointer_start: &Vec<i64>,
    pointer_end: &Vec<i64>,

    seg_size: &Vec<u64>,

    input: &arrayfire::Array<f64>,
    block: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    if pointer_start.len() < LOOP_THRESHOLD {
        return trans_block_sparse_matmul_loop(
            &pointer_start,
            &pointer_end,
            &seg_size,
            &input,
            &block,
        );
    } else {
        return trans_block_sparse_matmul_rayon(
            &pointer_start,
            &pointer_end,
            &seg_size,
            &input,
            &block,
        );
    }
}

pub fn parallel_lookup(
    batch_dim: u64,
    lookup_dim: u64,

    idx: &arrayfire::Array<u32>,
    target: &arrayfire::Array<f64>,
) -> arrayfire::Array<f64> {
    let target_dims = target.dims();

    let batch_num = target_dims[batch_dim as usize];
    let lookup_size = target_dims[lookup_dim as usize];

    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut tile_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    tile_dims[batch_dim as usize] = batch_num;

    let count = arrayfire::iota::<u32>(tile_dims, repeat_dims);

    let mut idx2 = batch_num * idx.clone();

    idx2 = arrayfire::add(&idx2, &count, true);

    drop(count);

    idx2 = arrayfire::flat(&idx2);

    let mut ouput_arr = arrayfire::flat(target);

    ouput_arr = arrayfire::lookup(&ouput_arr, &idx2, 0);

    drop(idx2);

    arrayfire::moddims(&ouput_arr, idx.dims())
}
