extern crate arrayfire;

use crate::graph::large_sparse_i32::COO_batch_find;

const COO_find_limit: u64 = 1500000000;

pub fn find_unique(arr: &arrayfire::Array<i32>, neuron_size: u64) -> arrayfire::Array<i32> {
    let table_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    let mut table = arrayfire::constant::<bool>(false, table_dims);

    let inarr = arrayfire::constant::<bool>(true, arr.dims());
    //let idxarr = arr.cast::<u32>();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(arr, 0, None);
    arrayfire::assign_gen(&mut table, &idxrs, &inarr);

    arrayfire::locate(&table).cast::<i32>()
}

pub fn traverse_backward(
    in_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,
    out_idx: &mut arrayfire::Array<i32>,
) {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    *out_idx = in_idx.clone();
    let mut valsel = arrayfire::constant::<i32>(0, temp_dims);

    let COO_batch_size = 1 + ((COO_find_limit / WColIdx.dims()[0]) as u64);

    for i in 0..depth {
        valsel = COO_batch_find(WRowIdxCOO, out_idx, COO_batch_size);
        if valsel.dims()[0] == 0 {
            break;
        }
        *out_idx = arrayfire::lookup(WColIdx, &valsel, 0);

        *out_idx = find_unique(out_idx, neuron_size);

        if out_idx.dims()[0] == 0 {
            break;
        }
    }
}

pub fn traverse_forward(
    in_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,
    out_idx: &mut arrayfire::Array<i32>,
) {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    *out_idx = in_idx.clone();
    let mut valsel = arrayfire::constant::<i32>(0, temp_dims);

    let COO_batch_size = 1 + ((COO_find_limit / WColIdx.dims()[0]) as u64);

    for i in 0..depth {
        valsel = COO_batch_find(WColIdx, out_idx, COO_batch_size);
        if valsel.dims()[0] == 0 {
            break;
        }
        *out_idx = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        *out_idx = find_unique(out_idx, neuron_size);

        if out_idx.dims()[0] == 0 {
            break;
        }
    }
}

pub fn check_connected(
    in_idx: &arrayfire::Array<i32>,
    out_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,
) -> bool {
    let mut connected: bool = true;

    let in_num = in_idx.dims()[0] as i64;

    let mut temp_out_idx = in_idx.clone();

    let mut input_idx = in_idx.clone();
    let mut detect_out_idx = in_idx.clone();

    let out_num = out_idx.dims()[0];

    let COO_batch_size = 1 + ((COO_find_limit / out_idx.dims()[0]) as u64);

    let mut con_out_idx = in_idx.clone();

    for i in 0..in_num {
        input_idx = arrayfire::row(&in_idx, i);

        traverse_forward(
            &input_idx,
            WRowIdxCOO,
            WColIdx,
            neuron_size,
            depth,
            &mut temp_out_idx,
        );

        detect_out_idx = COO_batch_find(out_idx, &temp_out_idx, COO_batch_size);

        con_out_idx = arrayfire::lookup(out_idx, &detect_out_idx, 0);

        if detect_out_idx.dims()[0] < out_num {
            connected = false;
            println!("Not Connected");
        } else {
            println!("Connected");
        }

        arrayfire::print_gen("input_idx".to_string(), &input_idx, Some(6));
        arrayfire::print_gen("con_out_idx".to_string(), &con_out_idx, Some(6));
    }

    connected
}

pub fn check_connected2(
    in_idx: &arrayfire::Array<i32>,
    out_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,
) -> bool {
    let mut connected: bool = true;

    let in_num = in_idx.dims()[0] as i64;

    let mut temp_out_idx = in_idx.clone();

    let mut input_idx = in_idx.clone();
    let mut detect_out_idx = in_idx.clone();

    let out_num = out_idx.dims()[0];

    let COO_batch_size = 1 + ((COO_find_limit / out_idx.dims()[0]) as u64);

    let mut con_out_idx = in_idx.clone();

    for i in 0..in_num {
        input_idx = arrayfire::row(&in_idx, i);

        traverse_forward(
            &input_idx,
            WRowIdxCOO,
            WColIdx,
            neuron_size,
            depth,
            &mut temp_out_idx,
        );

        detect_out_idx = COO_batch_find(out_idx, &temp_out_idx, COO_batch_size);

        con_out_idx = arrayfire::lookup(out_idx, &detect_out_idx, 0);

        if detect_out_idx.dims()[0] < out_num {
            connected = false;
            break;
            //println!("Not Connected");
        } else {
            //println!("Connected");
        }

        //arrayfire::print_gen("input_idx".to_string(), &input_idx,Some(6));
        //arrayfire::print_gen("con_out_idx".to_string(), &con_out_idx,Some(6));
    }

    connected
}
