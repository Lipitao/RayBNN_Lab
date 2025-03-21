extern crate arrayfire;
use crate::graph::large_sparse_i32::COO_batch_find;
use nohash_hasher;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::graph::large_sparse_i32::remap_rows;
use crate::graph::tree_i32::find_unique;

use crate::neural::network_f64::network_metadata_type;

use crate::graph::large_sparse_i32::COO_to_CSR;
use crate::graph::tree_i32::traverse_backward;

use crate::graph::adjacency_f64::get_global_weight_idx;
use crate::graph::adjacency_f64::get_global_weight_idx2;

use crate::graph::large_sparse_u64::COO_batch_find as COO_batch_find_u64;

use crate::graph::large_sparse_i32::integer_histogram;

const COO_find_limit: u64 = 1500000000;

pub fn find_path_backward_group2(
    netdata: &network_metadata_type,
    Xslices: u64,
    Yslices: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_idx: &arrayfire::Array<i32>,

    Wdims0: u64,
    Hdims0: u64,
    Adims0: u64,
    Bdims0: u64,
    Cdims0: u64,
    Ddims0: u64,
    Edims0: u64,

    idxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    valsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    cvec_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    dXsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    nrows_out: &mut nohash_hasher::IntMap<i64, u64>,
    sparseval_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    sparserow_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    sparsecol_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    Hidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Aidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Bidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Cidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Didxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Eidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    combidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    dAseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dBseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dCseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dDseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dEseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,

    Wseqs: &mut [arrayfire::Seq<i32>; 1],
    Hseqs: &mut [arrayfire::Seq<i32>; 1],
    Aseqs: &mut [arrayfire::Seq<i32>; 1],
    Bseqs: &mut [arrayfire::Seq<i32>; 1],
    Cseqs: &mut [arrayfire::Seq<i32>; 1],
    Dseqs: &mut [arrayfire::Seq<i32>; 1],
    Eseqs: &mut [arrayfire::Seq<i32>; 1],
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let COO_batch_size = 1 + ((COO_find_limit / WRowIdxCOO.dims()[0]) as u64);

    //Get current selection of neurons
    let active_size = neuron_idx.dims()[0];
    let mut newidxsel = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let mut idxsel = newidxsel.clone();
    let mut output_idxsel = newidxsel.clone();

    let mut yslicidx: i64 = (Yslices - 1) as i64;

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut valsel = arrayfire::constant::<i32>(0, temp_dims);

    let mut rvec = arrayfire::constant::<i32>(0, temp_dims);

    let mut cvec = arrayfire::constant::<i32>(0, temp_dims);

    let mut dXsel = arrayfire::constant::<i32>(0, temp_dims);

    let mut sparseval = arrayfire::constant::<i32>(0, temp_dims);
    let mut sparserow = arrayfire::constant::<i32>(0, temp_dims);
    let mut sparsecol = arrayfire::constant::<i32>(0, temp_dims);
    let mut gidx1 = arrayfire::constant::<u64>(0, temp_dims);

    let Hoffset = Wdims0 as i32;
    let Aoffset = (Wdims0 + Hdims0) as i32;
    let Boffset = ((Aoffset as u64) + Adims0) as i32;
    let Coffset = ((Boffset as u64) + Bdims0) as i32;
    let Doffset = ((Coffset as u64) + Cdims0) as i32;
    let Eoffset = ((Doffset as u64) + Ddims0) as i32;

    let mut Hidxsel = idxsel.clone();
    let mut Aidxsel = idxsel.clone();
    let mut Bidxsel = idxsel.clone();
    let mut Cidxsel = idxsel.clone();
    let mut Didxsel = idxsel.clone();
    let mut Eidxsel = idxsel.clone();
    let mut combidxsel = idxsel.clone();

    //Main loop
    for i in (0i64..(Xslices as i64)).rev() {
        idxsel = newidxsel.clone();
        idxsel_out.insert(i, idxsel.clone());

        Hidxsel = Hoffset + idxsel.clone();
        Aidxsel = Aoffset + idxsel.clone();
        Bidxsel = Boffset + idxsel.clone();
        Cidxsel = Coffset + idxsel.clone();
        Didxsel = Doffset + idxsel.clone();
        Eidxsel = Eoffset + idxsel.clone();

        Hidxsel_out.insert(i, Hidxsel.clone());
        Aidxsel_out.insert(i, Aidxsel.clone());
        Bidxsel_out.insert(i, Bidxsel.clone());
        Cidxsel_out.insert(i, Cidxsel.clone());
        Didxsel_out.insert(i, Didxsel.clone());
        Eidxsel_out.insert(i, Eidxsel.clone());

        let dAsize = idxsel.dims()[0];

        let dAstart = 0;
        let dAend = dAstart + dAsize - 1;

        let dBstart = dAend + 1;
        let dBend = dBstart + dAsize - 1;

        let dCstart = dBend + 1;
        let dCend = dCstart + dAsize - 1;

        let dDstart = dCend + 1;
        let dDend = dDstart + dAsize - 1;

        let dEstart = dDend + 1;
        let dEend = dEstart + dAsize - 1;

        dAseqs_out.insert(
            i,
            [
                arrayfire::Seq::new(dAstart as i32, dAend as i32, 1i32),
                arrayfire::Seq::default(),
            ],
        );
        dBseqs_out.insert(
            i,
            [
                arrayfire::Seq::new(dBstart as i32, dBend as i32, 1i32),
                arrayfire::Seq::default(),
            ],
        );
        dCseqs_out.insert(
            i,
            [
                arrayfire::Seq::new(dCstart as i32, dCend as i32, 1i32),
                arrayfire::Seq::default(),
            ],
        );
        dDseqs_out.insert(
            i,
            [
                arrayfire::Seq::new(dDstart as i32, dDend as i32, 1i32),
                arrayfire::Seq::default(),
            ],
        );
        dEseqs_out.insert(
            i,
            [
                arrayfire::Seq::new(dEstart as i32, dEend as i32, 1i32),
                arrayfire::Seq::default(),
            ],
        );

        combidxsel =
            arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[idxsel.dims()[0] * 5, 1, 1, 1]));
        arrayfire::assign_seq(
            &mut combidxsel,
            &[arrayfire::Seq::new(dAstart as i32, dAend as i32, 1i32)],
            &Aidxsel,
        );
        arrayfire::assign_seq(
            &mut combidxsel,
            &[arrayfire::Seq::new(dBstart as i32, dBend as i32, 1i32)],
            &Bidxsel,
        );
        arrayfire::assign_seq(
            &mut combidxsel,
            &[arrayfire::Seq::new(dCstart as i32, dCend as i32, 1i32)],
            &Cidxsel,
        );
        arrayfire::assign_seq(
            &mut combidxsel,
            &[arrayfire::Seq::new(dDstart as i32, dDend as i32, 1i32)],
            &Didxsel,
        );
        arrayfire::assign_seq(
            &mut combidxsel,
            &[arrayfire::Seq::new(dEstart as i32, dEend as i32, 1i32)],
            &Eidxsel,
        );

        combidxsel_out.insert(i, combidxsel.clone());

        //Get indexes of WValues
        valsel = COO_batch_find(WRowIdxCOO, &idxsel, COO_batch_size);
        valsel_out.insert(i, valsel.clone());

        //Get rows of WRowIdx
        rvec = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        //Get cols
        cvec = arrayfire::lookup(WColIdx, &valsel, 0);
        cvec_out.insert(i, cvec.clone());

        //Find idx of dX
        dXsel = remap_rows(&rvec, &idxsel, neuron_size);
        dXsel_out.insert(i, dXsel);

        //Compute global index
        gidx1 = get_global_weight_idx2(neuron_size, &rvec, &cvec);

        //Sort array
        let (_, idx) = arrayfire::sort_index(&gidx1, 0, true);

        //Sparse value
        sparseval = arrayfire::lookup(&valsel, &idx, 0);

        //Sparse Col vector
        sparsecol = arrayfire::lookup(&rvec, &idx, 0);

        let mut temparr =
            arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]));

        let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

        let mut counts = arrayfire::iota::<i32>(idxsel.dims(), repeat_dims);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&idxsel, 0, None);
        arrayfire::assign_gen(&mut temparr, &idxrs, &counts);

        sparsecol = arrayfire::lookup(&temparr, &sparsecol, 0);

        //Sparse Row
        sparserow = arrayfire::lookup(&cvec, &idx, 0);

        let ones = arrayfire::constant::<i32>(1, sparserow.dims());
        let (_, mut sumarr) = arrayfire::sum_by_key(&sparserow, &ones, 0);

        nrows_out.insert(i, sumarr.dims()[0].clone());

        sparserow = arrayfire::accum(&sumarr, 0);

        let constarr = arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        sparserow = arrayfire::join(0, &constarr, &sparserow);

        sparseval_out.insert(i, sparseval.clone());
        sparserow_out.insert(i, sparserow.clone());
        sparsecol_out.insert(i, sparsecol.clone());

        //Next idxsel
        newidxsel = find_unique(&cvec, neuron_size);

        //Add new Y error
        if (yslicidx > 0) {
            yslicidx = yslicidx - 1;

            newidxsel = arrayfire::join(0, &newidxsel, &output_idxsel);
        }
    }

    let Wstart = 0;
    let Wend = (Wdims0 as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (Hdims0 as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (Adims0 as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (Bdims0 as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (Cdims0 as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (Ddims0 as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (Edims0 as i64) - 1;

    *Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    *Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    *Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    *Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    *Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    *Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    *Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];
}

pub fn delete_loops(
    last_idx: &arrayfire::Array<i32>,
    first_idx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut cur_idx = last_idx.clone();
    let mut cur_num = cur_idx.dims()[0] as i64;
    let mut filter_idx = arrayfire::join(0, &first_idx, &last_idx);

    let mut input_idx = arrayfire::constant::<i32>(0, single_dims);
    let mut temp_first_idx = arrayfire::constant::<i32>(0, single_dims);
    let mut detect_first_idx = arrayfire::constant::<i32>(0, single_dims);
    let mut next_idx = arrayfire::constant::<i32>(0, single_dims);
    let mut con_first_idx = arrayfire::constant::<i32>(0, single_dims);

    let mut delWRowIdxCOO = arrayfire::constant::<i32>(0, single_dims);
    let mut delWColIdx = arrayfire::constant::<i32>(0, single_dims);

    let mut COO_batch_size = 1 + ((COO_find_limit / filter_idx.dims()[0]) as u64);

    let mut table = arrayfire::constant::<bool>(true, single_dims);

    let mut inarr = arrayfire::constant::<bool>(false, single_dims);

    let mut tempidx = arrayfire::locate(&table);

    for j in 0..depth {
        cur_num = cur_idx.dims()[0] as i64;

        if j == (depth - 1) {
            filter_idx = arrayfire::rows(
                &filter_idx,
                first_idx.dims()[0] as i64,
                (filter_idx.dims()[0] - 1) as i64,
            );
        }

        input_idx = arrayfire::row(&cur_idx, 0);

        traverse_backward(
            &input_idx,
            WRowIdxCOO,
            WColIdx,
            neuron_size,
            1,
            &mut temp_first_idx,
        );

        if detect_first_idx.dims()[0] > 0 {
            COO_batch_size = 1 + ((COO_find_limit / temp_first_idx.dims()[0]) as u64);
            detect_first_idx = COO_batch_find(&temp_first_idx, &filter_idx, COO_batch_size);

            if detect_first_idx.dims()[0] > 0 {
                con_first_idx = arrayfire::lookup(&temp_first_idx, &detect_first_idx, 0);

                input_idx = arrayfire::tile(&input_idx, con_first_idx.dims());

                delWRowIdxCOO = arrayfire::join(0, &delWRowIdxCOO, &input_idx);
                delWColIdx = arrayfire::join(0, &delWColIdx, &con_first_idx);

                table = arrayfire::constant::<bool>(true, temp_first_idx.dims());
                inarr = arrayfire::constant::<bool>(false, detect_first_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&detect_first_idx, 0, None);
                arrayfire::assign_gen(&mut table, &idxrs, &inarr);

                tempidx = arrayfire::locate(&table);

                if (tempidx.dims()[0] > 0) {
                    temp_first_idx = arrayfire::lookup(&temp_first_idx, &tempidx, 0);
                }
            }
        }

        next_idx = temp_first_idx.clone();

        for i in 1..cur_num {
            input_idx = arrayfire::row(&cur_idx, i);

            traverse_backward(
                &input_idx,
                WRowIdxCOO,
                WColIdx,
                neuron_size,
                1,
                &mut temp_first_idx,
            );

            if (temp_first_idx.dims()[0] == 0) {
                continue;
            }

            COO_batch_size = 1 + ((COO_find_limit / temp_first_idx.dims()[0]) as u64);
            detect_first_idx = COO_batch_find(&temp_first_idx, &filter_idx, COO_batch_size);

            if detect_first_idx.dims()[0] > 0 {
                con_first_idx = arrayfire::lookup(&temp_first_idx, &detect_first_idx, 0);

                input_idx = arrayfire::tile(&input_idx, con_first_idx.dims());

                delWRowIdxCOO = arrayfire::join(0, &delWRowIdxCOO, &input_idx);
                delWColIdx = arrayfire::join(0, &delWColIdx, &con_first_idx);

                table = arrayfire::constant::<bool>(true, temp_first_idx.dims());
                inarr = arrayfire::constant::<bool>(false, detect_first_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&detect_first_idx, 0, None);
                arrayfire::assign_gen(&mut table, &idxrs, &inarr);

                tempidx = arrayfire::locate(&table);

                if (tempidx.dims()[0] == 0) {
                    continue;
                }

                temp_first_idx = arrayfire::lookup(&temp_first_idx, &tempidx, 0);
            }

            next_idx = arrayfire::join(0, &next_idx, &temp_first_idx);
            next_idx = find_unique(&next_idx, neuron_size);
        }
        cur_idx = next_idx.clone();

        filter_idx = arrayfire::join(0, &next_idx, &filter_idx);
        filter_idx = find_unique(&filter_idx, neuron_size);
    }

    drop(cur_idx);
    drop(filter_idx);
    drop(input_idx);
    drop(temp_first_idx);
    drop(detect_first_idx);

    delWRowIdxCOO = arrayfire::rows(&delWRowIdxCOO, 1, (delWRowIdxCOO.dims()[0] - 1) as i64);
    delWColIdx = arrayfire::rows(&delWColIdx, 1, (delWColIdx.dims()[0] - 1) as i64);

    //Compute global index
    let gidx1 = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    let gidx2 = get_global_weight_idx(neuron_size, &delWRowIdxCOO, &delWColIdx);

    //TO CPU
    let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
    gidx1.host(&mut gidx1_cpu);

    let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
    gidx2.host(&mut gidx2_cpu);

    let mut WValues_cpu = vec![f64::default(); WValues.elements()];
    WValues.host(&mut WValues_cpu);

    let mut WRowIdxCOO_cpu = vec![i32::default(); WRowIdxCOO.elements()];
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

    let mut WColIdx_cpu = vec![i32::default(); WColIdx.elements()];
    WColIdx.host(&mut WColIdx_cpu);

    let mut join_WValues = nohash_hasher::IntMap::default();
    let mut join_WColIdx = nohash_hasher::IntMap::default();
    let mut join_WRowIdxCOO = nohash_hasher::IntMap::default();

    //Place old values
    for qq in 0..gidx1.elements() {
        let cur_gidx = gidx1_cpu[qq].clone();

        join_WValues.insert(cur_gidx, WValues_cpu[qq].clone());
        join_WColIdx.insert(cur_gidx, WColIdx_cpu[qq].clone());
        join_WRowIdxCOO.insert(cur_gidx, WRowIdxCOO_cpu[qq].clone());
    }

    //Remove values
    for qq in 0..gidx2.elements() {
        let cur_gidx = gidx2_cpu[qq].clone();

        join_WValues.remove(&cur_gidx);
        join_WColIdx.remove(&cur_gidx);
        join_WRowIdxCOO.remove(&cur_gidx);
    }

    let mut gidx3: Vec<u64> = join_WValues.clone().into_keys().collect();
    gidx3.par_sort_unstable();

    WValues_cpu = Vec::new();
    WRowIdxCOO_cpu = Vec::new();
    WColIdx_cpu = Vec::new();

    for qq in gidx3 {
        WValues_cpu.push(join_WValues[&qq].clone());
        WColIdx_cpu.push(join_WColIdx[&qq].clone());
        WRowIdxCOO_cpu.push(join_WRowIdxCOO[&qq].clone());
    }

    *WValues = arrayfire::Array::new(
        &WValues_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    *WColIdx = arrayfire::Array::new(
        &WColIdx_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    *WRowIdxCOO = arrayfire::Array::new(
        &WRowIdxCOO_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );

    /*
    COO_batch_size = 1 + ((COO_find_limit/gidx1.dims()[0]) as u64);

    let gidx3 = COO_batch_find_u64(&gidx1, &gidx2, COO_batch_size);



    //Filter out existing connections
    if gidx3.dims()[0] > 0
    {

        table = arrayfire::constant::<bool>(true,gidx1.dims());


        inarr = arrayfire::constant::<bool>(false, gidx3.dims());

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&gidx3, 0, None);
        arrayfire::assign_gen(&mut table, &idxrs, &inarr);

        tempidx = arrayfire::locate(&table);


        *WValues = arrayfire::lookup(WValues, &tempidx, 0);
        *WRowIdxCOO = arrayfire::lookup(WRowIdxCOO, &tempidx, 0);
        *WColIdx = arrayfire::lookup(WColIdx, &tempidx, 0);
    }
    */
}
