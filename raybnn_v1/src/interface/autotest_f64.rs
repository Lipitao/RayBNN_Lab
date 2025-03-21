extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

use crate::neural::network_f64::neural_network_type;

use crate::interface::automatic_f64::arch_search_type;

use crate::graph::large_sparse_i32::CSR_to_COO;

use crate::graph::path_f64::find_path_backward_group2;

use crate::neural::network_f64::state_space_forward_batch;

// Validate one neural network using Cross Validation Dataset

//           Input arguments
// validationdata_X: Training input array to the neural network
// 				validationdata_X.dims()[0]  input feature size
// 				validationdata_X.dims()[1]  batch size
// 				validationdata_X.dims()[2]  sequence/traj size
// validationdata_Y: Training ground truth output array
// 				validationdata_Y.dims()[0]  output feature size
// 				validationdata_Y.dims()[1]  batch size
// 				validationdata_Y.dims()[2]  sequence/traj size
// eval_metric: Function with inputs X and Yhat that gives evaluation metrics
// neural_network:  Entire neural network archtecture

//          Output argumenets
// Yhat_out:  Output of the network in an hashmap
// eval_metric_out:  Result of the evaluation metric on Yhat and Y

pub fn validate_network(
    validationdata_X: &nohash_hasher::IntMap<u64, Vec<f64>>,
    validationdata_Y: &nohash_hasher::IntMap<u64, Vec<f64>>,

    eval_metric: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> f64 + Copy,
    arch_search: &arch_search_type,

    Yhat_out: &mut nohash_hasher::IntMap<u64, Vec<f64>>,
    eval_metric_out: &mut Vec<f64>,
) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f64 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f64 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f64 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f64 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f64 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f64 = (*arch_search).neural_network.netdata.con_rad.clone();
    let center_const: f64 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f64 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f64 = (*arch_search).neural_network.netdata.repel_const.clone();

    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let traj_size: u64 = (validationdata_X[&0].len() as u64) / (input_size * batch_size);

    let mut WValues = arrayfire::rows(
        &((*arch_search).neural_network.network_params),
        0,
        ((*arch_search).neural_network.WColIdx.dims()[0] - 1) as i64,
    );

    let traj_steps = traj_size + proc_num - 1;

    let Z_dims = arrayfire::Dim4::new(&[neuron_size, batch_size, traj_steps, 1]);
    let mut Z = arrayfire::constant::<f64>(0.0, Z_dims);
    let mut Q = arrayfire::constant::<f64>(0.0, Z_dims);

    let active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(
        &((*arch_search).neural_network.neuron_idx),
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let Qslices: u64 = Q.dims()[2];

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let total_param_size = (*arch_search).neural_network.network_params.dims()[0];
    let mt_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);
    let mut mt = arrayfire::constant::<f64>(0.0, mt_dims);
    let mut vt = arrayfire::constant::<f64>(0.0, mt_dims);
    let mut grad = arrayfire::constant::<f64>(0.0, mt_dims);

    //let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_steps,1]);
    //let mut X = arrayfire::constant::<f64>(0.0,X_dims);

    let mut idxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut valsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut cvec_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut dXsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut nrows_out: nohash_hasher::IntMap<i64, u64> = nohash_hasher::IntMap::default();
    let mut sparseval_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut sparsecol_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut sparserow_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut Wseqs = [arrayfire::Seq::default()];
    let mut Hseqs = [arrayfire::Seq::default()];
    let mut Aseqs = [arrayfire::Seq::default()];
    let mut Bseqs = [arrayfire::Seq::default()];
    let mut Cseqs = [arrayfire::Seq::default()];
    let mut Dseqs = [arrayfire::Seq::default()];
    let mut Eseqs = [arrayfire::Seq::default()];

    let mut Hidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Aidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Bidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Cidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Didxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Eidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut combidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut dAseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dBseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dCseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dDseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dEseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();

    find_path_backward_group2(
        &((*arch_search).neural_network.netdata),
        traj_steps,
        traj_size,
        &WRowIdxCOO,
        &((*arch_search).neural_network.WColIdx),
        &((*arch_search).neural_network.neuron_idx),
        WValues.dims()[0],
        neuron_size,
        neuron_size,
        neuron_size,
        neuron_size,
        neuron_size,
        neuron_size,
        &mut idxsel_out,
        &mut valsel_out,
        &mut cvec_out,
        &mut dXsel_out,
        &mut nrows_out,
        &mut sparseval_out,
        &mut sparserow_out,
        &mut sparsecol_out,
        &mut Hidxsel_out,
        &mut Aidxsel_out,
        &mut Bidxsel_out,
        &mut Cidxsel_out,
        &mut Didxsel_out,
        &mut Eidxsel_out,
        &mut combidxsel_out,
        &mut dAseqs_out,
        &mut dBseqs_out,
        &mut dCseqs_out,
        &mut dDseqs_out,
        &mut dEseqs_out,
        &mut Wseqs,
        &mut Hseqs,
        &mut Aseqs,
        &mut Bseqs,
        &mut Cseqs,
        &mut Dseqs,
        &mut Eseqs,
    );

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_steps, 1]);
    let mut X = arrayfire::constant::<f64>(0.0, X_dims);

    let train_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
    let mut train_X = arrayfire::constant::<f64>(0.0, temp_dims);
    let Y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
    let mut Y = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut batch_idx = 0;
    let epoch_num = validationdata_X.len() as u64;

    train_X = arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);
    arrayfire::set_slices(&mut X, &train_X, 0, (traj_size - 1) as i64);

    //X =  arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);

    Y = arrayfire::Array::new(&validationdata_Y[&batch_idx], Y_dims);

    *eval_metric_out = Vec::new();

    *Yhat_out = nohash_hasher::IntMap::default();

    arrayfire::device_gc();
    for batch_idx in 0..epoch_num {
        train_X = arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);
        arrayfire::set_slices(&mut X, &train_X, 0, (traj_size - 1) as i64);

        //X =  arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);

        Y = arrayfire::Array::new(&validationdata_Y[&batch_idx], Y_dims);

        state_space_forward_batch(
            &((*arch_search).neural_network.netdata),
            &X,
            &((*arch_search).neural_network.WRowIdxCSR),
            &((*arch_search).neural_network.WColIdx),
            &Wseqs,
            &Hseqs,
            &Aseqs,
            &Bseqs,
            &Cseqs,
            &Dseqs,
            &Eseqs,
            &((*arch_search).neural_network.network_params),
            &mut Z,
            &mut Q,
        );

        //Get Yhat
        let mut idxrs = arrayfire::Indexer::default();
        let seq1 = arrayfire::Seq::new(0.0f64, (batch_size - 1) as f64, 1.0);
        let seq2 = arrayfire::Seq::new((proc_num - 1) as f64, (Qslices - 1) as f64, 1.0);
        idxrs.set_index(&idxsel, 0, None);
        idxrs.set_index(&seq1, 1, None);
        idxrs.set_index(&seq2, 2, None);
        let Yhat = arrayfire::index_gen(&Q, idxrs);
        let loss_output = eval_metric(&Yhat, &Y);

        let mut Yhat_out_cpu = vec![f64::default(); Yhat.elements()];
        Yhat.host(&mut Yhat_out_cpu);

        Yhat_out.insert(batch_idx, Yhat_out_cpu);

        eval_metric_out.push(loss_output);
    }
}

pub fn test_network(
    validationdata_X: &nohash_hasher::IntMap<u64, Vec<f64>>,
    //validationdata_Y: &nohash_hasher::IntMap<u64, Vec<f64> >,

    //eval_metric: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> f64   + Copy,
    arch_search: &arch_search_type,

    Yhat_out: &mut nohash_hasher::IntMap<u64, Vec<f64>>,
    //eval_metric_out: &mut Vec<f64>
) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f64 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f64 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f64 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f64 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f64 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f64 = (*arch_search).neural_network.netdata.con_rad.clone();
    let center_const: f64 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f64 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f64 = (*arch_search).neural_network.netdata.repel_const.clone();

    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let traj_size: u64 = (validationdata_X[&0].len() as u64) / (input_size * batch_size);

    let mut WValues = arrayfire::rows(
        &((*arch_search).neural_network.network_params),
        0,
        ((*arch_search).neural_network.WColIdx.dims()[0] - 1) as i64,
    );

    let traj_steps = traj_size + proc_num - 1;

    let Z_dims = arrayfire::Dim4::new(&[neuron_size, batch_size, traj_steps, 1]);
    let mut Z = arrayfire::constant::<f64>(0.0, Z_dims);
    let mut Q = arrayfire::constant::<f64>(0.0, Z_dims);

    let active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(
        &((*arch_search).neural_network.neuron_idx),
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let Qslices: u64 = Q.dims()[2];

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let total_param_size = (*arch_search).neural_network.network_params.dims()[0];
    let mt_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);
    let mut mt = arrayfire::constant::<f64>(0.0, mt_dims);
    let mut vt = arrayfire::constant::<f64>(0.0, mt_dims);
    let mut grad = arrayfire::constant::<f64>(0.0, mt_dims);

    //let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_steps,1]);
    //let mut X = arrayfire::constant::<f64>(0.0,X_dims);

    let mut idxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut valsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut cvec_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut dXsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut nrows_out: nohash_hasher::IntMap<i64, u64> = nohash_hasher::IntMap::default();
    let mut sparseval_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut sparsecol_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut sparserow_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut Wseqs = [arrayfire::Seq::default()];
    let mut Hseqs = [arrayfire::Seq::default()];
    let mut Aseqs = [arrayfire::Seq::default()];
    let mut Bseqs = [arrayfire::Seq::default()];
    let mut Cseqs = [arrayfire::Seq::default()];
    let mut Dseqs = [arrayfire::Seq::default()];
    let mut Eseqs = [arrayfire::Seq::default()];

    let mut Hidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Aidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Bidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Cidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Didxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut Eidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();
    let mut combidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32>> =
        nohash_hasher::IntMap::default();

    let mut dAseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dBseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dCseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dDseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();
    let mut dEseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]> =
        nohash_hasher::IntMap::default();

    find_path_backward_group2(
        &((*arch_search).neural_network.netdata),
        traj_steps,
        traj_size,
        &WRowIdxCOO,
        &((*arch_search).neural_network.WColIdx),
        &((*arch_search).neural_network.neuron_idx),
        WValues.dims()[0],
        neuron_size,
        neuron_size,
        neuron_size,
        neuron_size,
        neuron_size,
        neuron_size,
        &mut idxsel_out,
        &mut valsel_out,
        &mut cvec_out,
        &mut dXsel_out,
        &mut nrows_out,
        &mut sparseval_out,
        &mut sparserow_out,
        &mut sparsecol_out,
        &mut Hidxsel_out,
        &mut Aidxsel_out,
        &mut Bidxsel_out,
        &mut Cidxsel_out,
        &mut Didxsel_out,
        &mut Eidxsel_out,
        &mut combidxsel_out,
        &mut dAseqs_out,
        &mut dBseqs_out,
        &mut dCseqs_out,
        &mut dDseqs_out,
        &mut dEseqs_out,
        &mut Wseqs,
        &mut Hseqs,
        &mut Aseqs,
        &mut Bseqs,
        &mut Cseqs,
        &mut Dseqs,
        &mut Eseqs,
    );

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_steps, 1]);
    let mut X = arrayfire::constant::<f64>(0.0, X_dims);

    let train_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
    let mut train_X = arrayfire::constant::<f64>(0.0, temp_dims);
    let Y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
    let mut Y = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut batch_idx = 0;
    let epoch_num = validationdata_X.len() as u64;

    train_X = arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);
    arrayfire::set_slices(&mut X, &train_X, 0, (traj_size - 1) as i64);

    //X =  arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);

    // Y = arrayfire::Array::new(&validationdata_Y[&batch_idx], Y_dims);

    // *eval_metric_out = Vec::new();

    *Yhat_out = nohash_hasher::IntMap::default();

    arrayfire::device_gc();
    for batch_idx in 0..epoch_num {
        train_X = arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);
        arrayfire::set_slices(&mut X, &train_X, 0, (traj_size - 1) as i64);

        //X =  arrayfire::Array::new(&validationdata_X[&batch_idx], train_X_dims);

        //Y = arrayfire::Array::new(&validationdata_Y[&batch_idx], Y_dims);

        state_space_forward_batch(
            &((*arch_search).neural_network.netdata),
            &X,
            &((*arch_search).neural_network.WRowIdxCSR),
            &((*arch_search).neural_network.WColIdx),
            &Wseqs,
            &Hseqs,
            &Aseqs,
            &Bseqs,
            &Cseqs,
            &Dseqs,
            &Eseqs,
            &((*arch_search).neural_network.network_params),
            &mut Z,
            &mut Q,
        );

        //Get Yhat
        let mut idxrs = arrayfire::Indexer::default();
        let seq1 = arrayfire::Seq::new(0.0f64, (batch_size - 1) as f64, 1.0);
        let seq2 = arrayfire::Seq::new((proc_num - 1) as f64, (Qslices - 1) as f64, 1.0);
        idxrs.set_index(&idxsel, 0, None);
        idxrs.set_index(&seq1, 1, None);
        idxrs.set_index(&seq2, 2, None);
        let Yhat = arrayfire::index_gen(&Q, idxrs);
        //let loss_output = eval_metric(&Yhat,&Y);

        let mut Yhat_out_cpu = vec![f64::default(); Yhat.elements()];
        Yhat.host(&mut Yhat_out_cpu);

        Yhat_out.insert(batch_idx, Yhat_out_cpu);

        //eval_metric_out.push(loss_output);
    }
}
