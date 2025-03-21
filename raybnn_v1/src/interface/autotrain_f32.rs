extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

use num::Float;

use crate::neural::network_f32::network_metadata_type;
use crate::neural::network_f32::neural_network_type;

use crate::export::dataloader_u64::find_cube_paths;

use crate::export::dataloader_f32::extract_file_info2;

use crate::export::dataloader_f32::load_network_structure;

use crate::graph::large_sparse_i32::CSR_to_COO;

use crate::interface::autotransfer_f32::shuffle_weights;

use crate::interface::automatic_f32::arch_search_type;
use crate::interface::automatic_f32::network_info_seed_type;
use crate::interface::automatic_f32::set_network_seed;

use crate::graph::path_f32::find_path_backward_group2;

use crate::neural::network_f32::state_space_forward_batch;

use crate::neural::network_f32::state_space_backward_group2;

use crate::optimal::gd_f32::adam;

use crate::optimal::control_f32::statespace_BTLS;

use serde::{Deserialize, Serialize};

use crate::interface::autotest_f32::validate_network;

use crate::physics::update_f32::add_neuron_to_existing2;

use crate::physics::update_f32::reduce_network_size;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum stop_strategy_type {
    STOP_AT_EPOCH,
    STOP_AT_TRAIN_LOSS,
    CROSSVAL_STOPPING,
    NONE,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum lr_strategy_type {
    COSINE_ANNEALING,
    SHUFFLE_CONNECTIONS,
    NONE,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum lr_strategy2_type {
    BTLS_ALPHA,
    MAX_ALPHA,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
pub enum loss_status_type {
    LOSS_OVERFLOW,
    LOSS_PLATEAU,
    NO_CONVERGENCE,
    PREDETERMINED_STOP,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct train_network_options_type {
    pub stop_strategy: stop_strategy_type,
    pub lr_strategy: lr_strategy_type,
    pub lr_strategy2: lr_strategy2_type,

    pub max_epoch: u64,
    pub stop_epoch: u64,
    pub stop_train_loss: f32,

    pub exit_counter_threshold: u64,
    pub shuffle_counter_threshold: u64,
}

const LR_MAX: f32 = 1.0;
const LR_BUFFER: usize = 20;
const LARGE_POS_NUM_f32: f32 = 1.0e9;
const LARGE_POS_NUM_u64: u64 = 1000000000;

pub fn loss_wrapper(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f32>,

    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,

    Wseqs: &[arrayfire::Seq<i32>; 1],
    Hseqs: &[arrayfire::Seq<i32>; 1],
    Aseqs: &[arrayfire::Seq<i32>; 1],
    Bseqs: &[arrayfire::Seq<i32>; 1],
    Cseqs: &[arrayfire::Seq<i32>; 1],
    Dseqs: &[arrayfire::Seq<i32>; 1],
    Eseqs: &[arrayfire::Seq<i32>; 1],
    network_params: &arrayfire::Array<f32>,

    idxsel: &arrayfire::Array<i32>,
    Y: &arrayfire::Array<f32>,
    eval_metric: impl Fn(&arrayfire::Array<f32>, &arrayfire::Array<f32>) -> f32 + Copy,

    Z: &mut arrayfire::Array<f32>,
    Q: &mut arrayfire::Array<f32>,
    loss_output: &mut f32,
) {
    state_space_forward_batch(
        netdata,
        X,
        WRowIdxCSR,
        WColIdx,
        Wseqs,
        Hseqs,
        Aseqs,
        Bseqs,
        Cseqs,
        Dseqs,
        Eseqs,
        network_params,
        Z,
        Q,
    );

    let batch_size: u64 = netdata.batch_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let Qslices: u64 = Q.dims()[2];

    //Get Yhat
    let mut idxrs = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0f32, (batch_size - 1) as f32, 1.0);
    let seq2 = arrayfire::Seq::new((proc_num - 1) as f32, (Qslices - 1) as f32, 1.0);
    idxrs.set_index(idxsel, 0, None);
    idxrs.set_index(&seq1, 1, None);
    idxrs.set_index(&seq2, 2, None);
    let Yhat = arrayfire::index_gen(&Q, idxrs);
    *loss_output = eval_metric(&Yhat, Y);
}

// Train one neural network using ADAM and BTLS

//           Input arguments
// traindata_X: Training input array to the neural network
// 				traindata_X.dims()[0]  input feature size
// 				traindata_X.dims()[1]  batch size
// 				traindata_X.dims()[2]  sequence/traj size
// traindata_Y: Training ground truth output array
// 				traindata_Y.dims()[0]  output feature size
// 				traindata_Y.dims()[1]  batch size
// 				traindata_Y.dims()[2]  sequence/traj size
// eval_metric: Function with inputs X and Yhat that gives evaluation metrics
// eval_metric_grad: Gradient of eval_metric

//          Output argumenets
// stop_strategy:  How to train the network: Early stopping, cosine annealing, none
// alpha_max_vec: Vector of maximum learning rates for each epoch
// loss_vec: Training loss of the neural network
// neural_network:  Entire neural network archtecture
// loss_status: Loss convergence based on training loss
pub fn train_network(
    traindata_X: &nohash_hasher::IntMap<u64, Vec<f32>>,
    traindata_Y: &nohash_hasher::IntMap<u64, Vec<f32>>,

    validationdata_X: &nohash_hasher::IntMap<u64, Vec<f32>>,
    validationdata_Y: &nohash_hasher::IntMap<u64, Vec<f32>>,

    eval_metric: impl Fn(&arrayfire::Array<f32>, &arrayfire::Array<f32>) -> f32 + Copy,
    eval_metric_grad: impl Fn(&arrayfire::Array<f32>, &arrayfire::Array<f32>) -> arrayfire::Array<f32>
        + Copy,

    train_network_options: train_network_options_type,

    alpha_max_vec: &mut Vec<f32>,
    loss_vec: &mut Vec<f32>,
    crossval_vec: &mut Vec<f32>,
    arch_search: &mut arch_search_type,
    loss_status: &mut loss_status_type,
) {
    *loss_status = loss_status_type::NO_CONVERGENCE;

    let stop_strategy = train_network_options.stop_strategy;
    let lr_strategy = train_network_options.lr_strategy;
    let lr_strategy2 = train_network_options.lr_strategy2;

    let max_epoch = train_network_options.max_epoch;

    let stop_epoch = train_network_options.stop_epoch;
    let stop_train_loss = train_network_options.stop_train_loss;

    let exit_counter_threshold = train_network_options.exit_counter_threshold;
    let shuffle_counter_threshold = train_network_options.shuffle_counter_threshold;

    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let mut active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f32 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f32 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f32 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f32 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f32 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f32 = (*arch_search).neural_network.netdata.con_rad.clone();
    let center_const: f32 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f32 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f32 = (*arch_search).neural_network.netdata.repel_const.clone();

    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let traj_size: u64 = (traindata_X[&0].len() as u64) / (input_size * batch_size);

    let traj_steps = traj_size + proc_num - 1;

    let Z_dims = arrayfire::Dim4::new(&[neuron_size, batch_size, traj_steps, 1]);
    let mut Z = arrayfire::constant::<f32>(0.0, Z_dims);
    let mut Q = arrayfire::constant::<f32>(0.0, Z_dims);

    let mut alpha: f32 = 0.0001;

    active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(
        &((*arch_search).neural_network.neuron_idx),
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let Qslices: u64 = Q.dims()[2];

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let mut total_param_size = (*arch_search).neural_network.network_params.dims()[0];
    let mut mt_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);
    let mut mt = arrayfire::constant::<f32>(0.0, mt_dims);
    let mut vt = arrayfire::constant::<f32>(0.0, mt_dims);
    let mut grad = arrayfire::constant::<f32>(0.0, mt_dims);

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
        (*arch_search).neural_network.WColIdx.dims()[0],
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

    let train_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
    let mut X = arrayfire::constant::<f32>(0.0, temp_dims);
    let Y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
    let mut Y = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut batch_idx = 0;

    // epoch_num = number of batches
    let epoch_num = traindata_X.len() as u64;

    X = arrayfire::Array::new(&traindata_X[&batch_idx], train_X_dims);

    Y = arrayfire::Array::new(&traindata_Y[&batch_idx], Y_dims);

    let mut loss_val = LARGE_POS_NUM_f32;

    loss_wrapper(
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
        &idxsel,
        &Y,
        eval_metric,
        &mut Z,
        &mut Q,
        &mut loss_val,
    );

    let first_loss = loss_val.clone();

    state_space_backward_group2(
        &((*arch_search).neural_network.netdata),
        &X,
        &((*arch_search).neural_network.network_params),
        &Z,
        &Q,
        &Y,
        eval_metric_grad,
        &((*arch_search).neural_network.neuron_idx),
        &idxsel_out,
        &valsel_out,
        &cvec_out,
        &dXsel_out,
        &nrows_out,
        &sparseval_out,
        &sparserow_out,
        &sparsecol_out,
        &Hidxsel_out,
        &Aidxsel_out,
        &Bidxsel_out,
        &Cidxsel_out,
        &Didxsel_out,
        &Eidxsel_out,
        &combidxsel_out,
        &dAseqs_out,
        &dBseqs_out,
        &dCseqs_out,
        &dDseqs_out,
        &dEseqs_out,
        &mut grad,
    );

    grad = -1.0f32 * grad;

    let mut global_alpha_max = LR_MAX;

    *loss_vec = Vec::new();
    *crossval_vec = Vec::new();
    let mut alpha_history_vec: Vec<f32> = Vec::new();
    let mut idx_history_vec: Vec<u64> = Vec::new();

    if alpha_max_vec.len() < LR_BUFFER {
        for k in 0..LR_BUFFER {
            alpha_max_vec.push(LR_MAX);
        }
    } else {
        let mut minelem = alpha_max_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        if minelem.is_infinite() || minelem.is_nan() {
            minelem = LR_MAX;
        }

        global_alpha_max = minelem;
    }

    let mut alpha_idx = 0;

    let mut cur_alpha_max = alpha_max_vec[0];

    let mut mean_loss = first_loss.clone();
    let mut min_loss = first_loss.clone();
    let mut loss_counter = 0;

    let mut eval_metric_out = Vec::new();
    let mut Yhat_out = nohash_hasher::IntMap::default();

    let mut avgelem = LARGE_POS_NUM_f32;
    let mut crossval_mean_loss = LARGE_POS_NUM_f32;

    //TRAINING REGIMENT
    match stop_strategy {
        stop_strategy_type::STOP_AT_EPOCH => (),
        stop_strategy_type::STOP_AT_TRAIN_LOSS => (),
        stop_strategy_type::CROSSVAL_STOPPING => {
            eval_metric_out = Vec::new();
            Yhat_out = nohash_hasher::IntMap::default();

            validate_network(
                validationdata_X,
                validationdata_Y,
                eval_metric,
                arch_search,
                &mut Yhat_out,
                &mut eval_metric_out,
            );

            avgelem = eval_metric_out.iter().sum::<f32>() / (eval_metric_out.len() as f32);

            crossval_mean_loss = avgelem;

            crossval_vec.push(crossval_mean_loss);
        }
        stop_strategy_type::NONE => (),
    }

    arrayfire::device_gc();
    // main loop here!
    for i in 0..max_epoch {
        batch_idx = i % epoch_num;
        //X is our traininig data feature
        //in each epoch we choose one batch of training data
        X = arrayfire::Array::new(&traindata_X[&batch_idx], train_X_dims);

        Y = arrayfire::Array::new(&traindata_Y[&batch_idx], Y_dims);

        alpha = cur_alpha_max.clone();

        match lr_strategy2 {
            lr_strategy2_type::BTLS_ALPHA => {
                statespace_BTLS(
                    1.5f32 * first_loss,
                    alpha.clone() / 10000.0,
                    &((*arch_search).neural_network.network_params),
                    &grad,
                    0.5,
                    0.1,
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
                    &idxsel,
                    &Y,
                    eval_metric,
                    eval_metric_grad,
                    &((*arch_search).neural_network.neuron_idx),
                    &idxsel_out,
                    &valsel_out,
                    &cvec_out,
                    &dXsel_out,
                    &nrows_out,
                    &sparseval_out,
                    &sparserow_out,
                    &sparsecol_out,
                    &Hidxsel_out,
                    &Aidxsel_out,
                    &Bidxsel_out,
                    &Cidxsel_out,
                    &Didxsel_out,
                    &Eidxsel_out,
                    &combidxsel_out,
                    &dAseqs_out,
                    &dBseqs_out,
                    &dCseqs_out,
                    &dDseqs_out,
                    &dEseqs_out,
                    &mut Z,
                    &mut Q,
                    &mut alpha,
                    &mut loss_val,
                );
            }
            lr_strategy2_type::MAX_ALPHA => (),
        };

        //GET CURRENT MAX LR
        alpha_idx = (i / epoch_num) as usize;

        if (alpha_idx + 5) >= alpha_max_vec.len() {
            let mut minelem = alpha_max_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));

            if minelem.is_infinite() || minelem.is_nan() {
                minelem = LR_MAX;
            }

            global_alpha_max = minelem;

            for k in 0..((alpha_idx + 10) - alpha_max_vec.len()) {
                alpha_max_vec.push(global_alpha_max);
            }
        }

        //OVERFLOW LOSS
        if (loss_val > 1.5f32 * first_loss) || (loss_val.is_nan()) || (loss_val.is_infinite()) {
            let mut elem_num = 3;
            let mut history_len = alpha_history_vec.len();

            if history_len == 0 {
                loss_vec.push(first_loss.clone());
                alpha_history_vec.push(alpha_max_vec[0].clone());
                idx_history_vec.push(0);
                history_len = 1;
            }

            if elem_num > history_len {
                elem_num = history_len;
            }

            let lastelems = &alpha_history_vec[history_len - elem_num..history_len];

            let lastelems2 = &idx_history_vec[history_len - elem_num..history_len];

            let lastelems_arr = arrayfire::Array::new(
                &lastelems,
                arrayfire::Dim4::new(&[lastelems.len() as u64, 1, 1, 1]),
            );
            let (_, max_idx) = arrayfire::imax(&lastelems_arr, 0);

            let mut max_idx_cpu = vec![u32::default(); max_idx.elements()];
            max_idx.host(&mut max_idx_cpu);

            let maxelem = lastelems[max_idx_cpu[0] as usize];

            let maxindex = lastelems2[max_idx_cpu[0] as usize];

            alpha_idx = (maxindex / epoch_num) as usize;

            let newalpha = (0.8f32 * alpha_max_vec[alpha_idx]).min(maxelem * 0.8f32);

            for zz in alpha_idx..alpha_max_vec.len() {
                alpha_max_vec[zz] = newalpha;
            }

            *loss_status = loss_status_type::LOSS_OVERFLOW;
            return;
        }

        //LOSS PLATAUE
        mean_loss = 0.9 * mean_loss + 0.1 * loss_val;

        if (mean_loss * 1.05 < min_loss) {
            min_loss = mean_loss;

            loss_counter = 0;
        } else {
            loss_counter = loss_counter + 1;
        }

        //STOPPING POINT
        match stop_strategy {
            stop_strategy_type::STOP_AT_EPOCH => {
                if (i == stop_epoch) {
                    *loss_status = loss_status_type::PREDETERMINED_STOP;
                    return;
                }
            }
            stop_strategy_type::STOP_AT_TRAIN_LOSS => {
                if (mean_loss < stop_train_loss) {
                    *loss_status = loss_status_type::PREDETERMINED_STOP;
                    return;
                }
            }
            stop_strategy_type::CROSSVAL_STOPPING => {
                if ((i % epoch_num) == 0) && (i > 0) {
                    eval_metric_out = Vec::new();
                    Yhat_out = nohash_hasher::IntMap::default();

                    validate_network(
                        validationdata_X,
                        validationdata_Y,
                        eval_metric,
                        arch_search,
                        &mut Yhat_out,
                        &mut eval_metric_out,
                    );

                    avgelem = eval_metric_out.iter().sum::<f32>() / (eval_metric_out.len() as f32);

                    crossval_mean_loss = avgelem;

                    crossval_vec.push(crossval_mean_loss);
                }
            }
            stop_strategy_type::NONE => (),
        }

        //EXIT WHEN ANNEALING NOT WORKING
        if loss_counter >= exit_counter_threshold {
            *loss_status = loss_status_type::LOSS_PLATEAU;
            return;
        }

        cur_alpha_max = alpha_max_vec[alpha_idx];
        if alpha > cur_alpha_max {
            alpha = cur_alpha_max;
        }

        loss_vec.push(loss_val);
        alpha_history_vec.push(alpha);
        idx_history_vec.push(i.clone());

        //Reset connections
        match lr_strategy {
            lr_strategy_type::COSINE_ANNEALING => {
                if (loss_counter >= shuffle_counter_threshold) {
                    alpha = 4.0f32
                        * alpha
                        * ((2.0f32
                            * std::f32::consts::PI
                            * ((i as f32) / (shuffle_counter_threshold as f32)))
                            .cos()
                            .abs());
                }
            }
            lr_strategy_type::SHUFFLE_CONNECTIONS => (),
            lr_strategy_type::NONE => (),
        };

        (*arch_search).neural_network.network_params =
            (*arch_search).neural_network.network_params.clone() + (alpha * grad.clone());

        loss_wrapper(
            &((*arch_search).neural_network.netdata),
            //here X is one batch of training data
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
            &idxsel,
            &Y,
            eval_metric,
            &mut Z,
            &mut Q,
            //loss_val is not loss of whole dataset, but the loss of one batch
            &mut loss_val,
        );

        //Reset connections
        match lr_strategy {
            lr_strategy_type::COSINE_ANNEALING => (),
            lr_strategy_type::SHUFFLE_CONNECTIONS => {
                if (loss_counter == shuffle_counter_threshold) {
                    /*
                    let connection_num = WRowIdxCOO.dims()[0];


                    let network_info = network_info_seed_type {
                        i: i.clone(),
                        proc_num: proc_num.clone(),
                        connection_num: connection_num.clone(),
                        active_size: active_size.clone(),
                    };

                    set_network_seed(&network_info);




                    println!("shuffle {}", i);




                    //Find all cube paths
                    let cube_path_vec = find_cube_paths(&arch_search.evolution_info.dir_path  );

                    let cube_path = cube_path_vec[0].clone();

                    let cube_radius= extract_file_info2(&cube_path)[0];

                    //Load cube pos
                    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
                    let mut cube_glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
                    let mut cube_neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);
                    let mut cube_neuron_idx = arrayfire::constant::<i32>(0,temp_dims);

                    load_network_structure(
                        &cube_path,

                        &mut cube_glia_pos,
                        &mut cube_neuron_pos,
                        &mut cube_neuron_idx
                    );
                    drop(cube_glia_pos);
                    drop(cube_neuron_idx);


                    //Add new neurons
                    add_neuron_to_existing2(
                        &cube_neuron_pos,
                        cube_radius,

                        active_size+2,

                        arch_search,
                    );
                    (*arch_search).neural_network.netdata.active_size = (*arch_search).neural_network.neuron_idx.dims()[0];


                    WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));











                    //Make the neural network smaller

                    reduce_network_size(
                        active_size,

                        arch_search,
                    );
                    (*arch_search).neural_network.netdata.active_size = (*arch_search).neural_network.neuron_idx.dims()[0];


                    WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

                    */

                    shuffle_weights(i, arch_search);

                    (*arch_search).neural_network.netdata.active_size =
                        (*arch_search).neural_network.neuron_idx.dims()[0];
                    active_size = (*arch_search).neural_network.netdata.active_size;
                    WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

                    total_param_size = (*arch_search).neural_network.network_params.dims()[0];
                    mt_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);
                    mt = arrayfire::constant::<f32>(0.0, mt_dims);
                    vt = arrayfire::constant::<f32>(0.0, mt_dims);
                    grad = arrayfire::constant::<f32>(0.0, mt_dims);

                    idxsel_out = nohash_hasher::IntMap::default();
                    valsel_out = nohash_hasher::IntMap::default();

                    cvec_out = nohash_hasher::IntMap::default();
                    dXsel_out = nohash_hasher::IntMap::default();

                    nrows_out = nohash_hasher::IntMap::default();
                    sparseval_out = nohash_hasher::IntMap::default();
                    sparsecol_out = nohash_hasher::IntMap::default();
                    sparserow_out = nohash_hasher::IntMap::default();

                    Wseqs = [arrayfire::Seq::default()];
                    Hseqs = [arrayfire::Seq::default()];
                    Aseqs = [arrayfire::Seq::default()];
                    Bseqs = [arrayfire::Seq::default()];
                    Cseqs = [arrayfire::Seq::default()];
                    Dseqs = [arrayfire::Seq::default()];
                    Eseqs = [arrayfire::Seq::default()];

                    Hidxsel_out = nohash_hasher::IntMap::default();
                    Aidxsel_out = nohash_hasher::IntMap::default();
                    Bidxsel_out = nohash_hasher::IntMap::default();
                    Cidxsel_out = nohash_hasher::IntMap::default();
                    Didxsel_out = nohash_hasher::IntMap::default();
                    Eidxsel_out = nohash_hasher::IntMap::default();
                    combidxsel_out = nohash_hasher::IntMap::default();

                    dAseqs_out = nohash_hasher::IntMap::default();
                    dBseqs_out = nohash_hasher::IntMap::default();
                    dCseqs_out = nohash_hasher::IntMap::default();
                    dDseqs_out = nohash_hasher::IntMap::default();
                    dEseqs_out = nohash_hasher::IntMap::default();

                    find_path_backward_group2(
                        &((*arch_search).neural_network.netdata),
                        traj_steps,
                        traj_size,
                        &WRowIdxCOO,
                        &((*arch_search).neural_network.WColIdx),
                        &((*arch_search).neural_network.neuron_idx),
                        (*arch_search).neural_network.WColIdx.dims()[0],
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

                    WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));
                }
            }
            lr_strategy_type::NONE => (),
        }

        state_space_backward_group2(
            &((*arch_search).neural_network.netdata),
            &X,
            &((*arch_search).neural_network.network_params),
            &Z,
            &Q,
            &Y,
            eval_metric_grad,
            &((*arch_search).neural_network.neuron_idx),
            &idxsel_out,
            &valsel_out,
            &cvec_out,
            &dXsel_out,
            &nrows_out,
            &sparseval_out,
            &sparserow_out,
            &sparsecol_out,
            &Hidxsel_out,
            &Aidxsel_out,
            &Bidxsel_out,
            &Cidxsel_out,
            &Didxsel_out,
            &Eidxsel_out,
            &combidxsel_out,
            &dAseqs_out,
            &dBseqs_out,
            &dCseqs_out,
            &dDseqs_out,
            &dEseqs_out,
            &mut grad,
        );

        grad = -1.0f32 * grad;

        adam(0.9, 0.999, &mut grad, &mut mt, &mut vt);

        println!("loss: {}, alpha0: {}, i: {}", loss_val, alpha, i);
    }
}
