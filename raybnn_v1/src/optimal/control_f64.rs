extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

const one: f64 = 1.0;
const onehalf: f64 = 0.5;

use crate::interface::autotrain_f64::loss_wrapper;

use crate::neural::network_f64::network_metadata_type;

use crate::neural::network_f64::state_space_backward_group2;

use serde::{Deserialize, Serialize};

const LR_MAX: f64 = 1.0;
const LR_BUFFER: usize = 20;
const LARGE_POS_NUM_f64: f64 = 1.0e9;
const LARGE_POS_NUM_u64: u64 = 1000000000;

#[derive(Serialize, Deserialize)]
pub struct neural_controller_type {
    pub start_epoch: u64,
    pub end_epoch: u64,
    pub window_epoch: u64,
    pub counter0: u64,
    pub counter1: u64,

    pub min_loss: f64,
    pub mean_loss: f64,

    pub decrease_alpha: f64,
    pub increase_alpha: f64,
    pub max_alpha: f64,
    pub min_alpha: f64,
}

pub fn create_nullcontrol() -> neural_controller_type {
    let testdata: neural_controller_type = neural_controller_type {
        start_epoch: 0,
        end_epoch: LARGE_POS_NUM_u64,
        window_epoch: 100,
        counter0: 0,
        counter1: 0,

        min_loss: LARGE_POS_NUM_f64,
        mean_loss: LARGE_POS_NUM_f64,

        decrease_alpha: 0.7,
        increase_alpha: 1.2,
        max_alpha: LARGE_POS_NUM_f64,
        min_alpha: 0.0,
    };

    testdata
}

pub fn cosine_annealing(
    control_state: &mut neural_controller_type,
    alpha0: &mut f64,
    alpha1: &mut f64,
) {
    let window_epoch = (*control_state).window_epoch;
    let min_alpha = (*control_state).min_alpha;
    let max_alpha = (*control_state).max_alpha;

    (*control_state).counter0 = ((*control_state).counter0 + 1);

    if (*control_state).counter0 > (*control_state).start_epoch {
        (*control_state).counter1 = ((*control_state).counter1 + 1) % window_epoch;

        *alpha0 = min_alpha
            + (onehalf
                * (max_alpha - min_alpha)
                * (one
                    + ((((*control_state).counter1 as f64) / (window_epoch as f64))
                        * std::f64::consts::PI)
                        .cos()));

        *alpha1 = *alpha0;
    } else {
        *alpha0 = min_alpha;

        *alpha1 = *alpha0;
    }
}

pub fn plateau(
    loss_val: f64,
    control_state: &mut neural_controller_type,
    alpha0: &mut f64,
    alpha1: &mut f64,
) {
    (*control_state).mean_loss = 0.9 * ((*control_state).mean_loss) + 0.1 * loss_val;

    if ((*control_state).mean_loss * 1.05 < (*control_state).min_loss) {
        (*control_state).min_loss = (*control_state).mean_loss;
        (*control_state).counter0 = 0;
    }

    if ((*control_state).counter0 > (*control_state).window_epoch) {
        *alpha0 = (*alpha0) * ((*control_state).decrease_alpha);
        *alpha1 = (*alpha1) * ((*control_state).decrease_alpha);
        (*control_state).counter0 = 0;
    } else {
        (*control_state).counter0 = (*control_state).counter0 + 1;
    }

    if (*alpha0 < (*control_state).min_alpha) && (*alpha1 > 0.0) {
        *alpha0 = (*control_state).max_alpha;
        *alpha1 = 0.0;
    }
}

pub fn BTLS(
    loss: impl Fn(&arrayfire::Array<f64>) -> f64,
    loss_grad: impl Fn(&arrayfire::Array<f64>) -> arrayfire::Array<f64>,
    init_point: &arrayfire::Array<f64>,
    direction: &arrayfire::Array<f64>,
    gamma: f64,
    rho: f64,
) -> f64 {
    let mut alpha: f64 = LR_MAX;
    let init_loss = loss(init_point);

    let mut next_point = init_point.clone() + (alpha) * direction.clone();
    let mut f0 = loss(&next_point);

    let init_grad = loss_grad(init_point);
    let v0 = rho * (arrayfire::mul(direction, &init_grad, false));
    let (t0, t1) = arrayfire::sum_all(&v0);
    let mut f1 = init_loss.clone() + (alpha) * t0;

    while (f0 > f1) {
        alpha = (alpha) * gamma;
        next_point = init_point.clone() + (alpha) * direction.clone();
        f0 = loss(&next_point);
        f1 = init_loss.clone() + (alpha) * t0;
    }
    alpha
}

pub fn statespace_BTLS(
    max_loss: f64,
    min_alpha: f64,

    init_point: &arrayfire::Array<f64>,
    direction: &arrayfire::Array<f64>,
    gamma: f64,
    rho: f64,

    netdata: &network_metadata_type,
    X: &arrayfire::Array<f64>,

    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,

    Wseqs: &[arrayfire::Seq<i32>; 1],
    Hseqs: &[arrayfire::Seq<i32>; 1],
    Aseqs: &[arrayfire::Seq<i32>; 1],
    Bseqs: &[arrayfire::Seq<i32>; 1],
    Cseqs: &[arrayfire::Seq<i32>; 1],
    Dseqs: &[arrayfire::Seq<i32>; 1],
    Eseqs: &[arrayfire::Seq<i32>; 1],

    idxsel: &arrayfire::Array<i32>,
    Y: &arrayfire::Array<f64>,
    eval_metric: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> f64 + Copy,
    eval_metric_grad: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> arrayfire::Array<f64>
        + Copy,

    neuron_idx: &arrayfire::Array<i32>,

    idxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    valsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    cvec_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    dXsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    nrows_out: &nohash_hasher::IntMap<i64, u64>,
    sparseval_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    sparserow_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    sparsecol_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    Hidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Aidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Bidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Cidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Didxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    Eidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    combidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,

    dAseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dBseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dCseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dDseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,
    dEseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2]>,

    Z: &mut arrayfire::Array<f64>,
    Q: &mut arrayfire::Array<f64>,

    alpha: &mut f64,
    loss_output: &mut f64,
) {
    //let init_loss = loss(init_point);
    let mut init_loss = LARGE_POS_NUM_f64;

    loss_wrapper(
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
        init_point,
        idxsel,
        Y,
        eval_metric,
        Z,
        Q,
        &mut init_loss,
    );

    if (init_loss.is_nan()) || (init_loss.is_infinite()) {
        *loss_output = f64::INFINITY;
        return;
    }

    let mut init_grad = arrayfire::constant::<f64>(0.0, init_point.dims());

    state_space_backward_group2(
        netdata,
        X,
        init_point,
        Z,
        Q,
        Y,
        eval_metric_grad,
        neuron_idx,
        idxsel_out,
        valsel_out,
        cvec_out,
        dXsel_out,
        nrows_out,
        sparseval_out,
        sparserow_out,
        sparsecol_out,
        Hidxsel_out,
        Aidxsel_out,
        Bidxsel_out,
        Cidxsel_out,
        Didxsel_out,
        Eidxsel_out,
        combidxsel_out,
        dAseqs_out,
        dBseqs_out,
        dCseqs_out,
        dDseqs_out,
        dEseqs_out,
        &mut init_grad,
    );

    let mut next_point = init_point.clone() + (*alpha) * direction.clone();
    //let mut f0  = loss(&next_point);
    let mut f0 = LARGE_POS_NUM_f64;

    loss_wrapper(
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
        &next_point,
        idxsel,
        Y,
        eval_metric,
        Z,
        Q,
        &mut f0,
    );

    if (f0.is_nan()) || (f0.is_infinite()) {
        *loss_output = f64::INFINITY;
        return;
    }

    //let init_grad = loss_grad(init_point);

    let v0 = rho * (arrayfire::mul(direction, &init_grad, false));
    let (t0, t1) = arrayfire::sum_all(&v0);
    let mut f1 = init_loss.clone() + (*alpha) * t0;

    while (f0 > f1) || (f0 > max_loss) {
        *alpha = (*alpha) * gamma;

        if (*alpha) < min_alpha {
            break;
        }

        next_point = init_point.clone() + (*alpha) * direction.clone();
        //f0  = loss(&next_point);

        loss_wrapper(
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
            &next_point,
            idxsel,
            Y,
            eval_metric,
            Z,
            Q,
            &mut f0,
        );

        f1 = init_loss.clone() + (*alpha) * t0;
    }

    *loss_output = f0.clone();
}
