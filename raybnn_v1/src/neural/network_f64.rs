extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

use crate::neural::activation_f64::deriUAF;
use crate::neural::activation_f64::UAF;

use crate::graph::large_sparse_i32::COO_batch_find;

use crate::graph::tree_i32::find_unique;
use serde::{Deserialize, Serialize};

const COO_find_limit: u64 = 1500000000;

const one: f64 = 1.0;

#[derive(Serialize, Deserialize)]
pub struct network_metadata_type {
    pub neuron_size: u64,
    pub input_size: u64,
    pub output_size: u64,
    pub proc_num: u64,
    pub active_size: u64,
    pub space_dims: u64,
    pub step_num: u64,
    pub batch_size: u64,
    pub del_unused_neuron: bool,
    pub time_step: f64,
    pub nratio: f64,
    pub neuron_std: f64,
    pub sphere_rad: f64,
    pub neuron_rad: f64,
    pub con_rad: f64,
    pub init_prob: f64,
    pub add_neuron_rate: f64,
    pub del_neuron_rate: f64,
    pub center_const: f64,
    pub spring_const: f64,
    pub repel_const: f64,
}

#[derive(Serialize, Deserialize)]
pub struct neural_network_type {
    pub netdata: network_metadata_type,
    pub WRowIdxCSR: arrayfire::Array<i32>,
    pub WColIdx: arrayfire::Array<i32>,
    pub network_params: arrayfire::Array<f64>,
    pub glia_pos: arrayfire::Array<f64>,
    pub neuron_pos: arrayfire::Array<f64>,
    pub neuron_idx: arrayfire::Array<i32>,
}

pub fn print_netdata(netdata: &network_metadata_type) {
    println!("\n\n******Network Information******");
    println!("neuron_size: {}", netdata.neuron_size);
    println!("input_size: {}", netdata.input_size);
    println!("output_size: {}", netdata.output_size);
    println!("proc_num: {}", netdata.proc_num);
    println!("active_size: {}", netdata.active_size);
    println!("space_dims: {}", netdata.space_dims);
    println!("step_num: {}", netdata.step_num);
    println!("batch_size: {}", netdata.batch_size);
    println!("del_unused_neuron: {}", netdata.del_unused_neuron);

    println!("time_step: {}", netdata.time_step);
    println!("nratio: {}", netdata.nratio);
    println!("neuron_std: {}", netdata.neuron_std);
    println!("sphere_rad: {}", netdata.sphere_rad);
    println!("neuron_rad: {}", netdata.neuron_rad);
    println!("con_rad: {}", netdata.con_rad);
    println!("init_prob: {}", netdata.init_prob);
    println!("add_neuron_rate: {}", netdata.add_neuron_rate);
    println!("del_neuron_rate: {}", netdata.del_neuron_rate);
    println!("center_const: {}", netdata.center_const);
    println!("spring_const: {}", netdata.spring_const);
    println!("repel_const: {}", netdata.repel_const);
    println!("************\n\n");
}

pub fn clone_netdata(netdata: &network_metadata_type) -> network_metadata_type {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let add_neuron_rate: f64 = netdata.add_neuron_rate.clone();
    let del_neuron_rate: f64 = netdata.del_neuron_rate.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let newnetdata = network_metadata_type {
        neuron_size: neuron_size,
        input_size: input_size,
        output_size: output_size,
        proc_num: proc_num,
        active_size: active_size,
        space_dims: space_dims,
        step_num: step_num,
        batch_size: batch_size,
        del_unused_neuron: del_unused_neuron,

        time_step: time_step,
        nratio: nratio,
        neuron_std: neuron_std,
        sphere_rad: sphere_rad,
        neuron_rad: neuron_rad,
        con_rad: con_rad,
        init_prob: init_prob,
        add_neuron_rate: add_neuron_rate,
        del_neuron_rate: del_neuron_rate,
        center_const: center_const,
        spring_const: spring_const,
        repel_const: repel_const,
    };

    newnetdata
}

pub fn clone_neural_network(neural_network: &neural_network_type) -> neural_network_type {
    let mut new_neural_network = neural_network_type {
        netdata: clone_netdata(&neural_network.netdata),
        WRowIdxCSR: neural_network.WRowIdxCSR.clone(),
        WColIdx: neural_network.WColIdx.clone(),
        network_params: neural_network.network_params.clone(),
        glia_pos: neural_network.glia_pos.clone(),
        neuron_pos: neural_network.neuron_pos.clone(),
        neuron_idx: neural_network.neuron_idx.clone(),
    };

    new_neural_network
}

pub fn create_nullnetdata() -> network_metadata_type {
    let netdata: network_metadata_type = network_metadata_type {
        neuron_size: 0,
        input_size: 0,
        output_size: 0,
        proc_num: 0,
        active_size: 0,
        space_dims: 0,
        step_num: 0,
        batch_size: 0,
        del_unused_neuron: false,

        time_step: 0.0,
        nratio: 0.0,
        neuron_std: 0.0,
        sphere_rad: 0.0,
        neuron_rad: 0.0,
        con_rad: 0.0,
        init_prob: 0.0,
        add_neuron_rate: 0.0,
        del_neuron_rate: 0.0,
        center_const: 0.0,
        spring_const: 0.0,
        repel_const: 0.0,
    };

    netdata
}

pub fn UAF_initial_as_identity(
    netdata: &network_metadata_type,
    //H: &mut arrayfire::Array<f64>,
    A: &mut arrayfire::Array<f64>,
    B: &mut arrayfire::Array<f64>,
    C: &mut arrayfire::Array<f64>,
    D: &mut arrayfire::Array<f64>,
    E: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let H_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    //  *H = 0.001*neuron_std*arrayfire::randn::<f64>(H_dims);
    *A = one + 0.0001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *B = 0.0001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *C = 0.00001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *D = -one + 0.00001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *E = 0.00001 * neuron_std * arrayfire::randn::<f64>(H_dims);
}

pub fn UAF_initial_as_tanh(
    netdata: &network_metadata_type,
    //H: &mut arrayfire::Array<f64>,
    A: &mut arrayfire::Array<f64>,
    B: &mut arrayfire::Array<f64>,
    C: &mut arrayfire::Array<f64>,
    D: &mut arrayfire::Array<f64>,
    E: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let H_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    //   *H = 0.0000001*neuron_std*arrayfire::randn::<f64>(H_dims);
    *A = 2.12616013f64 + 0.0001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *B = (1.0f64 / 2.12616013f64) + 0.0001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *C = 0.0000001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *D = 2.12616013f64 + 0.0001 * neuron_std * arrayfire::randn::<f64>(H_dims);
    *E = -1.0f64 + 0.0001 * neuron_std * arrayfire::randn::<f64>(H_dims);
}

pub fn xavier_init(
    in_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,

    WValues: &mut arrayfire::Array<f64>,
    H: &mut arrayfire::Array<f64>,
) {
    *WValues = 0.000001f64 * arrayfire::randn::<f64>(WValues.dims());

    let H_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    *H = 0.000001f64 * arrayfire::randn::<f64>(H_dims);

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut out_idx = in_idx.clone();
    let mut valsel = arrayfire::constant::<i32>(0, temp_dims);

    let COO_batch_size = 1 + ((COO_find_limit / WColIdx.dims()[0]) as u64);

    let mut input_degree = 0;
    let mut output_degree = 0;

    for i in 0..depth {
        valsel = COO_batch_find(WColIdx, &out_idx, COO_batch_size);
        if valsel.dims()[0] == 0 {
            break;
        }
        out_idx = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        out_idx = find_unique(&out_idx, neuron_size);

        output_degree = valsel.dims()[0];

        let mulitiplier = (6.0f64 / ((input_degree + output_degree) as f64)).sqrt() * 2.0f64;
        let mut newWValues = arrayfire::randu::<f64>(valsel.dims());
        newWValues = (newWValues - 0.5f64) * (mulitiplier);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel, 0, None);
        arrayfire::assign_gen(WValues, &idxrs, &newWValues);

        drop(newWValues);
        drop(idxrs);

        let mut newH = arrayfire::randu::<f64>(out_idx.dims());
        newH = (newH - 0.5f64) * (mulitiplier);

        let mut idxrs2 = arrayfire::Indexer::default();
        idxrs2.set_index(&out_idx, 0, None);
        arrayfire::assign_gen(H, &idxrs2, &newH);

        if out_idx.dims()[0] == 0 {
            break;
        }

        input_degree = valsel.dims()[0];
    }
}

/*
Forward pass using CSR weighted adjacency sparse matrices and UAF.
Generates all internal states and the neural network output


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
WRowIdxCSR:          Row sparse matrix of the weighted adjacency matrix
WColIdx:             Column sparse matrix of the weighted adjacency matrix
Wseqs:               Indexes of the weight parameters
Hseqs:               Indexes of the bias parameters
Aseqs:               Indexes of the UAF A parameters
Bseqs:               Indexes of the UAF B parameters
Cseqs:               Indexes of the UAF C parameters
Dseqs:               Indexes of the UAF D parameters
Eseqs:               Indexes of the UAF E parameters
network_params:      All trainable parameters in the network


Outputs:
Z:                   Internal State Matrix Z
Q:                   Internal State Matrix Q

*/

pub fn state_space_forward_batch(
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
    network_params: &arrayfire::Array<f64>,

    Z: &mut arrayfire::Array<f64>,
    Q: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    //let proc_num: u64 = netdata.proc_num.clone();
    let input_size: u64 = netdata.input_size.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let Zslices: i64 = Z.dims()[2] as i64;

    let X_slices: i64 = X.dims()[2] as i64;

    let S_dims = arrayfire::Dim4::new(&[neuron_size, batch_size, 1, 1]);
    let mut S = arrayfire::constant::<f64>(0.0, S_dims);

    let mut tempx = arrayfire::slice(X, 0);
    let seqs = &[
        arrayfire::Seq::new(0.0f64, (input_size - 1) as f64, 1.0f64),
        arrayfire::Seq::default(),
    ];

    let H = arrayfire::index(&network_params, Hseqs);
    let A = arrayfire::index(&network_params, Aseqs);
    let B = arrayfire::index(&network_params, Bseqs);
    let C = arrayfire::index(&network_params, Cseqs);
    let D = arrayfire::index(&network_params, Dseqs);
    let E = arrayfire::index(&network_params, Eseqs);

    let WValues = arrayfire::index(network_params, Wseqs);

    let W = arrayfire::sparse::<f64>(
        neuron_size,
        neuron_size,
        &WValues,
        WRowIdxCSR,
        WColIdx,
        arrayfire::SparseFormat::CSR,
    );

    for i in 0i64..Zslices {
        if X_slices > 1 {
            tempx = arrayfire::slice(X, i);
            arrayfire::assign_seq(&mut S, seqs, &tempx);
            drop(tempx);
        } else {
            arrayfire::assign_seq(&mut S, seqs, &X);
        }

        S = arrayfire::matmul(&W, &S, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);
        S = arrayfire::add(&S, &H, true);
        arrayfire::set_slice(Z, &S, i);

        S = UAF(&S, &A, &B, &C, &D, &E);
        arrayfire::set_slice(Q, &S, i);
    }
}

/*
Backward pass using CSR weighted adjacency sparse matrices and UAF.
Generates the gradients of the sparse weighted adjacency matrix


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
network_params:      All trainable parameters in the network
Z:                   Internal State matrix Z
Q:                   Internal State matrix Q
Y:                   Target Matrix to fit
loss_grad:           Loss function
neuron_idx:          Indexes of the neurons
idxsel_out:          Indexes of the UAF parameters
valsel_out:          Indexes of the UAF values
cvec_out:            Indexes of the UAF column sparse vector
dXsel_out:           Indexes of the dX values
nrows_out:           Number of rows in UAF
sparseval_out:       Indexes of the values in the sparse matrix
sparserow_out:       Indexes of the rows in the sparse matrix
sparsecol_out:       Indexes of the columns in the sparse matrix
Hidxsel_out:         Indexes of the bias vector H
Aidxsel_out:         Indexes of the UAF vector A
Bidxsel_out:         Indexes of the UAF vector B
Cidxsel_out:         Indexes of the UAF vector C
Didxsel_out:         Indexes of the UAF vector D
Eidxsel_out:         Indexes of the UAF vector E
combidxsel_out:      Indexes of the all vectors
dAseqs_out:          Indexes of the dA
dBseqs_out:          Indexes of the dB
dCseqs_out:          Indexes of the dC
dDseqs_out:          Indexes of the dD
dEseqs_out:          Indexes of the dE


Outputs:
grad:                   Gradient of all trainable parameters


*/

pub fn state_space_backward_group2(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f64>,

    network_params: &arrayfire::Array<f64>,

    Z: &arrayfire::Array<f64>,
    Q: &arrayfire::Array<f64>,
    Y: &arrayfire::Array<f64>,
    loss_grad: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> arrayfire::Array<f64>,
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

    grad: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();

    let batch_size: u64 = netdata.batch_size.clone();

    //Set output to zero
    *grad = arrayfire::constant::<f64>(0.0, network_params.dims());

    //Get current selection of neurons
    let active_size = neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );

    let Qslices: u64 = Q.dims()[2];
    let Yslices: u64 = Y.dims()[2];

    //Get Yhat
    let mut idxrs = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0f64, (batch_size - 1) as f64, 1.0);
    let seq2 = arrayfire::Seq::new((proc_num - 1) as f64, (Qslices - 1) as f64, 1.0);
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&seq1, 1, None);
    idxrs.set_index(&seq2, 2, None);
    let Yhat = arrayfire::index_gen(Q, idxrs);
    drop(idxsel);

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let S_dims = arrayfire::Dim4::new(&[neuron_size, batch_size, 1, 1]);

    //Calculate error
    let total_error = loss_grad(&Yhat, Y);
    let mut yslicidx: i64 = (Yslices - 1) as i64;
    let mut error = arrayfire::slice(&total_error, yslicidx);

    let Zslices: i64 = Z.dims()[2] as i64;

    let X_slices: i64 = X.dims()[2] as i64;

    let mut inx = arrayfire::constant::<f64>(0.0, S_dims);

    let mut tempx = arrayfire::slice(X, 0);

    let seqs = &[
        arrayfire::Seq::new(0.0f64, (input_size - 1) as f64, 1.0f64),
        arrayfire::Seq::default(),
    ];

    let mut Xtemp = arrayfire::slice(Z, 0);

    let mut sA = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut sB = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut sC = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut sD = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut sE = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut dX = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut dA = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut dB = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut dC = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut dD = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut dE = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut tempW = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut gtemperr = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut tempinx = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut tempdX = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut tempgW = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut tempgH = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut derror = arrayfire::slice(&total_error, 0);

    let batchseq = arrayfire::Seq::new(0.0f64, (batch_size - 1) as f64, 1.0);
    let mut sliceseq = arrayfire::Seq::new((proc_num - 1) as f64, (Qslices - 1) as f64, 1.0);

    let mut keys = arrayfire::constant::<i32>(0, temp_dims);
    let mut vals = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut UAFgroup = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut tileerror = arrayfire::constant::<f64>(0.0, temp_dims);
    let tileerror_dims = arrayfire::Dim4::new(&[5, 1, 1, 1]);

    //Main loop
    for i in (0i64..Zslices).rev() {
        //Select X value
        let mut idxrs = arrayfire::Indexer::default();
        sliceseq = arrayfire::Seq::new(i as f64, i as f64, 1.0);
        idxrs.set_index(&idxsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        idxrs.set_index(&sliceseq, 2, None);
        Xtemp = arrayfire::index_gen(Z, idxrs);

        //Get current UAF parameters
        sA = arrayfire::lookup(&network_params, &Aidxsel_out[&i], 0);

        sB = arrayfire::lookup(&network_params, &Bidxsel_out[&i], 0);

        sC = arrayfire::lookup(&network_params, &Cidxsel_out[&i], 0);

        sD = arrayfire::lookup(&network_params, &Didxsel_out[&i], 0);

        sE = arrayfire::lookup(&network_params, &Eidxsel_out[&i], 0);

        //Compute derivative of UAF
        deriUAF(
            &Xtemp, &sA, &sB, &sC, &sD, &sE, &mut dX, &mut dA, &mut dB, &mut dC, &mut dD, &mut dE,
        );
        drop(Xtemp);
        drop(sA);
        drop(sB);
        drop(sC);
        drop(sD);
        drop(sE);

        //Compute dX
        dX = arrayfire::mul(&dX, &error, false);

        //Update H
        tempgH = arrayfire::lookup(grad, &Hidxsel_out[&i], 0) + (arrayfire::sum(&dX, 1));

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&Hidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgH);
        drop(tempgH);

        //Join all

        UAFgroup = arrayfire::constant::<f64>(
            0.0,
            arrayfire::Dim4::new(&[dA.dims()[0] * 5, dA.dims()[1], 1, 1]),
        );
        arrayfire::assign_seq(&mut UAFgroup, &dAseqs_out[&i], &dA);
        arrayfire::assign_seq(&mut UAFgroup, &dBseqs_out[&i], &dB);
        arrayfire::assign_seq(&mut UAFgroup, &dCseqs_out[&i], &dC);
        arrayfire::assign_seq(&mut UAFgroup, &dDseqs_out[&i], &dD);
        arrayfire::assign_seq(&mut UAFgroup, &dEseqs_out[&i], &dE);

        tileerror = arrayfire::tile(&error, tileerror_dims);

        UAFgroup = arrayfire::mul(&tileerror, &UAFgroup, false);
        drop(tileerror);

        UAFgroup = arrayfire::sum(&UAFgroup, 1) + arrayfire::lookup(grad, &combidxsel_out[&i], 0);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&combidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &UAFgroup);
        drop(UAFgroup);

        //Get dX of each row
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&dXsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempdX = arrayfire::index_gen(&dX, idxrs);

        //Get input values
        inx = arrayfire::constant::<f64>(0.0, S_dims);

        if (i > 0) {
            inx = arrayfire::slice(Q, (i - 1));
        }

        if X_slices > 1 {
            tempx = arrayfire::slice(X, i);
            arrayfire::assign_seq(&mut inx, seqs, &tempx);
            drop(tempx);
        } else {
            arrayfire::assign_seq(&mut inx, seqs, &X);
        }

        //Upadate gW
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&cvec_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempinx = arrayfire::index_gen(&inx, idxrs);
        drop(inx);

        tempgW = arrayfire::mul(&tempdX, &tempinx, false);
        drop(tempinx);
        tempgW = (arrayfire::sum(&tempgW, 1)) + arrayfire::lookup(grad, &valsel_out[&i], 0);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgW);
        drop(tempgW);

        //Propagate Errors
        tempW = arrayfire::lookup(network_params, &sparseval_out[&i], 0);

        tempW = arrayfire::sparse::<f64>(
            nrows_out[&i],
            dX.dims()[0],
            &tempW,
            &sparserow_out[&i],
            &sparsecol_out[&i],
            arrayfire::SparseFormat::CSR,
        );

        error = arrayfire::matmul(
            &tempW,
            &dX,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        );

        drop(tempW);

        //Add new Y error
        if (yslicidx > 0) {
            yslicidx = yslicidx - 1;
            derror = arrayfire::slice(&total_error, yslicidx);

            error = arrayfire::join(0, &error, &derror);
        }
    }
}
