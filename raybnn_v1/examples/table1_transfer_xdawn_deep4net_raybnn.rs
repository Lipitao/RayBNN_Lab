/*
Data for Table 1: RayBNN in the EEG Motor Imagery Dataset

Run the 54 fold data for Table 1: RayBNN in the EEG Motor Imagery Dataset
Note that CUDA has compile the kernels at runtime so the first run is slower. Tested on RTX 3090.

Generated files
./info_*.csv          Information about the training time and number of parameters
./test_act_*.csv      Actual test results
./test_pred_*.csv     Predicted test results

*/

use arrayfire;
use nohash_hasher;
use raybnn;
use raybnn::export::dataloader_f32::load_network2;
use raybnn::interface::automatic_f32::arch_search_type;
use raybnn::optimal::evolution_f32::evolution_info_type;
use raybnn::optimal::evolution_f32::evolution_search_type;
use std::collections::HashMap;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use rayon::prelude::*;

use raybnn::{
    optimal::loss_f32::{MSE_grad, MSE},
    physics::update_f32::add_neuron_option_type,
};

fn sigmoid_loss(yhat: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>) -> f32 {
    raybnn::optimal::loss_f32::weighted_sigmoid_cross_entropy(yhat, y, 2.0)
}

fn sigmoid_loss_grad(
    yhat: &arrayfire::Array<f32>,
    y: &arrayfire::Array<f32>,
) -> arrayfire::Array<f32> {
    raybnn::optimal::loss_f32::weighted_sigmoid_cross_entropy_grad(yhat, y, 2.0)
}

#[allow(unused_must_use)]
fn main() {
    //Set CUDA and GPU Device 0
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let args: Vec<String> = std::env::args().collect();

    let input_dir = args[1].clone();

    let output_dir = args[2].clone();

    let fold = args[3].clone().parse::<u64>().unwrap();

    println!("fold {}", fold);

    arrayfire::set_seed(42);

    arrayfire::sync(DEVICE);
    arrayfire::device_gc();
    arrayfire::sync(DEVICE);

    let dir_path = "/tmp/".to_string();

    let dataset_dir = input_dir.clone();

    let shapefile = format!("{}/shape_{}.txt", dataset_dir, fold);
    let mut shape_info = raybnn::export::dataloader_u64::file_to_vec_cpu(&shapefile);
    println!("shape_info {:?}", shape_info);

    let mut CVLoss_exists = false;
    let save_filename = format!("{}/CVLOSS_{}_{}.csv", output_dir, fold, shape_info[1]);
    let mut min_idx = 5000;
    if std::path::Path::new(&save_filename).exists() {
        let CVLoss_vec = raybnn::export::dataloader_f32::file_to_vec_cpu(&save_filename);
        CVLoss_exists = true;
        let CVLoss_arr = arrayfire::Array::new(
            &CVLoss_vec,
            arrayfire::Dim4::new(&[CVLoss_vec.len() as u64, 1, 1, 1]),
        );
        (_, _, min_idx) = arrayfire::imin_all(&CVLoss_arr);
    }
    let min_idx = (min_idx) as u64;

    //Set More Neural Network Parameters

    let max_input_size: u64 = 2700;
    let mut input_size: u64 = shape_info[1];

    let max_output_size: u64 = 1;
    let output_size: u64 = 1;

    let max_neuron_size: u64 = 4000;
    let mut active_size = 1000;

    let mut batch_size: u64 = 16;
    let traj_size = 1;

    let mut proc_num = 3;

    let mut val_size = 0.001 * (shape_info[2] as f32);
    let mut val_batch_size: u64 = 1000;
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    //Create Start Neural Network
    println!("Creating Network");
    /*
    let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture2(
        input_size,
        max_input_size,


        output_size,
        max_output_size,

        active_size,
        max_neuron_size,


        batch_size,
        traj_size,

        proc_num,

        &dir_path
    );
    */

    let evolution_info: evolution_info_type = evolution_info_type {
        dir_path: dir_path.clone().to_string(),
        cur_path: "".to_string(),

        search_strategy: evolution_search_type::METROPOLIS_SEARCH,
        success_idx: 0,
        total_tries: 0,

        crossval_vec: Vec::new(),
        netdata_vec: Vec::new(),

        traj_size: traj_size,

        max_input_size: max_input_size,
        max_output_size: max_output_size,

        max_active_size: max_neuron_size,
        min_active_size: active_size,

        max_proc_num: 20,
        min_proc_num: 2,

        max_proc_num_step: 1,
        min_proc_num_step: 0,

        max_active_size_step: 0.3,
        min_active_size_step: 0.05,

        max_prune_num: 0.07,
        min_prune_num: 0.01,

        max_search_num: 1,
    };

    let old_input_size = 1400;
    let model_path = format!("{}/network_{}_{}.csv", output_dir, fold, old_input_size);
    let neural_network = load_network2(&model_path);

    let mut arch_search: arch_search_type = arch_search_type {
        neural_network: neural_network,

        evolution_info: evolution_info,
    };

    arch_search.neural_network.netdata.batch_size = batch_size;

    //Load dataset
    println!("Load Dataset");

    let mut TOTAL_EEG_TRAINX: nohash_hasher::IntMap<u64, Vec<f32>> =
        nohash_hasher::IntMap::default();
    let mut EEG_TRAINY: nohash_hasher::IntMap<u64, Vec<f32>> = nohash_hasher::IntMap::default();

    let mut EEG_VALX: nohash_hasher::IntMap<u64, Vec<f32>> = nohash_hasher::IntMap::default();
    let mut EEG_VALY: nohash_hasher::IntMap<u64, Vec<f32>> = nohash_hasher::IntMap::default();

    let dataset_path = format!("{}/Y_train_{}.txt", dataset_dir, fold);
    let mut strvec = raybnn::export::dataloader_f32::file_to_vec_cpu(&dataset_path);
    let num = strvec.len() as u64;
    let mut train_Yarr = arrayfire::Array::new(
        &strvec,
        arrayfire::Dim4::new(&[output_size, num / output_size, 1, 1]),
    );

    let dataset_path = format!("{}/X_train_{}.txt", dataset_dir, fold);
    let mut strvec = raybnn::export::dataloader_f32::file_to_vec_cpu(&dataset_path);
    let num = strvec.len() as u64;
    let mut train_Xarr = arrayfire::Array::new(
        &strvec,
        arrayfire::Dim4::new(&[input_size, num / input_size, 1, 1]),
    );

    let dataset_path = format!("{}/Y_val_{}.txt", dataset_dir, fold);
    let mut strvec = raybnn::export::dataloader_f32::file_to_vec_cpu(&dataset_path);
    let num = strvec.len() as u64;
    let mut validate_Yarr = arrayfire::Array::new(
        &strvec,
        arrayfire::Dim4::new(&[output_size, num / output_size, 1, 1]),
    );

    let dataset_path = format!("{}/X_val_{}.txt", dataset_dir, fold);
    let mut strvec = raybnn::export::dataloader_f32::file_to_vec_cpu(&dataset_path);
    let num = strvec.len() as u64;
    let mut validate_Xarr = arrayfire::Array::new(
        &strvec,
        arrayfire::Dim4::new(&[input_size, num / input_size, 1, 1]),
    );

    let mut traindata_idx = 0;
    let mut valdata_idx = 0;

    println!("train_Yarr.dims() {:?}", train_Yarr.dims());
    println!("train_Xarr.dims() {:?}", train_Xarr.dims());

    println!("validate_Yarr.dims() {:?}", validate_Yarr.dims());
    println!("validate_Xarr.dims() {:?}", validate_Xarr.dims());

    let mut iterations = train_Yarr.dims()[1] / batch_size;

    let mut start_idx = 0;
    let mut end_idx = 0;

    for qq in 0..iterations {
        end_idx = start_idx + batch_size - 1;

        let tmpY = arrayfire::cols(&train_Yarr, start_idx as i64, end_idx as i64);
        let tmpX = arrayfire::cols(&train_Xarr, start_idx as i64, end_idx as i64);

        let mut strvec = vec![f32::default(); tmpY.elements()];
        tmpY.host(&mut strvec);
        EEG_TRAINY.insert(traindata_idx, strvec.clone());

        let mut strvec = vec![f32::default(); tmpX.elements()];
        tmpX.host(&mut strvec);
        TOTAL_EEG_TRAINX.insert(traindata_idx, strvec.clone());

        traindata_idx = traindata_idx + 1;

        start_idx = end_idx + 1;
    }

    let mut iterations = validate_Yarr.dims()[1] / val_batch_size;

    let mut start_idx = 0;
    let mut end_idx = 0;

    for qq in 0..iterations {
        end_idx = start_idx + val_batch_size - 1;

        let tmpY = arrayfire::cols(&validate_Yarr, start_idx as i64, end_idx as i64);
        let tmpX = arrayfire::cols(&validate_Xarr, start_idx as i64, end_idx as i64);

        let mut strvec = vec![f32::default(); tmpY.elements()];
        tmpY.host(&mut strvec);
        EEG_VALY.insert(valdata_idx, strvec.clone());

        let mut strvec = vec![f32::default(); tmpX.elements()];
        tmpX.host(&mut strvec);
        EEG_VALX.insert(valdata_idx, strvec.clone());

        valdata_idx = valdata_idx + 1;

        start_idx = end_idx + 1;
    }

    println!(" EEG_TRAINX {}", TOTAL_EEG_TRAINX.len());
    println!(" EEG_VALX {}", EEG_VALX.len());

    let mut EEG_TRAINX = TOTAL_EEG_TRAINX.clone();

    arrayfire::sync(DEVICE);

    //Transfer Learning Training Loop

    let mut prev_target_neuron_size = 0;

    //for ii in 0..arg_input_vec.len()
    //{
    //EEG_TRAINX = TOTAL_EEG_TRAINX.clone();

    //EEG_TESTX = TOTAL_EEG_TESTX.clone();

    arch_search.neural_network.neuron_pos = arrayfire::rows(
        &arch_search.neural_network.neuron_pos,
        old_input_size as i64,
        (arch_search.neural_network.neuron_pos.dims()[0] - 1) as i64,
    );

    let mut WRowIdxCOO =
        raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

    arch_search.neural_network.netdata.neuron_std = 0.001;
    arch_search.neural_network.netdata.con_rad =
        (arch_search.neural_network.netdata.sphere_rad / (proc_num as f32)) * 2.0;
    arch_search.neural_network.netdata.input_size = input_size;

    //Initialize Input Neurons

    let input_neurons = raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere_1D(
        arch_search.neural_network.netdata.sphere_rad + 0.2,
        input_size,
    );

    arch_search.neural_network.neuron_pos =
        arrayfire::join(0, &input_neurons, &arch_search.neural_network.neuron_pos);

    arch_search.neural_network.netdata.active_size =
        arch_search.neural_network.neuron_pos.dims()[0];

    raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &arch_search.neural_network.netdata,
        &arch_search.neural_network.neuron_pos,
        &mut arch_search.neural_network.neuron_idx,
    );

    let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
        new_active_size: 10,
        init_connection_num: 1,
        input_neuron_con_rad: 127.0 * arch_search.neural_network.netdata.neuron_rad,
        hidden_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
        output_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
    };

    raybnn::physics::update_f32::add_neuron_to_existing3(&add_neuron_options, &mut arch_search);

    raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &arch_search.neural_network.netdata,
        &arch_search.neural_network.neuron_pos,
        &mut arch_search.neural_network.neuron_idx,
    );

    println!("params {}", arch_search.neural_network.WColIdx.dims()[0]);

    let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
    arch_search.neural_network.netdata.active_size = active_size;

    WRowIdxCOO =
        raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

    arch_search.neural_network.WRowIdxCSR = raybnn::graph::large_sparse_i32::COO_to_CSR(
        &WRowIdxCOO,
        arch_search.neural_network.netdata.neuron_size,
    );

    arrayfire::device_gc();
    arrayfire::sync(DEVICE);

    let traj_steps = traj_size + proc_num - 1;

    let Z_dims = arrayfire::Dim4::new(&[
        arch_search.neural_network.netdata.neuron_size,
        batch_size,
        traj_steps,
        1,
    ]);
    let mut Z = arrayfire::constant::<f32>(0.0, Z_dims);
    let mut Q = arrayfire::constant::<f32>(0.0, Z_dims);

    let mut alpha0: f32 = 0.002;

    let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(
        &arch_search.neural_network.neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let Qslices: u64 = Q.dims()[2];

    let mut WRowIdxCOO =
        raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

    let total_param_size = arch_search.neural_network.network_params.dims()[0];

    let mt_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);
    let mut mt = arrayfire::constant::<f32>(0.0, mt_dims);
    let mut vt = arrayfire::constant::<f32>(0.0, mt_dims);
    let mut grad = arrayfire::constant::<f32>(0.0, mt_dims);

    let mut Wseqs = [arrayfire::Seq::default()];
    let mut Hseqs = [arrayfire::Seq::default()];
    let mut Aseqs = [arrayfire::Seq::default()];
    let mut Bseqs = [arrayfire::Seq::default()];
    let mut Cseqs = [arrayfire::Seq::default()];
    let mut Dseqs = [arrayfire::Seq::default()];
    let mut Eseqs = [arrayfire::Seq::default()];

    let X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_steps, 1]);
    let mut X = arrayfire::constant::<f32>(0.0, X_dims);

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

    let WValuesdims0 = arch_search.neural_network.WColIdx.dims()[0];

    let network_paramsdims0 = arch_search.neural_network.network_params.dims()[0];

    let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

    //Create backward graph

    raybnn::graph::path_f32::find_path_backward_group2(
        &arch_search.neural_network.netdata,
        traj_steps,
        traj_size,
        &WRowIdxCOO,
        &arch_search.neural_network.WColIdx,
        &arch_search.neural_network.neuron_idx,
        WValuesdims0,
        Hdims0,
        Hdims0,
        Hdims0,
        Hdims0,
        Hdims0,
        Hdims0,
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

    let train_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
    let mut train_X = arrayfire::constant::<f32>(0.0, temp_dims);
    let Y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
    let mut Y = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut batch_idx = 0;
    let epoch_num = (EEG_TRAINY.len()) as u64;

    let mut max_epoch = 10000;

    let mut loss_counter = 0;

    let mut loss_val = 100000.0;
    let mut prev_loss = 100000.0;

    arrayfire::device_gc();
    arrayfire::sync(DEVICE);

    let start_time = std::time::Instant::now();

    //Training Loop

    let mask0_dims = arrayfire::Dim4::new(&[WValuesdims0, 1, 1, 1]);
    let mut mask0 = arrayfire::constant::<f32>(1.0, mask0_dims);

    let mask1_dims = arrayfire::Dim4::new(&[network_paramsdims0 - WValuesdims0, 1, 1, 1]);
    let mut mask1 = arrayfire::constant::<f32>(0.0, mask1_dims);

    mask0 = arrayfire::join(0, &mask0, &mask1);

    let mut sel_mask = false;

    let mut shuffle_counter = 0;
    let mut CV_Loss = Vec::new();
    for i in 0..max_epoch {
        batch_idx = i % epoch_num;

        train_X = arrayfire::Array::new(&EEG_TRAINX[&batch_idx], train_X_dims);

        Y = arrayfire::Array::new(&EEG_TRAINY[&batch_idx], Y_dims);
        arrayfire::set_slices(&mut X, &train_X, 0, (traj_size - 1) as i64);

        //Forward pass

        raybnn::neural::network_f32::state_space_forward_batch(
            &arch_search.neural_network.netdata,
            &X,
            &arch_search.neural_network.WRowIdxCSR,
            &arch_search.neural_network.WColIdx,
            &Wseqs,
            &Hseqs,
            &Aseqs,
            &Bseqs,
            &Cseqs,
            &Dseqs,
            &Eseqs,
            &arch_search.neural_network.network_params,
            &mut Z,
            &mut Q,
        );

        let mut eval_metric_out = Vec::new();
        let mut Yhat_out = nohash_hasher::IntMap::default();

        arch_search.neural_network.netdata.batch_size = val_batch_size;
        raybnn::interface::autotest_f32::validate_network(
            &EEG_VALX,
            &EEG_VALY,
            MSE,
            &mut arch_search,
            &mut Yhat_out,
            &mut eval_metric_out,
        );
        arch_search.neural_network.netdata.batch_size = batch_size;
        loss_val = eval_metric_out.par_iter().sum::<f32>() / val_size;
        CV_Loss.push(loss_val);

        println!("loss_val {}, i {}", loss_val, i);

        //Backward pass

        raybnn::neural::network_f32::state_space_backward_group2(
            &arch_search.neural_network.netdata,
            &X,
            &arch_search.neural_network.network_params,
            &Z,
            &Q,
            &Y,
            MSE_grad,
            &arch_search.neural_network.neuron_idx,
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

        //Update weights with ADAM

        raybnn::optimal::gd_f32::adam(0.9, 0.999, &mut grad, &mut mt, &mut vt);

        if sel_mask {
            grad = grad.clone() * mask0.clone();
        }

        arch_search.neural_network.network_params =
            arch_search.neural_network.network_params + (alpha0 * -grad.clone());

        if loss_val > prev_loss {
            loss_counter = loss_counter + 1;
        } else {
            loss_counter = 0;
        }

        if i >= 10000 {
            println!("min_idx {}", min_idx);
            break;
        }

        if i == 4400 {
            alpha0 = 0.0002;
            loss_counter = 0;
            sel_mask = true;
        }

        if (loss_counter > 3) && (i < 4400) {
            println!("shuffle");

            loss_counter = 0;

            raybnn::interface::autotransfer_f32::shuffle_weights(i, &mut arch_search);

            (arch_search).neural_network.netdata.active_size =
                (arch_search).neural_network.neuron_idx.dims()[0];
            active_size = (arch_search).neural_network.netdata.active_size;
            WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(
                &((arch_search).neural_network.WRowIdxCSR),
            );

            let total_param_size = (arch_search).neural_network.network_params.dims()[0];
            let mt_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);
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

            let WValuesdims0 = arch_search.neural_network.WColIdx.dims()[0];

            let network_paramsdims0 = arch_search.neural_network.network_params.dims()[0];

            let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

            //Create backward graph

            raybnn::graph::path_f32::find_path_backward_group2(
                &arch_search.neural_network.netdata,
                traj_steps,
                traj_size,
                &WRowIdxCOO,
                &arch_search.neural_network.WColIdx,
                &arch_search.neural_network.neuron_idx,
                WValuesdims0,
                Hdims0,
                Hdims0,
                Hdims0,
                Hdims0,
                Hdims0,
                Hdims0,
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

            WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(
                &((arch_search).neural_network.WRowIdxCSR),
            );
        }
        prev_loss = loss_val;
    }

    arrayfire::sync(DEVICE);
    println!("loss_val {}", loss_val);

    let elapsed: f32 = start_time.elapsed().as_secs_f32();
    println!("elapsed {}", elapsed);

    //traj_size = 20;

    arrayfire::device_gc();

    //Change Input Dimensions of Testing dataset
    /*
    if (input_size != max_input_size)
    {
        let mut tempRSSI = TOTAL_EEG_TESTX.clone();

        for (key, value) in &TOTAL_EEG_TESTX {
            let mut tempvec = value.clone();

            let veclen =  tempvec.len() as u64;
            let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[max_input_size, veclen/max_input_size, 1, 1]));


            formatarr = arrayfire::rows(&formatarr, 0, (input_size-1) as i64);

            tempvec = vec!(f32::default();formatarr.elements());
            formatarr.host(&mut tempvec);

            tempRSSI.insert(key.clone(), tempvec);
        }

        EEG_TESTX = tempRSSI;
    }
    */
    arrayfire::sync(DEVICE);

    batch_size = 200;
    arch_search.neural_network.netdata.batch_size = batch_size;

    let mut TOTAL_EEG_TESTX: nohash_hasher::IntMap<u64, Vec<f32>> =
        nohash_hasher::IntMap::default();
    let mut EEG_TESTY: nohash_hasher::IntMap<u64, Vec<f32>> = nohash_hasher::IntMap::default();

    let dataset_path = format!("{}/Y_test_{}.txt", dataset_dir, fold);
    let mut strvec = raybnn::export::dataloader_f32::file_to_vec_cpu(&dataset_path);
    let num = strvec.len() as u64;
    let mut test_Yarr = arrayfire::Array::new(
        &strvec,
        arrayfire::Dim4::new(&[output_size, num / output_size, 1, 1]),
    );

    let dataset_path = format!("{}/X_test_{}.txt", dataset_dir, fold);
    let mut strvec = raybnn::export::dataloader_f32::file_to_vec_cpu(&dataset_path);
    let num = strvec.len() as u64;
    let mut test_Xarr = arrayfire::Array::new(
        &strvec,
        arrayfire::Dim4::new(&[input_size, num / input_size, 1, 1]),
    );

    println!("test_Yarr.dims() {:?}", test_Yarr.dims());
    println!("test_Xarr.dims() {:?}", test_Xarr.dims());

    let mut testdata_idx = 0;
    let mut iterations = test_Yarr.dims()[1] / batch_size;

    let mut start_idx = 0;
    let mut end_idx = 0;

    for qq in 0..iterations {
        end_idx = start_idx + batch_size - 1;

        let tmpY = arrayfire::cols(&test_Yarr, start_idx as i64, end_idx as i64);
        let tmpX = arrayfire::cols(&test_Xarr, start_idx as i64, end_idx as i64);

        let mut strvec = vec![f32::default(); tmpY.elements()];
        tmpY.host(&mut strvec);
        EEG_TESTY.insert(testdata_idx, strvec.clone());

        let mut strvec = vec![f32::default(); tmpX.elements()];
        tmpX.host(&mut strvec);
        TOTAL_EEG_TESTX.insert(testdata_idx, strvec.clone());

        testdata_idx = testdata_idx + 1;

        start_idx = end_idx + 1;
    }

    arrayfire::sync(DEVICE);

    let mut EEG_TESTX = TOTAL_EEG_TESTX.clone();

    let traj_steps = traj_size + proc_num - 1;

    let Z_dims = arrayfire::Dim4::new(&[
        arch_search.neural_network.netdata.neuron_size,
        batch_size,
        traj_steps,
        1,
    ]);
    let mut Z = arrayfire::constant::<f32>(0.0, Z_dims);
    let mut Q = arrayfire::constant::<f32>(0.0, Z_dims);

    let Qslices: u64 = Q.dims()[2];

    let X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_steps, 1]);
    let mut X = arrayfire::constant::<f32>(0.0, X_dims);

    let test_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
    let mut test_X = arrayfire::constant::<f32>(0.0, temp_dims);
    let Y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
    let mut Y = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut batch_idx = 0;
    let epoch_num = (EEG_TESTY.len()) as u64;

    let temp_Y_dims = arrayfire::Dim4::new(&[1, output_size, 1, 1]);

    let mut Yhat_arr = arrayfire::constant::<f32>(0.0, temp_Y_dims);
    let mut Y_arr = arrayfire::constant::<f32>(0.0, temp_Y_dims);

    let newdims = arrayfire::Dim4::new(&[output_size, batch_size * traj_size, 1, 1]);

    //Compute testing dataset

    arrayfire::device_gc();
    arrayfire::sync(DEVICE);
    for batch_idx in 0..epoch_num {
        test_X = arrayfire::Array::new(&EEG_TESTX[&batch_idx], test_X_dims);

        Y = arrayfire::Array::new(&EEG_TESTY[&batch_idx], Y_dims);
        arrayfire::set_slices(&mut X, &test_X, 0, (traj_size - 1) as i64);

        raybnn::neural::network_f32::state_space_forward_batch(
            &arch_search.neural_network.netdata,
            &X,
            &arch_search.neural_network.WRowIdxCSR,
            &arch_search.neural_network.WColIdx,
            &Wseqs,
            &Hseqs,
            &Aseqs,
            &Bseqs,
            &Cseqs,
            &Dseqs,
            &Eseqs,
            &arch_search.neural_network.network_params,
            &mut Z,
            &mut Q,
        );

        //Get Yhat
        let mut idxrs = arrayfire::Indexer::default();
        let seq1 = arrayfire::Seq::new(0.0f32, (batch_size - 1) as f32, 1.0);
        let seq2 = arrayfire::Seq::new((proc_num - 1) as f32, (Qslices - 1) as f32, 1.0);
        idxrs.set_index(&idxsel, 0, None);
        idxrs.set_index(&seq1, 1, None);
        idxrs.set_index(&seq2, 2, None);
        let mut Yhat = arrayfire::index_gen(&Q, idxrs);

        Yhat = arrayfire::flat(&Yhat);
        Y = arrayfire::flat(&Y);

        Yhat = arrayfire::moddims(&Yhat, newdims);
        Y = arrayfire::moddims(&Y, newdims);

        Yhat = arrayfire::transpose(&Yhat, false);
        Y = arrayfire::transpose(&Y, false);

        Yhat_arr = arrayfire::join(0, &Yhat_arr, &Yhat);
        Y_arr = arrayfire::join(0, &Y_arr, &Y);
    }

    Yhat_arr = arrayfire::rows(&Yhat_arr, 1, (Yhat_arr.dims()[0] - 1) as i64);
    Y_arr = arrayfire::rows(&Y_arr, 1, (Y_arr.dims()[0] - 1) as i64);

    arrayfire::device_gc();
    arrayfire::sync(DEVICE);

    let totalparam = arch_search.neural_network.network_params.dims()[0] as f32;

    let v0: [f32; 2] = [elapsed, totalparam];
    let info = arrayfire::Array::new(&v0, arrayfire::Dim4::new(&[2, 1, 1, 1]));

    //Save dataset

    //input_size = 1400;
    let save_filename = format!("{}/info_{}_{}.csv", output_dir, fold, input_size);
    raybnn::export::dataloader_f32::write_arr_to_csv(&save_filename, &info);

    let save_filename = format!("{}/test_pred_{}_{}.csv", output_dir, fold, input_size);
    raybnn::export::dataloader_f32::write_arr_to_csv(&save_filename, &Yhat_arr);

    let save_filename = format!("{}/test_act_{}_{}.csv", output_dir, fold, input_size);
    raybnn::export::dataloader_f32::write_arr_to_csv(&save_filename, &Y_arr);

    if CVLoss_exists == false {
        let save_filename = format!("{}/CVLOSS_{}_{}.csv", output_dir, fold, input_size);
        raybnn::export::dataloader_f32::write_vec_cpu_to_csv(&save_filename, &CV_Loss);
    }
}
