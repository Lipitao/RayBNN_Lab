extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

use crate::physics::initial_f64::sphere_cell_collision_minibatch;

use crate::neural::network_f64::network_metadata_type;
use crate::neural::network_f64::neural_network_type;
use crate::neural::network_f64::UAF_initial_as_identity;

use crate::optimal::evolution_f64::evolution_info_type;
use crate::physics::raytrace_f64::raytrace_option_type;

use crate::optimal::evolution_f64::evolution_search_type;

use crate::optimal::evolution_f64::evolve_network;

use crate::graph::adjacency_f64::select_forward_sphere;

use crate::physics::initial_f64::cube_struct;

use crate::physics::construct_f64::get_outside_idx_cube;

use crate::physics::dynamic_f64::run;

use crate::physics::update_f64::reposition;

use crate::physics::update_f64::create_objects;

use crate::physics::construct_f64::get_inside_idx_cube;

use crate::export::dataloader_f64::save_network;

use crate::export::dataloader_u64::find_model_paths;
use crate::physics::raytrace_f64::RT3_distance_limited_directly_connected;

use crate::physics::construct_f64::NDsphere_from_NDcube;

use crate::physics::distance_f64::sort_neuron_pos_sphere;

use crate::physics::construct_f64::plane_surface_on_NDsphere;

use crate::physics::initial_f64::assign_neuron_idx_with_buffer;

use crate::physics::initial_f64::input_and_output_layers;

use crate::physics::initial_f64::hidden_layers;

use crate::physics::initial_f64::fully_connected_hidden_layers;

use crate::physics::initial_f64::self_loops;

use crate::physics::initial_f64::assign_self_loop_value;

use crate::neural::network_f64::UAF_initial_as_tanh;

use crate::graph::adjacency_f64::clear_input;

use crate::graph::adjacency_f64::clear_output;

use crate::graph::adjacency_f64::delete_unused_neurons;

use crate::graph::large_sparse_i32::COO_to_CSR;

use crate::graph::large_sparse_i32::CSR_to_COO;

use crate::graph::tree_i32::check_connected2;

use crate::graph::path_f64::find_path_backward_group2;

use crate::neural::network_f64::xavier_init;

use crate::neural::network_f64::state_space_backward_group2;

use crate::optimal::gd_f64::adam;

use crate::optimal::control_f64::statespace_BTLS;

use crate::neural::network_f64::clone_neural_network;

use crate::export::dataloader_u64::extract_file_info;

use crate::export::dataloader_f64::load_network2;

use crate::export::dataloader_f64::save_network2;

use crate::export::dataloader_f64::hash_batch_to_files;
use crate::export::dataloader_f64::write_vec_cpu_to_csv;

use crate::interface::autotrain_f64::loss_wrapper;

use crate::interface::autotrain_f64::train_network;

use crate::interface::autotest_f64::validate_network;

use crate::interface::autotrain_f64::stop_strategy_type;

use crate::interface::autotrain_f64::lr_strategy_type;

use crate::interface::autotrain_f64::lr_strategy2_type;

use crate::interface::autotrain_f64::loss_status_type;

use crate::interface::autotrain_f64::train_network_options_type;

use serde::{Deserialize, Serialize};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Serialize, Deserialize)]
pub struct arch_search_type {
    pub neural_network: neural_network_type,

    pub evolution_info: evolution_info_type,
}

#[derive(Hash, Serialize, Deserialize)]
pub struct network_info_seed_type {
    pub i: u64,
    pub proc_num: u64,
    pub connection_num: u64,
    pub active_size: u64,
}

const LR_MAX: f64 = 1.0;
const LR_BUFFER: usize = 20;
const LARGE_POS_NUM_f64: f64 = 1.0e9;
const LARGE_POS_NUM_u64: u64 = 1000000000;
const SPHERE_RAD_MIN: f64 = 0.4;
const INOUT_FACTOR: f64 = 0.15;
const PACKING_FACTOR: f64 = 0.02;
const SPHERE_DENSITY: f64 = 0.28;

pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

pub fn set_network_seed(network_info: &network_info_seed_type) {
    let seed = calculate_hash(network_info);

    arrayfire::set_seed(seed);
}

pub fn set_network_seed2(crossval_vec: &Vec<f64>) {
    let strarr = format!("{:?}", crossval_vec.clone());

    let rand_seed = calculate_hash(&strarr);
    arrayfire::set_seed(rand_seed);
}

pub fn create_start_archtecture(
    input_size: u64,
    max_input_size: u64,

    output_size: u64,
    max_output_size: u64,

    max_neuron_size: u64,

    batch_size: u64,
    traj_size: u64,

    dir_path: &str,
) -> arch_search_type {
    let proc_num: u64 = 2;
    let mut active_size: u64 = (1.2 * ((input_size + output_size) as f64)) as u64;
    let space_dims: u64 = 3;
    let sim_steps: u64 = 5000;

    if active_size < 20 {
        active_size = 20;
    }

    let mut neuron_rad: f64 = 0.1;

    //let mut sphere_radius: f64 = neuron_rad*(((active_size as f64)/PACKING_FACTOR).cbrt());

    let mut sphere_radius = (((2 * active_size) as f64) / SPHERE_DENSITY);
    sphere_radius = (sphere_radius / ((4.0 / 3.0) * std::f64::consts::PI)).cbrt();

    if sphere_radius < SPHERE_RAD_MIN {
        sphere_radius = SPHERE_RAD_MIN;
    }

    let mut netdata: network_metadata_type = network_metadata_type {
        neuron_size: max_neuron_size,
        input_size: input_size,
        output_size: output_size,
        proc_num: proc_num,
        active_size: active_size,
        space_dims: space_dims,
        step_num: sim_steps,
        batch_size: batch_size,
        del_unused_neuron: true,

        time_step: 0.1,
        nratio: 0.5,
        neuron_std: 0.001,
        sphere_rad: sphere_radius,
        neuron_rad: neuron_rad,
        con_rad: (sphere_radius / (proc_num as f64)) * 1.4f64,
        init_prob: 0.01,
        add_neuron_rate: 0.0,
        del_neuron_rate: 0.0,
        center_const: 0.005,
        spring_const: 0.01,
        repel_const: 0.01,
    };

    let temp_dims = arrayfire::Dim4::new(&[4, 1, 1, 1]);

    let mut glia_pos = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut neuron_pos = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut H = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut A = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut B = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut C = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut D = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut E = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut neuron_idx = arrayfire::constant::<i32>(0, temp_dims);

    let mut WValues = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut WRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
    let mut WRowIdxCSR = arrayfire::constant::<i32>(0, temp_dims);
    let mut WColIdx = arrayfire::constant::<i32>(0, temp_dims);

    sphere_cell_collision_minibatch(&netdata, &mut glia_pos, &mut neuron_pos);
    active_size = neuron_pos.dims()[0];
    netdata.active_size = active_size;

    let savefilename = format!(
        "{}/{}D_sphere_{:.5}_rad.csv",
        dir_path, space_dims, sphere_radius
    );

    save_network(
        &savefilename,
        &netdata,
        &WValues,
        &WRowIdxCSR,
        &WColIdx,
        &H,
        &A,
        &B,
        &C,
        &D,
        &E,
        &glia_pos,
        &neuron_pos,
        &neuron_idx,
    );

    /*
    neuron_pos = NDsphere_from_NDcube(
        &neuron_pos,
        sphere_radius,

        sphere_radius
    );



    glia_pos = NDsphere_from_NDcube(
        &glia_pos,
        sphere_radius,

        sphere_radius
    );
    */

    sort_neuron_pos_sphere(&mut neuron_pos);

    let sqrt_input = (input_size as f64).sqrt().ceil() as u64;
    let plane_shape = vec![sqrt_input, sqrt_input];

    let mut input_neurons =
        plane_surface_on_NDsphere(&plane_shape, sphere_radius + (neuron_rad * 2.0));

    if input_neurons.dims()[0] > input_size {
        input_neurons = arrayfire::rows(&input_neurons, 0, (input_size - 1) as i64);
    }

    neuron_pos = arrayfire::join(0, &input_neurons, &neuron_pos);

    let output_neurons =
        arrayfire::constant::<f64>(0.0, arrayfire::Dim4::new(&[output_size, space_dims, 1, 1]));

    neuron_pos = arrayfire::join(0, &neuron_pos, &output_neurons);

    netdata.active_size = neuron_pos.dims()[0];

    active_size = neuron_pos.dims()[0];

    assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &netdata,
        &neuron_pos,
        &mut neuron_idx,
    );

    /*
    let hidden_num2 = active_size - output_size - input_size;
    let mut init_connection_num = 10;

    if init_connection_num >= hidden_num2
    {
        init_connection_num = hidden_num2;
    }
    else
    {
        let temp_num = ((hidden_num2 as f64)*INOUT_FACTOR) as u64 ;
        if temp_num >  init_connection_num
        {
            init_connection_num = temp_num;
        }

    }


    input_and_output_layers(
        &netdata,
        &neuron_pos,
        &neuron_idx,

        init_connection_num,
        init_connection_num,

        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx
    );






    fully_connected_hidden_layers(
        &neuron_pos,
        &neuron_idx,

        &mut netdata,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx
    );



    let in_idx = arrayfire::rows(&neuron_idx, 0, (input_size-1)  as i64);
    */

    let mut hidden_num2 = neuron_pos.dims()[0] - output_size - input_size;
    let mut init_connection_num = 10;

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * input_size,
        ray_neuron_intersect: false,
        ray_glia_intersect: false,
    };

    netdata.con_rad = 60.0 * netdata.neuron_rad;

    let neuron_num = neuron_pos.dims()[0];
    let hidden_idx_total = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = arrayfire::rows(&neuron_idx, 0, (input_size - 1) as i64);
    let input_pos = arrayfire::rows(&neuron_pos, 0, (input_size - 1) as i64);

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &netdata,
        &glia_pos,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    let neuron_num = neuron_pos.dims()[0];

    let hidden_idx_total = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = hidden_idx_total.clone();
    let input_pos = hidden_pos_total.clone();

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * hidden_num2,
        ray_neuron_intersect: true,
        ray_glia_intersect: true,
    };

    netdata.con_rad = 15.0 * netdata.neuron_rad;

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &netdata,
        &glia_pos,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    let neuron_num = neuron_pos.dims()[0];

    let input_idx = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let input_pos = arrayfire::rows(
        &neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let hidden_idx_total = arrayfire::rows(
        &neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &neuron_pos,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    netdata.con_rad = 60.0 * netdata.neuron_rad;

    let glia_pos_temp_dims = arrayfire::Dim4::new(&[4, space_dims, 1, 1]);

    let glia_pos_temp = arrayfire::constant::<f64>(-1000.0, glia_pos_temp_dims);

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * output_size,
        ray_neuron_intersect: false,
        ray_glia_intersect: false,
    };

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &netdata,
        &glia_pos_temp,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    //Populate WValues with random values
    let rand_dims = arrayfire::Dim4::new(&[WColIdx.dims()[0], 1, 1, 1]);
    WValues = netdata.neuron_std * arrayfire::randn::<f64>(rand_dims);

    //Clear input/output
    clear_input(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        netdata.input_size,
    );

    clear_output(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        netdata.neuron_size - netdata.output_size,
    );

    let in_idx = arrayfire::rows(&neuron_idx, 0, (input_size - 1) as i64);

    /*
    if traj_size > 1
    {
        self_loops(
            &netdata,
            //&neuron_pos,
            &neuron_idx,


            &mut WValues,
            &mut WRowIdxCOO,
            &mut WColIdx
        );
    }
    else
    {
        select_forward_sphere(
            &netdata,
            &mut WValues,
            &mut WRowIdxCOO,
            &mut WColIdx,
            &neuron_pos,
            &neuron_idx
        );
    }
    */

    self_loops(
        &netdata,
        //&neuron_pos,
        &neuron_idx,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    UAF_initial_as_identity(
        &netdata, //&mut H,
        &mut A, &mut B, &mut C, &mut D, &mut E,
    );

    delete_unused_neurons(
        &netdata,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        &mut glia_pos,
        &mut neuron_pos,
        &mut neuron_idx,
    );

    xavier_init(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        max_neuron_size,
        proc_num,
        &mut WValues,
        &mut H,
    );

    WRowIdxCSR = COO_to_CSR(&WRowIdxCOO, max_neuron_size);

    let mut active_size = neuron_idx.dims()[0];
    netdata.active_size = active_size;
    let con_num = WValues.dims()[0];
    let filename = format!(
        "{}/active_size_{}_proc_num_{}_con_num_{}.csv",
        dir_path, active_size, proc_num, con_num
    );

    save_network(
        &filename,
        &netdata,
        &WValues,
        &WRowIdxCSR,
        &WColIdx,
        &H,
        &A,
        &B,
        &C,
        &D,
        &E,
        &glia_pos,
        &neuron_pos,
        &neuron_idx,
    );

    let total_param_size = WValues.dims()[0]
        + H.dims()[0]
        + A.dims()[0]
        + B.dims()[0]
        + C.dims()[0]
        + D.dims()[0]
        + E.dims()[0];
    let network_params_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);

    let Wstart = 0;
    let Wend = (WValues.dims()[0] as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (H.dims()[0] as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (A.dims()[0] as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (B.dims()[0] as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (C.dims()[0] as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (D.dims()[0] as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (E.dims()[0] as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let mut network_params = arrayfire::constant::<f64>(0.0, network_params_dims);
    arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
    arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
    arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
    arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
    arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
    arrayfire::assign_seq(&mut network_params, &Dseqs, &D);
    arrayfire::assign_seq(&mut network_params, &Eseqs, &E);

    let mut neural_network: neural_network_type = neural_network_type {
        netdata: netdata,
        WRowIdxCSR: WRowIdxCSR,
        WColIdx: WColIdx,
        network_params: network_params,
        glia_pos: glia_pos,
        neuron_pos: neuron_pos,
        neuron_idx: neuron_idx,
    };

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

    let arch_search: arch_search_type = arch_search_type {
        neural_network: neural_network,

        evolution_info: evolution_info,
    };

    arch_search
}

// Search optimal neural network archtecture to minimize function eval_metric

//           Input arguments
// traindata_X: Training input array to the neural network
// 				traindata_X.dims()[0]  input feature size
// 				traindata_X.dims()[1]  batch size
// 				traindata_X.dims()[2]  sequence/traj size
// traindata_Y: Training ground truth output array
// 				traindata_Y.dims()[0]  output feature size
// 				traindata_Y.dims()[1]  batch size
// 				traindata_Y.dims()[2]  sequence/traj size
// crossvaldata_X: Cross validation input array to the neural network
// crossvaldata_Y: Cross validation ground truth output array
// eval_metric: Function with inputs X and Yhat that gives evaluation metrics
// eval_metric_grad: Gradient of eval_metric
// max_iter: Maximum number of searches

//          Output argumenets
// arch_search:  Entire neural network archtecture
pub fn architecture_search(
    traindata_X: &nohash_hasher::IntMap<u64, Vec<f64>>,
    traindata_Y: &nohash_hasher::IntMap<u64, Vec<f64>>,

    crossvaldata_X: &nohash_hasher::IntMap<u64, Vec<f64>>,
    crossvaldata_Y: &nohash_hasher::IntMap<u64, Vec<f64>>,

    testdata_X: &nohash_hasher::IntMap<u64, Vec<f64>>,
    testdata_Y: &nohash_hasher::IntMap<u64, Vec<f64>>,

    eval_metric: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> f64 + Copy,
    eval_metric_grad: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> arrayfire::Array<f64>
        + Copy,
    max_search_num: u64,

    arch_search: &mut arch_search_type,
) {
    (*arch_search).evolution_info.max_search_num = max_search_num;

    (*arch_search).evolution_info.success_idx = 0;

    let model_path_vec = find_model_paths(&arch_search.evolution_info.dir_path);

    let mut min_info: Vec<u64> = vec![LARGE_POS_NUM_u64, LARGE_POS_NUM_u64, LARGE_POS_NUM_u64];
    let mut min_path: String = model_path_vec[0].clone();

    //Find smallest model
    for path in model_path_vec {
        let info_vec = extract_file_info(&path);

        if info_vec.len() == 3 {
            if (info_vec[0] <= min_info[0])
                && (info_vec[1] <= min_info[1])
                && (info_vec[2] <= min_info[2])
            {
                min_info = info_vec.clone();
                min_path = path.clone();
            }
        }
    }

    //Set smallest model
    (*arch_search).evolution_info.cur_path = min_path.clone();

    let epoch_num = traindata_X.len() as u64;

    let mut alpha_max_vec = Vec::new();

    let mut loss_status = loss_status_type::LOSS_OVERFLOW;

    (*arch_search).neural_network = load_network2(&(*arch_search).evolution_info.cur_path);

    let mut loss_vec: Vec<f64> = Vec::new();

    let mut crossval_vec: Vec<f64> = Vec::new();

    let mut stuck_idx: u64 = 0;
    let mut stuck_counter: u64 = 0;

    while 1 == 1 {
        loss_status = loss_status_type::LOSS_OVERFLOW;

        //Load network
        (*arch_search).neural_network = load_network2(&(*arch_search).evolution_info.cur_path);

        //Train options
        let train_normal_options: train_network_options_type = train_network_options_type {
            stop_strategy: stop_strategy_type::CROSSVAL_STOPPING,
            lr_strategy: lr_strategy_type::SHUFFLE_CONNECTIONS,
            lr_strategy2: lr_strategy2_type::BTLS_ALPHA,

            max_epoch: 1000 * epoch_num,
            stop_epoch: 0,
            stop_train_loss: 0.0f64,

            exit_counter_threshold: epoch_num * 40,
            shuffle_counter_threshold: epoch_num * 20,
        };

        //Train network, stop at platue
        train_network(
            traindata_X,
            traindata_Y,
            crossvaldata_X,
            crossvaldata_Y,
            eval_metric,
            eval_metric_grad,
            train_normal_options,
            &mut alpha_max_vec,
            &mut loss_vec,
            &mut crossval_vec,
            arch_search,
            &mut loss_status,
        );

        //Prevent from getting stuck
        let loss_len = loss_vec.len() as u64;
        if loss_len > stuck_idx {
            stuck_idx = loss_len;
            stuck_counter = 0;
        } else {
            stuck_counter = stuck_counter + 1;
        }

        //Counter overflow
        if stuck_counter >= 4 {
            if loss_len <= 3 {
                //Network can not be trainned
                //Reset to first network
                println!("fail");

                //Delete file
                std::fs::remove_file((*arch_search).evolution_info.cur_path.clone());

                loss_status = loss_status_type::LOSS_OVERFLOW;

                (*arch_search).evolution_info.cur_path = min_path.clone();

                if (*arch_search).evolution_info.success_idx > 0 {
                    evolve_network(arch_search);
                }

                alpha_max_vec = Vec::new();

                stuck_counter = 0;
                stuck_idx = 0;
            } else {
                //Network plataue
                loss_status = loss_status_type::LOSS_PLATEAU;
            }
        }

        //When loss is not overflow
        if (loss_status == loss_status_type::LOSS_PLATEAU)
            || (loss_status == loss_status_type::NO_CONVERGENCE)
        {
            stuck_counter = 0;
            stuck_idx = 0;

            //Find lowest crossval
            let crossval_arr = arrayfire::Array::new(
                &crossval_vec,
                arrayfire::Dim4::new(&[crossval_vec.len() as u64, 1, 1, 1]),
            );
            let (_, min_idx) = arrayfire::imin(&crossval_arr, 0);

            let mut min_idx_cpu = vec![u32::default(); min_idx.elements()];
            min_idx.host(&mut min_idx_cpu);

            let stop_epoch = (min_idx_cpu[0] as u64) * epoch_num;

            loss_status = loss_status_type::LOSS_OVERFLOW;

            //Load network
            (*arch_search).neural_network = load_network2(&(*arch_search).evolution_info.cur_path);

            //Train Options
            let train_stop_options: train_network_options_type = train_network_options_type {
                stop_strategy: stop_strategy_type::STOP_AT_EPOCH,
                lr_strategy: lr_strategy_type::SHUFFLE_CONNECTIONS,
                lr_strategy2: lr_strategy2_type::BTLS_ALPHA,

                max_epoch: 1000 * epoch_num,
                stop_epoch: stop_epoch,
                stop_train_loss: 0.0f64,

                exit_counter_threshold: epoch_num * 40,
                shuffle_counter_threshold: epoch_num * 20,
            };

            //Train network, stop at lowest crossval
            train_network(
                traindata_X,
                traindata_Y,
                crossvaldata_X,
                crossvaldata_Y,
                eval_metric,
                eval_metric_grad,
                train_stop_options,
                &mut alpha_max_vec,
                &mut loss_vec,
                &mut crossval_vec,
                arch_search,
                &mut loss_status,
            );

            //Error in training
            if (loss_status != loss_status_type::PREDETERMINED_STOP) {
                continue;
            }

            //SAVE TRAINED NETWORK
            let removecsv = (*arch_search)
                .evolution_info
                .cur_path
                .clone()
                .replace(".csv", "");
            let tmpfilename = format!(
                "{}_train_num_{}.csv",
                removecsv,
                (*arch_search).evolution_info.success_idx
            );
            save_network2(&tmpfilename, &((*arch_search).neural_network));

            //SAVE LOSS DATA
            let tmpfilename2 = format!(
                "{}_train_num_{}.train_loss",
                removecsv,
                (*arch_search).evolution_info.success_idx
            );
            write_vec_cpu_to_csv(&tmpfilename2, &loss_vec);

            //SAVE ALPHA VALUES
            let tmpfilename3 = format!(
                "{}_train_num_{}.alpha",
                removecsv,
                (*arch_search).evolution_info.success_idx
            );
            write_vec_cpu_to_csv(&tmpfilename3, &alpha_max_vec);

            let tmpfilename4 = format!(
                "{}_train_num_{}.crossval_loss",
                removecsv,
                (*arch_search).evolution_info.success_idx
            );
            write_vec_cpu_to_csv(&tmpfilename4, &crossval_vec);

            //Crossval results
            let mut eval_metric_out = Vec::new();
            let mut Yhat_out = nohash_hasher::IntMap::default();

            validate_network(
                crossvaldata_X,
                crossvaldata_Y,
                eval_metric,
                arch_search,
                &mut Yhat_out,
                &mut eval_metric_out,
            );

            //Save crossvalidation loss
            let tmpfilename4 = format!(
                "{}_train_num_{}.metric",
                removecsv,
                (*arch_search).evolution_info.success_idx
            );
            write_vec_cpu_to_csv(&tmpfilename4, &eval_metric_out);

            //Test results
            let mut eval_metric_out = Vec::new();
            let mut Yhat_out = nohash_hasher::IntMap::default();

            validate_network(
                testdata_X,
                testdata_Y,
                eval_metric,
                arch_search,
                &mut Yhat_out,
                &mut eval_metric_out,
            );

            let tmpfilename5 = format!(
                "{}_train_num_{}",
                removecsv,
                (*arch_search).evolution_info.success_idx
            );
            hash_batch_to_files(&tmpfilename5, &mut Yhat_out);

            arch_search.evolution_info.cur_path = tmpfilename.clone();
            evolve_network(arch_search);

            alpha_max_vec = Vec::new();
        }

        if (*arch_search).evolution_info.success_idx > max_search_num {
            break;
        }
    }
}

pub fn create_start_archtecture2(
    input_size: u64,
    max_input_size: u64,

    output_size: u64,
    max_output_size: u64,

    active_size: u64,
    max_neuron_size: u64,

    batch_size: u64,
    traj_size: u64,

    proc_num: u64,

    dir_path: &str,
) -> arch_search_type {
    let mut active_size: u64 = active_size;
    let space_dims: u64 = 3;
    let sim_steps: u64 = 5000;

    if active_size < 20 {
        active_size = 20;
    }

    let mut neuron_rad: f64 = 0.1;

    //let mut sphere_radius: f64 = neuron_rad*(((active_size as f64)/PACKING_FACTOR).cbrt());

    let mut sphere_radius = (((2 * active_size) as f64) / SPHERE_DENSITY);
    sphere_radius = (sphere_radius / ((4.0 / 3.0) * std::f64::consts::PI)).cbrt();

    if sphere_radius < SPHERE_RAD_MIN {
        sphere_radius = SPHERE_RAD_MIN;
    }

    let mut netdata: network_metadata_type = network_metadata_type {
        neuron_size: max_neuron_size,
        input_size: input_size,
        output_size: output_size,
        proc_num: proc_num,
        active_size: active_size,
        space_dims: space_dims,
        step_num: sim_steps,
        batch_size: batch_size,
        del_unused_neuron: true,

        time_step: 0.1,
        nratio: 0.5,
        neuron_std: 0.001,
        sphere_rad: sphere_radius,
        neuron_rad: neuron_rad,
        con_rad: (sphere_radius / (proc_num as f64)) * 1.4f64,
        init_prob: 0.01,
        add_neuron_rate: 0.0,
        del_neuron_rate: 0.0,
        center_const: 0.005,
        spring_const: 0.01,
        repel_const: 0.01,
    };

    let temp_dims = arrayfire::Dim4::new(&[4, 1, 1, 1]);

    let mut glia_pos = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut neuron_pos = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut H = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut A = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut B = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut C = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut D = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut E = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut neuron_idx = arrayfire::constant::<i32>(0, temp_dims);

    let mut WValues = arrayfire::constant::<f64>(0.0, temp_dims);
    let mut WRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
    let mut WRowIdxCSR = arrayfire::constant::<i32>(0, temp_dims);
    let mut WColIdx = arrayfire::constant::<i32>(0, temp_dims);

    sphere_cell_collision_minibatch(&netdata, &mut glia_pos, &mut neuron_pos);
    active_size = neuron_pos.dims()[0];
    netdata.active_size = active_size;

    let savefilename = format!(
        "{}/{}D_sphere_{:.5}_rad.csv",
        dir_path, space_dims, sphere_radius
    );

    save_network(
        &savefilename,
        &netdata,
        &WValues,
        &WRowIdxCSR,
        &WColIdx,
        &H,
        &A,
        &B,
        &C,
        &D,
        &E,
        &glia_pos,
        &neuron_pos,
        &neuron_idx,
    );

    /*
    neuron_pos = NDsphere_from_NDcube(
        &neuron_pos,
        sphere_radius,

        sphere_radius
    );



    glia_pos = NDsphere_from_NDcube(
        &glia_pos,
        sphere_radius,

        sphere_radius
    );
    */

    sort_neuron_pos_sphere(&mut neuron_pos);

    let sqrt_input = (input_size as f64).sqrt().ceil() as u64;
    let plane_shape = vec![sqrt_input, sqrt_input];

    let mut input_neurons =
        plane_surface_on_NDsphere(&plane_shape, sphere_radius + (neuron_rad * 2.0));

    if input_neurons.dims()[0] > input_size {
        input_neurons = arrayfire::rows(&input_neurons, 0, (input_size - 1) as i64);
    }

    neuron_pos = arrayfire::join(0, &input_neurons, &neuron_pos);

    let output_neurons =
        arrayfire::constant::<f64>(0.0, arrayfire::Dim4::new(&[output_size, space_dims, 1, 1]));

    neuron_pos = arrayfire::join(0, &neuron_pos, &output_neurons);

    netdata.active_size = neuron_pos.dims()[0];

    active_size = neuron_pos.dims()[0];

    assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &netdata,
        &neuron_pos,
        &mut neuron_idx,
    );

    /*
    let hidden_num2 = active_size - output_size - input_size;
    let mut init_connection_num = 10;

    if init_connection_num >= hidden_num2
    {
        init_connection_num = hidden_num2;
    }
    else
    {
        let temp_num = ((hidden_num2 as f64)*INOUT_FACTOR) as u64 ;
        if temp_num >  init_connection_num
        {
            init_connection_num = temp_num;
        }

    }


    input_and_output_layers(
        &netdata,
        &neuron_pos,
        &neuron_idx,

        init_connection_num,
        init_connection_num,

        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx
    );






    fully_connected_hidden_layers(
        &neuron_pos,
        &neuron_idx,

        &mut netdata,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx
    );



    let in_idx = arrayfire::rows(&neuron_idx, 0, (input_size-1)  as i64);
    */

    let mut hidden_num2 = neuron_pos.dims()[0] - output_size - input_size;
    let mut init_connection_num = 10;

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * input_size,
        ray_neuron_intersect: false,
        ray_glia_intersect: false,
    };

    netdata.con_rad = 60.0 * netdata.neuron_rad;

    let neuron_num = neuron_pos.dims()[0];
    let hidden_idx_total = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = arrayfire::rows(&neuron_idx, 0, (input_size - 1) as i64);
    let input_pos = arrayfire::rows(&neuron_pos, 0, (input_size - 1) as i64);

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &netdata,
        &glia_pos,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    let neuron_num = neuron_pos.dims()[0];

    let hidden_idx_total = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = hidden_idx_total.clone();
    let input_pos = hidden_pos_total.clone();

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * hidden_num2,
        ray_neuron_intersect: true,
        ray_glia_intersect: true,
    };

    netdata.con_rad = 15.0 * netdata.neuron_rad;

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &netdata,
        &glia_pos,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    let neuron_num = neuron_pos.dims()[0];

    let input_idx = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let input_pos = arrayfire::rows(
        &neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let hidden_idx_total = arrayfire::rows(
        &neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &neuron_pos,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    netdata.con_rad = 60.0 * netdata.neuron_rad;

    let glia_pos_temp_dims = arrayfire::Dim4::new(&[4, space_dims, 1, 1]);

    let glia_pos_temp = arrayfire::constant::<f64>(-1000.0, glia_pos_temp_dims);

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * output_size,
        ray_neuron_intersect: false,
        ray_glia_intersect: false,
    };

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &netdata,
        &glia_pos_temp,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    //Populate WValues with random values
    let rand_dims = arrayfire::Dim4::new(&[WColIdx.dims()[0], 1, 1, 1]);
    WValues = netdata.neuron_std * arrayfire::randn::<f64>(rand_dims);

    //Clear input/output
    clear_input(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        netdata.input_size,
    );

    clear_output(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        netdata.neuron_size - netdata.output_size,
    );

    let in_idx = arrayfire::rows(&neuron_idx, 0, (input_size - 1) as i64);

    /*
    if traj_size > 1
    {
        self_loops(
            &netdata,
            //&neuron_pos,
            &neuron_idx,


            &mut WValues,
            &mut WRowIdxCOO,
            &mut WColIdx
        );
    }
    else
    {
        select_forward_sphere(
            &netdata,
            &mut WValues,
            &mut WRowIdxCOO,
            &mut WColIdx,
            &neuron_pos,
            &neuron_idx
        );
    }
    */

    self_loops(
        &netdata,
        //&neuron_pos,
        &neuron_idx,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );

    UAF_initial_as_identity(
        &netdata, //&mut H,
        &mut A, &mut B, &mut C, &mut D, &mut E,
    );

    delete_unused_neurons(
        &netdata,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        &mut glia_pos,
        &mut neuron_pos,
        &mut neuron_idx,
    );

    xavier_init(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        max_neuron_size,
        proc_num,
        &mut WValues,
        &mut H,
    );

    WRowIdxCSR = COO_to_CSR(&WRowIdxCOO, max_neuron_size);

    let mut active_size = neuron_idx.dims()[0];
    netdata.active_size = active_size;
    let con_num = WValues.dims()[0];
    let filename = format!(
        "{}/active_size_{}_proc_num_{}_con_num_{}.csv",
        dir_path, active_size, proc_num, con_num
    );

    save_network(
        &filename,
        &netdata,
        &WValues,
        &WRowIdxCSR,
        &WColIdx,
        &H,
        &A,
        &B,
        &C,
        &D,
        &E,
        &glia_pos,
        &neuron_pos,
        &neuron_idx,
    );

    let total_param_size = WValues.dims()[0]
        + H.dims()[0]
        + A.dims()[0]
        + B.dims()[0]
        + C.dims()[0]
        + D.dims()[0]
        + E.dims()[0];
    let network_params_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);

    let Wstart = 0;
    let Wend = (WValues.dims()[0] as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (H.dims()[0] as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (A.dims()[0] as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (B.dims()[0] as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (C.dims()[0] as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (D.dims()[0] as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (E.dims()[0] as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let mut network_params = arrayfire::constant::<f64>(0.0, network_params_dims);
    arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
    arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
    arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
    arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
    arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
    arrayfire::assign_seq(&mut network_params, &Dseqs, &D);
    arrayfire::assign_seq(&mut network_params, &Eseqs, &E);

    let mut neural_network: neural_network_type = neural_network_type {
        netdata: netdata,
        WRowIdxCSR: WRowIdxCSR,
        WColIdx: WColIdx,
        network_params: network_params,
        glia_pos: glia_pos,
        neuron_pos: neuron_pos,
        neuron_idx: neuron_idx,
    };

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

    let arch_search: arch_search_type = arch_search_type {
        neural_network: neural_network,

        evolution_info: evolution_info,
    };

    arch_search
}
