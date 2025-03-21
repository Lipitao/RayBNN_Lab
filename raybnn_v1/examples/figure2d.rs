/*
Plot Figure 2d Runtimes of Various Raytracing Algorithms

This code benchmarks the runtimes of RT-1,RT-2, and RT-3.
RT-3 has variable 20, 40, and 60 neuron radii

Generated files
./RT1_run_time.csv       List of time benchmarks for RT1 algorithm
./RT2_run_time.csv       List of time benchmarks for RT2 algorithm
./RT3_20_run_time.csv    List of time benchmarks for RT3 algorithm with neuron radius 20
./RT3_40_run_time.csv    List of time benchmarks for RT3 algorithm with neuron radius 40
./RT3_60_run_time.csv    List of time benchmarks for RT3 algorithm with neuron radius 60
./neuron_num_list.csv    List of neuron sizes

*/

extern crate arrayfire;
extern crate raybnn;

use std::time::{Duration, Instant};

use raybnn::physics::raytrace_f32::raytrace_option_type;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

const two: f64 = 2.0;

use raybnn::physics::update_f32::add_neuron_option_type;

fn run_RT3(RT3_rad: f32) {
    //List of Neuron Sizes and Number of connections

    let neuron_num_list = vec![
        250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 240000,
    ];
    let con_num_list = vec![
        14000, 25000, 50000, 100000, 200489, 400466, 800588, 1600000, 3200000, 6400000, 12000000,
    ];

    let mut neuron_num_list_f64 = Vec::new();
    for element in neuron_num_list.clone() {
        neuron_num_list_f64.push(element as f64);
    }

    //Save Collision Runtime
    raybnn::export::dataloader_f64::write_vec_cpu_to_csv(
        "./neuron_num_list.csv",
        &neuron_num_list_f64,
    );

    let mut timevec: Vec<f64> = Vec::new();
    let mut convec: Vec<u64> = Vec::new();

    for ii in 0..neuron_num_list.len() {
        let neuron_num = neuron_num_list[ii].clone();

        let dir_path = "/tmp/".to_string();

        let input_size: u64 = 105;
        let max_input_size: u64 = 105;

        let max_output_size: u64 = 19;
        let output_size: u64 = 19;

        let max_neuron_size: u64 = neuron_num + 2000;

        let mut batch_size: u64 = 100;
        let mut traj_size = 20;

        //Create Neural Network

        let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(
            input_size,
            max_input_size,
            output_size,
            max_output_size,
            max_neuron_size,
            batch_size,
            traj_size,
            &dir_path,
        );

        //Limit connection radiuses for raytracing

        let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
            new_active_size: neuron_num,
            init_connection_num: 1,
            input_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
            hidden_neuron_con_rad: 15.0 * arch_search.neural_network.netdata.neuron_rad,
            output_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
        };

        //Add new neurons
        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::update_f32::add_neuron_to_existing3(&add_neuron_options, &mut arch_search);
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Add neurons {}, {}",
            duration.as_secs_f64(),
            arch_search.neural_network.neuron_pos.dims()[0]
        );

        let mut WRowIdxCOO =
            raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

        //Weight Initialization

        let WValuesdims0 = arch_search.neural_network.WColIdx.dims()[0];

        let network_paramsdims0 = arch_search.neural_network.network_params.dims()[0];

        let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

        let Wstart = 0;
        let Wend = (WValuesdims0 as i64) - 1;

        let Hstart = Wend + 1;
        let Hend = Hstart + (Hdims0 as i64) - 1;

        let Astart = Hend + 1;
        let Aend = Astart + (Hdims0 as i64) - 1;

        let Bstart = Aend + 1;
        let Bend = Bstart + (Hdims0 as i64) - 1;

        let Cstart = Bend + 1;
        let Cend = Cstart + (Hdims0 as i64) - 1;

        let Dstart = Cend + 1;
        let Dend = Dstart + (Hdims0 as i64) - 1;

        let Estart = Dend + 1;
        let Eend = Estart + (Hdims0 as i64) - 1;

        let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
        let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
        let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
        let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
        let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
        let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
        let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

        let mut WValues = arrayfire::index(&(arch_search.neural_network.network_params), &Wseqs);

        arch_search.neural_network.netdata.con_rad =
            60.0 * arch_search.neural_network.netdata.neuron_rad;

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let gidx1 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
        gidx1.host(&mut gidx1_cpu);
        drop(gidx1);

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let neuron_num2 = arch_search.neural_network.neuron_pos.dims()[0];

        let hidden_idx_total = arrayfire::rows(
            &arch_search.neural_network.neuron_idx,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );
        let hidden_pos_total = arrayfire::rows(
            &arch_search.neural_network.neuron_pos,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );

        let input_idx = hidden_idx_total.clone();
        let input_pos = hidden_pos_total.clone();

        arch_search.neural_network.netdata.con_rad =
            RT3_rad * arch_search.neural_network.netdata.neuron_rad;

        println!("hidden_pos_total {}", hidden_pos_total.dims()[0]);
        println!("input_pos {}", input_pos.dims()[0]);

        let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

        WValues = arrayfire::constant::<f32>(0.0, temp_dims);
        WRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
        arch_search.neural_network.WColIdx = arrayfire::constant::<i32>(0, temp_dims);

        //Set target number of connections to raytrace

        let raytrace_options: raytrace_option_type = raytrace_option_type {
            max_rounds: 1000000,
            input_connection_num: con_num_list[ii].clone(),
            ray_neuron_intersect: true,
            ray_glia_intersect: true,
        };

        //Raytrace the neurons

        println!("start");
        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::raytrace_f32::RT3_distance_limited_directly_connected(
            &raytrace_options,
            &arch_search.neural_network.netdata,
            &arch_search.neural_network.glia_pos,
            &input_pos,
            &input_idx,
            &hidden_pos_total,
            &hidden_idx_total,
            &mut WRowIdxCOO,
            &mut arch_search.neural_network.WColIdx,
        );
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Ray trace {}, {}",
            duration.as_secs_f64(),
            WRowIdxCOO.dims()[0]
        );

        timevec.push(duration.as_secs_f64());
        convec.push(WRowIdxCOO.dims()[0]);
    }

    //Record the time for raytracing

    println!("con: {:?}", convec);
    println!("neurons: {:?}", neuron_num_list.clone());
    println!("time: {:?}", timevec);

    let filename = format!("./RT3_{}_run_time.csv", RT3_rad as u64);

    //Save Collision Runtime
    raybnn::export::dataloader_f64::write_vec_cpu_to_csv(&filename, &timevec);
}

#[allow(unused_must_use)]
fn main() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    //RT-2 Algorithm

    //List of Neuron Sizes and Number of connections

    let neuron_num_list = vec![250, 500, 1000, 2000, 4000, 8000, 10000];
    let con_num_list = vec![14000, 25000, 50000, 100000, 200489, 400466, 500466];

    let mut neuron_num_list_f64 = Vec::new();
    for element in neuron_num_list.clone() {
        neuron_num_list_f64.push(element as f64);
    }

    //Save Collision Runtime
    raybnn::export::dataloader_f64::write_vec_cpu_to_csv(
        "./neuron_num_list2.csv",
        &neuron_num_list_f64,
    );

    let mut timevec: Vec<f64> = Vec::new();
    let mut convec: Vec<u64> = Vec::new();

    for ii in 0..neuron_num_list.len() {
        let neuron_num = neuron_num_list[ii].clone();

        println!("neuron_num {}", neuron_num);

        let dir_path = "/tmp/".to_string();

        let input_size: u64 = 105;
        let max_input_size: u64 = 105;

        let max_output_size: u64 = 19;
        let output_size: u64 = 19;

        let max_neuron_size: u64 = neuron_num + 2000;

        let mut batch_size: u64 = 100;
        let mut traj_size = 20;

        //Create Neural Network

        let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(
            input_size,
            max_input_size,
            output_size,
            max_output_size,
            max_neuron_size,
            batch_size,
            traj_size,
            &dir_path,
        );

        //Limit connection radiuses for raytracing

        let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
            new_active_size: neuron_num,
            init_connection_num: 1,
            input_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
            hidden_neuron_con_rad: 15.0 * arch_search.neural_network.netdata.neuron_rad,
            output_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
        };

        //Add new neurons
        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::update_f32::add_neuron_to_existing3(&add_neuron_options, &mut arch_search);
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Add neurons {}, {}",
            duration.as_secs_f64(),
            arch_search.neural_network.neuron_pos.dims()[0]
        );

        let mut WRowIdxCOO =
            raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

        //Weight Initialization

        let WValuesdims0 = arch_search.neural_network.WColIdx.dims()[0];

        let network_paramsdims0 = arch_search.neural_network.network_params.dims()[0];

        let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

        let Wstart = 0;
        let Wend = (WValuesdims0 as i64) - 1;

        let Hstart = Wend + 1;
        let Hend = Hstart + (Hdims0 as i64) - 1;

        let Astart = Hend + 1;
        let Aend = Astart + (Hdims0 as i64) - 1;

        let Bstart = Aend + 1;
        let Bend = Bstart + (Hdims0 as i64) - 1;

        let Cstart = Bend + 1;
        let Cend = Cstart + (Hdims0 as i64) - 1;

        let Dstart = Cend + 1;
        let Dend = Dstart + (Hdims0 as i64) - 1;

        let Estart = Dend + 1;
        let Eend = Estart + (Hdims0 as i64) - 1;

        let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
        let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
        let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
        let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
        let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
        let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
        let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

        let mut WValues = arrayfire::index(&(arch_search.neural_network.network_params), &Wseqs);

        arch_search.neural_network.netdata.con_rad =
            60.0 * arch_search.neural_network.netdata.neuron_rad;

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let gidx1 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
        gidx1.host(&mut gidx1_cpu);
        drop(gidx1);

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let neuron_num2 = arch_search.neural_network.neuron_pos.dims()[0];

        let hidden_idx_total = arrayfire::rows(
            &arch_search.neural_network.neuron_idx,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );
        let hidden_pos_total = arrayfire::rows(
            &arch_search.neural_network.neuron_pos,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );

        let input_idx = hidden_idx_total.clone();
        let input_pos = hidden_pos_total.clone();

        arch_search.neural_network.netdata.con_rad =
            1000000.0 * arch_search.neural_network.netdata.neuron_rad;

        println!("hidden_pos_total {}", hidden_pos_total.dims()[0]);
        println!("input_pos {}", input_pos.dims()[0]);

        let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

        WValues = arrayfire::constant::<f32>(0.0, temp_dims);
        WRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
        arch_search.neural_network.WColIdx = arrayfire::constant::<i32>(0, temp_dims);

        //Set target number of connections to raytrace

        let raytrace_options: raytrace_option_type = raytrace_option_type {
            max_rounds: 1000000,
            input_connection_num: con_num_list[ii].clone(),
            ray_neuron_intersect: true,
            ray_glia_intersect: true,
        };

        //Raytrace the neurons

        println!("start");
        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::raytrace_f32::RT2_directly_connected(
            &raytrace_options,
            &arch_search.neural_network.netdata,
            &arch_search.neural_network.glia_pos,
            &input_pos,
            &input_idx,
            &hidden_pos_total,
            &hidden_idx_total,
            &mut WRowIdxCOO,
            &mut arch_search.neural_network.WColIdx,
        );
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Ray trace {}, {}",
            duration.as_secs_f64(),
            WRowIdxCOO.dims()[0]
        );

        timevec.push(duration.as_secs_f64());
        convec.push(WRowIdxCOO.dims()[0]);
    }

    //Record the time for raytracing

    println!("con: {:?}", convec);
    println!("neurons: {:?}", neuron_num_list.clone());
    println!("time: {:?}", timevec);

    //Save Collision Runtime
    raybnn::export::dataloader_f64::write_vec_cpu_to_csv("./RT2_run_time.csv", &timevec);
    std::thread::sleep(std::time::Duration::from_secs(20));

    //RT-1 Algorithm

    arrayfire::set_seed(1231);

    //List of Neuron Sizes and Number of connections

    let neuron_num_list = vec![250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000];
    let con_num_list = vec![
        14000, 25000, 50000, 100000, 200489, 400466, 800588, 1600000, 3200000,
    ];

    let mut neuron_num_list_f64 = Vec::new();
    for element in neuron_num_list.clone() {
        neuron_num_list_f64.push(element as f64);
    }

    //Save Collision Runtime
    raybnn::export::dataloader_f64::write_vec_cpu_to_csv(
        "./neuron_num_list.csv",
        &neuron_num_list_f64,
    );

    let mut timevec: Vec<f64> = Vec::new();
    let mut convec: Vec<u64> = Vec::new();

    for ii in 0..neuron_num_list.len() {
        let neuron_num = neuron_num_list[ii].clone();

        let dir_path = "/tmp/".to_string();

        let input_size: u64 = 105;
        let max_input_size: u64 = 105;

        let max_output_size: u64 = 19;
        let output_size: u64 = 19;

        let max_neuron_size: u64 = neuron_num + 2000;

        let mut batch_size: u64 = 100;
        let mut traj_size = 20;

        //Create Neural Network

        let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(
            input_size,
            max_input_size,
            output_size,
            max_output_size,
            max_neuron_size,
            batch_size,
            traj_size,
            &dir_path,
        );

        //Limit connection radiuses for raytracing

        let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
            new_active_size: neuron_num,
            init_connection_num: 1,
            input_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
            hidden_neuron_con_rad: 15.0 * arch_search.neural_network.netdata.neuron_rad,
            output_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
        };

        //Add new neurons
        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::update_f32::add_neuron_to_existing3(&add_neuron_options, &mut arch_search);
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Add neurons {}, {}",
            duration.as_secs_f64(),
            arch_search.neural_network.neuron_pos.dims()[0]
        );

        let mut WRowIdxCOO =
            raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

        //Weight Initialization

        let WValuesdims0 = arch_search.neural_network.WColIdx.dims()[0];

        let network_paramsdims0 = arch_search.neural_network.network_params.dims()[0];

        let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

        let Wstart = 0;
        let Wend = (WValuesdims0 as i64) - 1;

        let Hstart = Wend + 1;
        let Hend = Hstart + (Hdims0 as i64) - 1;

        let Astart = Hend + 1;
        let Aend = Astart + (Hdims0 as i64) - 1;

        let Bstart = Aend + 1;
        let Bend = Bstart + (Hdims0 as i64) - 1;

        let Cstart = Bend + 1;
        let Cend = Cstart + (Hdims0 as i64) - 1;

        let Dstart = Cend + 1;
        let Dend = Dstart + (Hdims0 as i64) - 1;

        let Estart = Dend + 1;
        let Eend = Estart + (Hdims0 as i64) - 1;

        let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
        let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
        let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
        let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
        let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
        let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
        let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

        let mut WValues = arrayfire::index(&(arch_search.neural_network.network_params), &Wseqs);

        arch_search.neural_network.netdata.con_rad =
            60.0 * arch_search.neural_network.netdata.neuron_rad;

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let gidx1 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
        gidx1.host(&mut gidx1_cpu);
        drop(gidx1);

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let neuron_num2 = arch_search.neural_network.neuron_pos.dims()[0];

        let hidden_idx_total = arrayfire::rows(
            &arch_search.neural_network.neuron_idx,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );
        let hidden_pos_total = arrayfire::rows(
            &arch_search.neural_network.neuron_pos,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );

        let input_idx = hidden_idx_total.clone();
        let input_pos = hidden_pos_total.clone();

        arch_search.neural_network.netdata.con_rad =
            1000.0 * arch_search.neural_network.netdata.neuron_rad;

        println!("hidden_pos_total {}", hidden_pos_total.dims()[0]);
        println!("input_pos {}", input_pos.dims()[0]);

        let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

        WValues = arrayfire::constant::<f32>(0.0, temp_dims);
        WRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
        arch_search.neural_network.WColIdx = arrayfire::constant::<i32>(0, temp_dims);

        let mut ray_num = 10000;

        //Raytrace the neurons

        println!("start");
        arrayfire::sync(0);
        let start = Instant::now();

        raybnn::physics::raytrace_f32::RT1_random_rays(
            ray_num,
            con_num_list[ii].clone(),
            &arch_search.neural_network.netdata,
            &arch_search.neural_network.neuron_pos,
            &arch_search.neural_network.neuron_idx,
            &mut WRowIdxCOO,
            &mut arch_search.neural_network.WColIdx,
        );
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Ray trace {}, {}",
            duration.as_secs_f64(),
            WRowIdxCOO.dims()[0]
        );

        timevec.push(duration.as_secs_f64());
        convec.push(WRowIdxCOO.dims()[0]);
    }

    //Record the time for raytracing

    println!("con: {:?}", convec);
    println!("neurons: {:?}", neuron_num_list.clone());
    println!("time: {:?}", timevec);

    //Save Collision Runtime
    raybnn::export::dataloader_f64::write_vec_cpu_to_csv("./RT1_run_time.csv", &timevec);

    std::thread::sleep(std::time::Duration::from_secs(20));

    //Run RT-3  20 Neuron radius
    arrayfire::set_seed(1231);

    run_RT3(20.0);

    std::thread::sleep(std::time::Duration::from_secs(20));

    //Run RT-3  60 Neuron radius
    arrayfire::set_seed(1231);

    run_RT3(60.0);

    std::thread::sleep(std::time::Duration::from_secs(20));

    //Run RT-3  40 Neuron radius
    arrayfire::set_seed(1231);

    run_RT3(40.0);

    std::thread::sleep(std::time::Duration::from_secs(20));
}
