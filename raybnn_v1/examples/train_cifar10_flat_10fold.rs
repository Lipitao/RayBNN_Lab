extern crate arrayfire;
extern crate raybnn;
use arrayfire::print;
use nohash_hasher;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};

use csv::Writer;
use std::error::Error;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use raybnn::physics::update_f32::add_neuron_option_type;

fn save_vec_to_file(vec: &Vec<f32>, filename: &str) -> io::Result<()> {
    let mut file = File::create(filename)?;
    for value in vec {
        writeln!(file, "{}", value)?;
    }
    Ok(())
}

fn write_csv(data: &Vec<Vec<f32>>, filename: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(filename)?;

    for row in data {
        let string_row: Vec<String> = row.iter().map(|&num| num.to_string()).collect();
        wtr.write_record(&string_row)?;
    }

    wtr.flush()?;
    Ok(())
}

fn reshape(matrix: Vec<Vec<f32>>, new_rows: usize, new_cols: usize) -> Vec<Vec<f32>> {
    let flat: Vec<f32> = matrix.into_iter().flatten().collect();
    flat.chunks(new_cols).map(|chunk| chunk.to_vec()).collect()
}

fn ten_fold(
    train_x_filename: &str,
    train_y_filename: &str,
    test_x_filename: &str,
    test_y_filename: &str,
) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut train_x = raybnn::export::dataloader_f32::file_to_vec_cpu(train_x_filename);
    let mut train_y = raybnn::export::dataloader_f32::file_to_vec_cpu(train_y_filename);
    let test_x = raybnn::export::dataloader_f32::file_to_vec_cpu(test_x_filename);
    let test_y = raybnn::export::dataloader_f32::file_to_vec_cpu(test_y_filename);
    train_x.extend(test_x);
    train_y.extend(test_y);
    let folds = raybnn::export::dataloader_f32::nfold(&train_x, &train_y, 10);
    return folds;
}

#[allow(unused_must_use)]
fn main() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let folds = ten_fold(
        "./test_data/cifar10_flat/cifar_train_features_G.dat",
        "./test_data/cifar10_flat/cifar_train_labels_G.dat",
        "./test_data/cifar10_flat/cifar_val_features_G.dat",
        "./test_data/cifar10_flat/cifar_val_labels_G.dat",
    );

    for fold_index in 0..1 {
        //just test
        //let active_size_vec = vec![2000, 4000, 6000, 8000, 10000];
        let active_size_vec: Vec<u64> = vec![4000];
        for active_size in active_size_vec {
            let input_size: u64 = 1024;
            let max_input_size: u64 = 1024;
            let output_size: u64 = 10;
            let max_output_size: u64 = 10;
            let max_neuron_size: u64 = active_size + 2000;
            let batch_size: u64 = 1000;
            let traj_size: u64 = 1;
            let proc_num: u64 = 5;
            let dir_path: String = "/scratch/lain1385/project/raybnn_v1/temp".to_string();

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
                &dir_path,
            );
            println!("Created network successfully");

            //Load dataset
            println!("Load Dataset");
            let (train_x, train_y, test_x, test_y) = &folds[fold_index];
            let RSSI_TRAINX = raybnn::export::dataloader_f32::vec_cpu_to_hash(
                &train_x,
                max_input_size,
                batch_size * traj_size,
            );
            let RSSI_TRAINY = raybnn::export::dataloader_f32::vec_cpu_to_hash(
                &train_y,
                output_size,
                batch_size * traj_size,
            );
            let RSSI_TESTX = raybnn::export::dataloader_f32::vec_cpu_to_hash(
                &test_x,
                max_input_size,
                batch_size * traj_size,
            );
            let RSSI_TESTY = raybnn::export::dataloader_f32::vec_cpu_to_hash(
                &test_y,
                output_size,
                batch_size * traj_size,
            );
            println!("length of RSSI_TRAINX:{}", RSSI_TRAINX.len());
            println!("length of RSSI_TRAINY:{}", RSSI_TRAINY.len());
            println!("length of RSSI_TESTX:{}", RSSI_TESTX.len());
            println!("length of RSSI_TESTY:{}", RSSI_TESTY.len());

            let test_y_flat: Vec<f32> = RSSI_TESTY
                .clone()
                .values()
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            let test_y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
            let af_test_y = arrayfire::Array::new(&test_y_flat, test_y_dims);
            /*
            let TOTAL_RSSI_TESTX = raybnn::export::dataloader_f32::file_to_hash_cpu(
                "./test_data/cifar10_flat/cifar_val_features_G.dat",
                max_input_size,
                batch_size*traj_size
            );
            println!("number of TESTX:{}", TOTAL_RSSI_TESTX.len());


            let RSSI_TESTY = raybnn::export::dataloader_f32::file_to_hash_cpu(
                "./test_data/cifar10_flat/cifar_val_labels_G.dat",
                output_size,
                batch_size*traj_size
            );
            println!("number of TESTy:{}", RSSI_TESTY.len());

            arrayfire::sync(DEVICE);
            arrayfire::device_gc();
            arrayfire::sync(DEVICE);


            let mut RSSI_TESTX = TOTAL_RSSI_TESTX.clone();



            let TOTAL_RSSI_TRAINX = raybnn::export::dataloader_f32::file_to_hash_cpu(
                "./test_data/cifar10_flat/cifar_train_features_G.dat",
                max_input_size,
                batch_size*traj_size
            );
            println!("number of TRAINX:{}", TOTAL_RSSI_TRAINX.len());

            let RSSI_TRAINY = raybnn::export::dataloader_f32::file_to_hash_cpu(
                "./test_data/cifar10_flat/cifar_train_labels_G.dat",
                output_size,
                batch_size*traj_size
            );
            println!("number of TRAINy:{}", RSSI_TRAINY.len());
            arrayfire::sync(DEVICE);
            arrayfire::device_gc();
            arrayfire::sync(DEVICE);

            let mut RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();
            */
            println!("success");

            let sphere_rad: f32 = arch_search.neural_network.netdata.sphere_rad;
            println!("Sphere Rad: {}", sphere_rad);
            /*
            let input_neurons = raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere_1D(
                arch_search.neural_network.netdata.sphere_rad+0.2,
                input_size
            );
            */
            let input_neurons = raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere(
                arch_search.neural_network.netdata.sphere_rad + 0.2,
                32,
                32,
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

            let input_neuron_con_rad: f32 = sphere_rad / 5.0;

            let new_active_size: u64 = 10;
            let init_connection_num: u64 = 10;
            let hidden_neuron_con_rad: f32 = input_neuron_con_rad;
            let output_neuron_con_rad: f32 = input_neuron_con_rad;

            let add_neuron_options: raybnn::physics::update_f32::add_neuron_option_type =
                raybnn::physics::update_f32::add_neuron_option_type {
                    new_active_size,
                    init_connection_num,
                    input_neuron_con_rad,
                    hidden_neuron_con_rad,
                    output_neuron_con_rad,
                };

            raybnn::physics::update_f32::add_neuron_to_existing3(
                &add_neuron_options,
                &mut arch_search,
            );

            let stop_strategy =
                raybnn::interface::autotrain_f32::stop_strategy_type::STOP_AT_TRAIN_LOSS;
            let lr_strategy =
                raybnn::interface::autotrain_f32::lr_strategy_type::SHUFFLE_CONNECTIONS;
            let lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::MAX_ALPHA;

            let max_epoch: u64 = 100;
            let stop_epoch: u64 = 100000;
            let stop_train_loss: f32 = 0.00005;
            let max_alpha: f32 = 0.005;

            let exit_counter_threshold: u64 = 100000;
            let shuffle_counter_threshold: u64 = 200;

            let total_epochs: u32 = 1000;

            //Train Options
            /*
            let _train_stop_options = raybnn::interface::autotrain_f32::train_network_options_type {
                stop_strategy: stop_strategy,
                lr_strategy: lr_strategy,
                lr_strategy2: lr_strategy2,

                max_epoch: max_epoch,
                stop_epoch: stop_epoch,
                stop_train_loss: stop_train_loss,

                exit_counter_threshold: exit_counter_threshold,
                shuffle_counter_threshold: shuffle_counter_threshold,
            };
            */

            let mut alpha_max_vec = vec![max_alpha; 1000];
            let mut loss_vec = Vec::new();
            let mut crossval_vec = Vec::new();
            let mut loss_status = raybnn::interface::autotrain_f32::loss_status_type::LOSS_OVERFLOW;

            println!("Start training");
            arrayfire::device_gc();
            let epoch = 100;

            for count in 1..=epoch {
                let train_stop_options =
                    raybnn::interface::autotrain_f32::train_network_options_type {
                        stop_strategy: stop_strategy.clone(),
                        lr_strategy: lr_strategy.clone(),
                        lr_strategy2: lr_strategy2.clone(),

                        max_epoch: count,
                        stop_epoch: stop_epoch,
                        stop_train_loss: stop_train_loss,

                        exit_counter_threshold: exit_counter_threshold,
                        shuffle_counter_threshold: shuffle_counter_threshold,
                    };
                println!("Epoch: {}", count);
                raybnn::interface::autotrain_f32::train_network(
                    &RSSI_TRAINX,
                    &RSSI_TRAINY,
                    &RSSI_TRAINX,
                    &RSSI_TRAINY,
                    raybnn::optimal::loss_f32::sigmoid_cross_entropy,
                    raybnn::optimal::loss_f32::sigmoid_cross_entropy_grad,
                    train_stop_options,
                    &mut alpha_max_vec,
                    &mut loss_vec,
                    &mut crossval_vec,
                    &mut arch_search,
                    &mut loss_status,
                );

                let mut Yhat_out = nohash_hasher::IntMap::default();
                raybnn::interface::autotest_f32::test_network(
                    &RSSI_TESTX,
                    &mut arch_search,
                    &mut Yhat_out,
                );
                let predict_y_flat: Vec<f32> = Yhat_out
                    .clone()
                    .values()
                    .into_iter()
                    .flatten()
                    .cloned()
                    .collect();
                let predict_y_dims =
                    arrayfire::Dim4::new(&[output_size, batch_size * traj_size, 1, 1]);
                let af_predict_y = arrayfire::Array::new(&predict_y_flat, predict_y_dims);
                let val_loss =
                    raybnn::optimal::loss_f32::sigmoid_cross_entropy(&af_predict_y, &af_test_y);
                println!("val_Loss: {}", val_loss);
            }

            /*
            let mut Yhat_out = nohash_hasher::IntMap::default();
            let num_pairs = RSSI_TESTX.len();
            //println!("number of pairs:{}", num_pairs);
            raybnn::interface::autotest_f32::test_network(
                &RSSI_TESTX,
                &mut arch_search,
                &mut Yhat_out,
            );



            let mut output_test_y: Vec<Vec<f32>> = Vec::new();
            for values in Yhat_out.values() {
                output_test_y.push(values.clone());
            }

            let output_test_y = reshape(output_test_y, 10000, 10);
            //println!("length of output_test_y:{}", output_test_y.len());
            let file_path = format!("./output_csv/cifar10_flat/G_size{}_{}.csv", active_size, fold_index);
            write_csv(&output_test_y, &file_path);
            println!("Data saved to {}", file_path);
            */
        }
    }
}
