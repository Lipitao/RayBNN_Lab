/*
Plot Figure 4 Running RayBNN for the Alcala Dataset

Run the 10 Fold Testing for the Alcala Dataset in Figure 4
Note that CUDA has compile the kernels at runtime so the first run is slower. Tested on RTX 3090.

Generated files
./info_*.csv          Information about the training time and number of parameters
./test_act_*.csv      Actual test results
./test_pred_*.csv     Predicted test results

*/

extern crate arrayfire;
extern crate raybnn;
use nohash_hasher;
use std::collections::HashMap;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use raybnn::physics::update_f32::add_neuron_option_type;

#[allow(unused_must_use)]
fn main() {
    //Set CUDA and GPU Device 0
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    //Run the code 11 times
    //The first run is slower because it has to compile CUDA/OpenCL kernels
    for fold in 0..11 {
        arrayfire::set_seed(10 + fold as u64);

        arrayfire::sync(DEVICE);
        arrayfire::device_gc();
        arrayfire::sync(DEVICE);

        //Set Parameters
        //let sys_time = std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs() as u64;
        //arrayfire::set_seed(sys_time);

        let conn_vec: Vec<u64> = vec![10, 10, 10, 50, 60, 20, 20, 20, 20, 20, 10];

        let neuron_size_vec: Vec<u64> = vec![40, 50, 70, 160, 170, 180, 190, 200, 230, 270, 300];

        let arg_input_vec: Vec<u64> = vec![6, 8, 12, 14, 16, 32, 50, 75, 100, 125, 162];

        let dir_path = "/tmp/".to_string();

        //Set More Neural Network Parameters

        let max_input_size: u64 = 162;
        let mut input_size: u64 = 6;

        let max_output_size: u64 = 2;
        let output_size: u64 = 2;

        let max_neuron_size: u64 = 700;

        let mut batch_size: u64 = 100;
        let mut traj_size = 20;

        let mut proc_num = 5;

        let train_size: u64 = 60000;
        let test_size: u64 = 10000;

        let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

        //Create Start Neural Network
        println!("Creating Network");

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

        arch_search.neural_network.netdata.proc_num = proc_num;

        //Load dataset
        println!("Load Dataset");

        let TOTAL_RSSI_TESTX = raybnn::export::dataloader_f32::file_to_hash_cpu(
            "./test_data/Alcala_TESTX.dat",
            max_input_size,
            batch_size * traj_size,
        );

        let RSSI_TESTY = raybnn::export::dataloader_f32::file_to_hash_cpu(
            "./test_data/Alcala_TESTY.dat",
            output_size,
            batch_size * traj_size,
        );
        arrayfire::sync(DEVICE);

        let mut RSSI_TESTX = TOTAL_RSSI_TESTX.clone();

        let TOTAL_RSSI_TRAINX = raybnn::export::dataloader_f32::file_to_hash_cpu(
            "./test_data/Alcala_TRAINX.dat",
            max_input_size,
            batch_size * traj_size,
        );

        let RSSI_TRAINY = raybnn::export::dataloader_f32::file_to_hash_cpu(
            "./test_data/Alcala_TRAINY.dat",
            output_size,
            batch_size * traj_size,
        );
        arrayfire::sync(DEVICE);

        let mut RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();

        //Transfer Learning Training Loop

        let mut prev_target_neuron_size = 0;

        for ii in 0..arg_input_vec.len() {
            RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();

            RSSI_TESTX = TOTAL_RSSI_TESTX.clone();

            arch_search.neural_network.neuron_pos = arrayfire::rows(
                &arch_search.neural_network.neuron_pos,
                input_size as i64,
                (arch_search.neural_network.neuron_pos.dims()[0] - 1) as i64,
            );

            input_size = arg_input_vec[ii].clone();
            arch_search.neural_network.netdata.input_size = input_size;

            let mut WRowIdxCOO =
                raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

            arch_search.neural_network.netdata.neuron_std = 0.001;
            arch_search.neural_network.netdata.con_rad =
                (arch_search.neural_network.netdata.sphere_rad / (proc_num as f32)) * 2.0;

            //Initialize Input Neurons

            /*
            let input_neurons = raybnn::physics::initial_f32::golden_spiral_3D(
                arch_search.neural_network.netdata.sphere_rad+0.2,
                input_size
            );
            */

            let input_neurons =
                raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere_1D(
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

            //Modify Input Dimensions

            if (prev_target_neuron_size != neuron_size_vec[ii]) {
                let new_neuron_num = neuron_size_vec[ii]
                    - (arch_search.neural_network.neuron_idx.dims()[0] - input_size - output_size);

                prev_target_neuron_size = neuron_size_vec[ii];

                let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
                    new_active_size: new_neuron_num,
                    init_connection_num: conn_vec[ii as usize].clone(),
                    input_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
                    hidden_neuron_con_rad: 110.0 * arch_search.neural_network.netdata.neuron_rad,
                    output_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
                };

                raybnn::physics::update_f32::add_neuron_to_existing3(
                    &add_neuron_options,
                    &mut arch_search,
                );
            }

            raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
                max_input_size,
                max_output_size,
                &arch_search.neural_network.netdata,
                &arch_search.neural_network.neuron_pos,
                &mut arch_search.neural_network.neuron_idx,
            );

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

            //Modify training dataset

            if (input_size != max_input_size) {
                let mut tempRSSI = TOTAL_RSSI_TRAINX.clone();

                for (key, value) in &TOTAL_RSSI_TRAINX {
                    let mut tempvec = value.clone();

                    let veclen = tempvec.len() as u64;
                    let mut formatarr = arrayfire::Array::new(
                        &tempvec,
                        arrayfire::Dim4::new(&[max_input_size, veclen / max_input_size, 1, 1]),
                    );

                    formatarr = arrayfire::rows(&formatarr, 0, (input_size - 1) as i64);

                    tempvec = vec![f32::default(); formatarr.elements()];
                    formatarr.host(&mut tempvec);

                    tempRSSI.insert(key.clone(), tempvec);
                }

                RSSI_TRAINX = tempRSSI;
            }
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

            let mut alpha0: f32 = 0.0001;

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
            let epoch_num = (train_size / (batch_size * traj_size)) - 1;

            let mut max_epoch = 1000;

            let mut loss_val = 100000.0;
            let mut loss_val2 = 100000.0;
            let mut prev_loss = 100000.0;

            arrayfire::device_gc();
            arrayfire::sync(DEVICE);

            let start_time = std::time::Instant::now();

            //Training Loop

            for i in 0..max_epoch {
                batch_idx = i % epoch_num;

                train_X = arrayfire::Array::new(&RSSI_TRAINX[&batch_idx], train_X_dims);

                Y = arrayfire::Array::new(&RSSI_TRAINY[&batch_idx], Y_dims);
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

                //Get Yhat
                let mut idxrs = arrayfire::Indexer::default();
                let seq1 = arrayfire::Seq::new(0.0f32, (batch_size - 1) as f32, 1.0);
                let seq2 = arrayfire::Seq::new((proc_num - 1) as f32, (Qslices - 1) as f32, 1.0);
                idxrs.set_index(&idxsel, 0, None);
                idxrs.set_index(&seq1, 1, None);
                idxrs.set_index(&seq2, 2, None);
                let Yhat = arrayfire::index_gen(&Q, idxrs);
                loss_val = raybnn::optimal::loss_f32::RMSE(&Yhat, &Y);

                if i == 0 {
                    loss_val2 = loss_val;
                } else {
                    loss_val2 = 0.99 * loss_val2 + 0.01 * loss_val;
                }

                //Backward pass

                raybnn::neural::network_f32::state_space_backward_group2(
                    &arch_search.neural_network.netdata,
                    &X,
                    &arch_search.neural_network.network_params,
                    &Z,
                    &Q,
                    &Y,
                    raybnn::optimal::loss_f32::MSE_grad,
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

                alpha0 = (0.002 / (loss_val + 1e-10)).min(0.007);

                if ii == 11 {
                    alpha0 = alpha0.min(0.005);
                }

                arch_search.neural_network.network_params =
                    arch_search.neural_network.network_params + (alpha0 * -grad.clone());

                if (i > 300) && ((i % 50) == 0) {
                    if (prev_loss - loss_val).abs() / loss_val < 0.007 {
                        break;
                    }

                    prev_loss = loss_val;
                }
            }

            arrayfire::sync(DEVICE);
            println!("loss_val {}", loss_val);

            let elapsed: f32 = start_time.elapsed().as_secs_f32();
            println!("elapsed {}", elapsed);

            traj_size = 20;

            arrayfire::device_gc();

            //Change Input Dimensions of Testing dataset

            if (input_size != max_input_size) {
                let mut tempRSSI = TOTAL_RSSI_TESTX.clone();

                for (key, value) in &TOTAL_RSSI_TESTX {
                    let mut tempvec = value.clone();

                    let veclen = tempvec.len() as u64;
                    let mut formatarr = arrayfire::Array::new(
                        &tempvec,
                        arrayfire::Dim4::new(&[max_input_size, veclen / max_input_size, 1, 1]),
                    );

                    formatarr = arrayfire::rows(&formatarr, 0, (input_size - 1) as i64);

                    tempvec = vec![f32::default(); formatarr.elements()];
                    formatarr.host(&mut tempvec);

                    tempRSSI.insert(key.clone(), tempvec);
                }

                RSSI_TESTX = tempRSSI;
            }
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

            let Qslices: u64 = Q.dims()[2];

            let X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_steps, 1]);
            let mut X = arrayfire::constant::<f32>(0.0, X_dims);

            let test_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
            let mut test_X = arrayfire::constant::<f32>(0.0, temp_dims);
            let Y_dims = arrayfire::Dim4::new(&[output_size, batch_size, traj_size, 1]);
            let mut Y = arrayfire::constant::<f32>(0.0, temp_dims);
            let mut batch_idx = 0;
            let epoch_num = (test_size / (batch_size * traj_size));

            let temp_Y_dims = arrayfire::Dim4::new(&[1, output_size, 1, 1]);

            let mut Yhat_arr = arrayfire::constant::<f32>(0.0, temp_Y_dims);
            let mut Y_arr = arrayfire::constant::<f32>(0.0, temp_Y_dims);

            let newdims = arrayfire::Dim4::new(&[output_size, batch_size * traj_size, 1, 1]);

            //Compute testing dataset

            arrayfire::device_gc();
            arrayfire::sync(DEVICE);
            for batch_idx in 0..epoch_num {
                test_X = arrayfire::Array::new(&RSSI_TESTX[&batch_idx], test_X_dims);

                Y = arrayfire::Array::new(&RSSI_TESTY[&batch_idx], Y_dims);
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

            let save_filename = format!("info_{}_{}.csv", fold, input_size);
            raybnn::export::dataloader_f32::write_arr_to_csv(&save_filename, &info);

            let save_filename = format!("test_pred_{}_{}.csv", fold, input_size);
            raybnn::export::dataloader_f32::write_arr_to_csv(&save_filename, &Yhat_arr);

            let save_filename = format!("test_act_{}_{}.csv", fold, input_size);
            raybnn::export::dataloader_f32::write_arr_to_csv(&save_filename, &Y_arr);
        }
    }
}
