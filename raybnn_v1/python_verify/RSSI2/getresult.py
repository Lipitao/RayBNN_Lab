import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error


def main():

    fold = 10
    input_size_list = [6,8,12,14,16,32,50,75,100,125,162]


    input_size_list = np.asarray(input_size_list)

    meanY = np.loadtxt("meanY.dat", delimiter=',')
    stdY = np.loadtxt("stdY.dat", delimiter=',')


    RMSE_arr = np.zeros((input_size_list.shape[0],fold))
    MAE_arr = np.zeros((input_size_list.shape[0],fold))
    MAPE_arr = np.zeros((input_size_list.shape[0],fold))
    time_arr = np.zeros((input_size_list.shape[0],fold))
    param_arr = np.zeros((input_size_list.shape[0],fold))



    for i in range(input_size_list.shape[0]):
        input_size =  input_size_list[i]
        for j in range(fold):
            info_name = "info_" + str(j) + "_" + str(input_size) + ".dat"
            info = np.loadtxt(info_name, delimiter=',')

            time_arr[i,j]  = info[0]
            param_arr[i,j]  = info[1]




            pred_name = "test_pred_" + str(j) + "_" + str(input_size) + ".dat"
            pred = np.loadtxt(pred_name, delimiter=',')

            pred = ( pred*stdY )  + meanY

            act_name = "test_act_" + str(j) + "_" + str(input_size) + ".dat"
            act = np.loadtxt(act_name , delimiter=',')

            act = ( act*stdY )  + meanY

            RMSE = mean_squared_error(act, pred, squared=False)
            RMSE_arr[i,j]  = RMSE

            MAE = mean_absolute_error(act, pred)
            MAE_arr[i,j]  = MAE


            MAPE = mean_absolute_percentage_error(act, pred)
            MAPE_arr[i,j]  = MAPE






    np.savetxt("MAE.csv", MAE_arr , delimiter=",", fmt='%.10f')
    np.savetxt("RMSE.csv", RMSE_arr , delimiter=",", fmt='%.10f')
    np.savetxt("MAPE.csv", MAPE_arr , delimiter=",", fmt='%.10f')
    np.savetxt("time.csv", time_arr , delimiter=",", fmt='%.10f')
    np.savetxt("param.csv", param_arr , delimiter=",", fmt='%.10f')



    MAE_mean = np.mean(MAE_arr,axis=1)
    MAE_std = np.std(MAE_arr,axis=1)

    plt.errorbar(input_size_list, MAE_mean, MAE_std)
    plt.xlabel("Number of APs")
    plt.ylabel("MAE (m)")
    #plt.xscale("log")
    plt.show()


    time_mean = np.mean(time_arr,axis=1)
    time_mean = np.add.accumulate(time_mean)
    time_std = np.std(time_arr,axis=1)

    plt.errorbar(input_size_list, time_mean, time_std)
    plt.xlabel("Number of APs")
    plt.ylabel("Cumulative Training Time (s)")
    plt.show()

































if __name__ == '__main__':
    main()
