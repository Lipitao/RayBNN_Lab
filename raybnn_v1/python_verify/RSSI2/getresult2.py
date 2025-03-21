import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import glob




def main():

    batches_num = 5


    a = np.array([0,1,2,3,4,5,6,7])
    print(a)
    a = a.reshape(4,2)
    print(a)


    meanY = np.loadtxt("meanY.dat", delimiter=',')
    stdY = np.loadtxt("stdY.dat", delimiter=',')




    RMSE_list = []
    MAE_list = []
    con_list = []
    neuron_list = []


    model_list = []
    model_list2 = []
    max_train_num = 0
    for filename in glob.glob('*.yhat'):

        stridx = filename.find("batch_idx_")


        print(filename)
        substr = filename[:stridx]
        print(substr)

        if (substr in model_list2) == False:
            model_list2.append(substr)

            strlist =  substr.split("_")
            if int(strlist[11]) > max_train_num:
                max_train_num = int(strlist[11])


    for i in range(max_train_num):
        for modelname in model_list2:
            print(modelname)
            strlist =  modelname.split("_")
            if int(strlist[11])  == i:
                model_list.append(modelname)
                break

    #print(model_list)
    #exit()

    for modelname in model_list:
        act = np.loadtxt("Alcala_TESTY.dat", delimiter=',')

        print(modelname)

        strlist =  modelname.split("_")
        print(strlist)

        pred = []
        for i in range(batches_num):
            filename = modelname + "batch_idx_" + str(i) + ".yhat"
            print(filename)

            predcur = np.loadtxt(filename, delimiter=',')

            predcur = predcur.reshape(int(predcur.shape[0]/2),2)

            if i == 0:
                pred = np.copy(predcur)
            else:
                pred = np.concatenate((pred, predcur), axis=0)
            #print(pred)
            #print(pred.shape)

        pred = ( pred*stdY )  + meanY
        act = ( act*stdY )  + meanY

        RMSE = mean_squared_error(act, pred, squared=False)

        MAE = mean_absolute_error(act, pred)

        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        con_list.append(int(strlist[8]))
        neuron_list.append(int(strlist[2]))

        print(MAE)

    print("min")
    print(np.min(MAE_list))


    np.savetxt("MAE.csv", MAE_list , delimiter=",", fmt='%.10f')
    np.savetxt("RMSE.csv", RMSE_list , delimiter=",", fmt='%.10f')

    np.savetxt("neuron_list.csv", neuron_list , delimiter=",", fmt='%.10f')
    np.savetxt("con_list.csv", con_list , delimiter=",", fmt='%.10f')
    np.savetxt("model_list.csv", model_list , delimiter=",", fmt='%s')



    plt.plot(neuron_list,MAE_list)
    plt.ylim([0.6, 4.0])
    plt.xlabel("Number of Neurons")
    plt.ylabel("Testing MAE (m)")
    plt.show()


    plt.plot(con_list,MAE_list)
    plt.ylim([0.6, 4.0])
    plt.xlabel("Number of Neural Connections")
    plt.ylabel("Testing MAE (m)")
    plt.show()









































if __name__ == '__main__':
    main()
