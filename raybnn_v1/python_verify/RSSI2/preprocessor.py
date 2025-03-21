import numpy as np
import pandas

from numpy import linalg as LA


def computeNormalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)


    std[std <= 0.3] = 1.0

    return [mean,std]





def runNormalize(X,mean,std):
    V = (X-mean )/  std


    return V







def main():

    input_size = 162

    input_csv = pandas.read_csv("1490779198_4046512_alc2017_training_set.dat")
    input_csv = input_csv.values.astype(np.float32)

    print(input_csv.shape)

    input_x = input_csv[:,:input_size]
    output_y = input_csv[:,input_size:]



    AP_samples = []

    meanarr = []
    stdarr = []
    for i in range(input_size):
        col = input_x[:,i]
        col = col[ col != 100.0 ]

        AP_samples.append(col.shape[0])
        

        if col.shape[0] == 0:
            col = np.array([-100,-100])
        

        mean = np.mean(col)
        std = np.std(col)

        if std < 0.3:
            std = 1.0
        
        meanarr.append(mean)
        stdarr.append(std)


    meanarr = np.asarray(meanarr)
    stdarr = np.asarray(stdarr)
    AP_samples = np.asarray(AP_samples)

    np.savetxt("AP_samples.dat", AP_samples, delimiter=",", fmt='%.10f')



    input_x[ input_x == 100.0 ]  = -100.0


    input_x = (input_x - meanarr)/stdarr








    savedataset = np.concatenate((output_y, input_x),axis=1)


    test = []
    test_y = []

    train = []
    train_y = []

    for i in range(output_y.shape[0]):

        in_test = False
        for j in range(len(test_y)):
            if LA.norm( output_y[i,:] - test_y[j]) == 0.0:
                in_test = True
                break

        in_train = False
        for j in range(len(train_y)):
            if LA.norm( output_y[i,:] - train_y[j]) == 0.0:
                in_train = True
                break
        
        if (in_test):
            test.append(savedataset[i,:])
        elif (in_train):
            train.append(savedataset[i,:])
        else:
            rand = np.random.uniform(0,1,1)[0]
            if (rand < 0.3):
                test.append(savedataset[i,:])
                test_y.append(output_y[i,:])
            else:
                train.append(savedataset[i,:])
                train_y.append(output_y[i,:])

    train = np.asarray(train)
    test = np.asarray(test)
    print("train")
    print(train.shape)
    print("test")
    print(test.shape)

    np.savetxt("RSSI_train.dat", train, delimiter=",", fmt='%.10f')
    np.savetxt("RSSI_test.dat", test, delimiter=",", fmt='%.10f')



    [meanarr,stdarr] = computeNormalize(output_y)
    #output_y = runNormalize(output_y,meanarr,stdarr)


    np.savetxt("meanY.dat", meanarr, delimiter=",", fmt='%.10f')
    np.savetxt("stdY.dat", stdarr, delimiter=",", fmt='%.10f')














    

















if __name__ == '__main__':
    main()
