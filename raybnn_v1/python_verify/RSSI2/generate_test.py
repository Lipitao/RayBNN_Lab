import numpy as np
from scipy.stats import norm


def main():

    input_csv = np.loadtxt("RSSI_test.dat", delimiter=',')
    print(input_csv.shape)

    in_size = input_csv.shape[1]
    database_size = input_csv.shape[0]

    total_traj_num = 5

    batch_size = 100
    traj_size = 20
    block_size = batch_size*traj_size
    RSSI_train = np.zeros((total_traj_num*block_size,in_size))

    for v in range(total_traj_num):

        for i in range(batch_size):

            idx = np.random.randint((database_size-1), size=1)[0]

            prev = input_csv[idx,0:2]

            RSSI_train[(batch_size*0)+ i + (v*block_size),:] = np.copy(input_csv[idx,:])
            for j in range(1,traj_size):
                mag = 10000
                p = 0.0
                rand = 1000000
                while ( p < rand ):
                    idx = np.random.randint((database_size-1), size=1)[0]
                    cur = input_csv[idx,0:2]
                    mag = np.linalg.norm(cur-prev)
                    p = 1-norm.cdf(x=mag-1.3, scale=0.5)
                    rand = np.random.uniform(0,1,1)[0]

                RSSI_train[(batch_size*j)+ i + (v*block_size),:] = np.copy(input_csv[idx,:])
                prev = cur




    meanY = np.loadtxt("meanY.dat", delimiter=',')
    stdY = np.loadtxt("stdY.dat", delimiter=',')


    RSSI_trainy =  ( RSSI_train[:,0:2]  - meanY ) / stdY
    print(RSSI_trainy)
    print(RSSI_trainy.shape)
    np.savetxt("Alcala_TESTY.dat", RSSI_trainy, delimiter=",", fmt='%.10f')



    RSSI_trainx = RSSI_train[:,2:]
    print(RSSI_trainx)
    print(RSSI_trainx.shape)
    np.savetxt("Alcala_TESTX.dat", RSSI_trainx, delimiter=",", fmt='%.10f')




if __name__ == '__main__':
    main()
