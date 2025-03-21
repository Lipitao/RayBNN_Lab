import glob
import os

import numpy as np


max_num = []

for name in glob.glob('./*'): 
    filepath = os.path.abspath(name)
    print(filepath)
    if "_info_" in filepath:
        num = filepath.split("_")[-1]
        num = num.replace(".txt","")
        num = int(num)
        print(num)
        max_num.append(num)

max_num = np.array(max_num)
max_num = np.max(max_num) + 1


metrics = np.zeros((max_num,6))

for name in glob.glob('./*'): 
    filepath = os.path.abspath(name)
    print(filepath)
    if "_info_" in filepath:

        num = filepath.split("_")[-1]
        num = num.replace(".txt","")
        num = int(num)


        data = np.loadtxt(filepath, delimiter=",")
        print(data)
        max_components = int(np.max(data, axis = 0)[0])
        avg = np.zeros((max_components, data.shape[1] ))
        for i in range(data.shape[0]):
            sample = data[i,:]
            idx = int(sample[0]) - 1
            avg[idx,:] = avg[idx,:] + sample
        avg = avg/avg[0,0]


        #acc[num] = np.max(avg[:,1])

        idx = np.argmax(avg[:,1])
        #n_comp[num] = avg[idx,0]
        metrics[num,:] = avg[idx, :]

meanz = np.mean(metrics, axis=0)
stdz = np.std(metrics, axis=0)

print("mean")
print(np.mean(metrics, axis=0))
print("std")
print(np.std(metrics, axis=0))


print(metrics[:,0])

np.savetxt("./metrics.txt", metrics, delimiter=',') 
np.savetxt("./acc_CSP_LDA.txt", metrics[:,1:], delimiter=',') 

for i in range(meanz.shape[0]):
    print(str(meanz[i])[:6] + " +/- " + str(stdz[i])[:6])
