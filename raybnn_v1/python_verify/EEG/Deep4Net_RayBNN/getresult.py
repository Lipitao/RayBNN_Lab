import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import glob


def main():

    data = np.zeros((54,5))
    for filename in glob.glob('./test_pred*'):

        num = int(filename.split("_")[2])
        pred = np.loadtxt(filename, delimiter=',')
        filename = filename.replace("test_pred","test_act")
        act = np.loadtxt(filename, delimiter=',')

        meanz = np.mean(pred)
        idx_pred = np.heaviside(pred-meanz,0.5)
        idx_act = act

        acc = accuracy_score(idx_act.astype(int), idx_pred.astype(int))
        p, r, f1, _ = precision_recall_fscore_support(idx_act.astype(int), idx_pred.astype(int), average='macro')
        roc = roc_auc_score(idx_act, pred)

        print(filename)
        print(accuracy_score(idx_act, idx_pred))
        #print(precision_recall_fscore_support(idx_act, idx_pred, average='macro') )
        #print(roc_auc_score(idx_act, pred))

        data[num] = np.array([acc, p, r, f1, roc])

    data = np.array(data)

    meanz = np.mean(data, axis=0)
    stdz = np.std(data, axis=0)

    #print(data)

    for i in range(meanz.shape[0]):
        print(str(meanz[i])[:6] + " +/- " + str(stdz[i])[:6])


    np.savetxt('acc_deep4net_raybnn.txt', data, delimiter=',')



















if __name__ == '__main__':
    main()
