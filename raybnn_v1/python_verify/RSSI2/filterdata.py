import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 



def main():

    Alcala_TRAINX = np.loadtxt("Alcala_TRAINX.dat", delimiter=',')
    
    a = Alcala_TRAINX

    a[a > 5.0] = 5.0
    a[a < -5.0] = -5.0

    
    np.savetxt("Alcala_TRAINX.dat", a, delimiter=",", fmt='%.10f')









    

















if __name__ == '__main__':
    main()
