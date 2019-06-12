# compare the execution time for the optimal estimator

from numpy import *
from numpy.linalg import *
from numpy.random import *
from scipy.linalg import sqrtm
from scipy.io import savemat
from matplotlib.pyplot import plot, savefig, show, legend
from estimators import plant, OLSETKF_estimator, oracle_estimator, MMSE_estimator, gpb_estimator, particle_filter, particle_filter_optimal
import time

A=diag([0.8,0.95],0)
C=array([[1,1]])
Q=diag([1,1],0)
R=1
Sigma0=diag([1,1],0)

Y=1
p=0.5

GPBHistoryLength=2
particleNumbers=20

[x,y,s,gamma]=plant(A,C,Q,R,Sigma0, Y, p, 20)

EstimatorRunTime=zeros((50,3))

for simLength in range(10,16):

    start_time = time.clock()
    MMSE_estimate=MMSE_estimator(A, C, Q, R, Sigma0, Y, p, simLength, gamma*s, gamma*s*y)
    EstimatorRunTime[simLength,0]=time.clock() - start_time

    start_time = time.clock()
    GPB_estimate=gpb_estimator(A, C, Q, R, Sigma0, Y, p, GPBHistoryLength, simLength,  gamma*s, gamma*s*y)
    EstimatorRunTime[simLength,1]=time.clock() - start_time

    start_time = time.clock()
    PF_estimate=particle_filter_optimal(A, C, Q, R, Sigma0, Y, p, particleNumbers, simLength, gamma*s, gamma*s*y)
    EstimatorRunTime[simLength,2]=time.clock() - start_time

## save and plot

savemat('EstimatorRunTime.mat', mdict={'EstimatorRunTime': EstimatorRunTime})

for i in range(3):
    plot(range(10,16), EstimatorRunTime[range(10,16),i])

legend(("MMSE","GPB","PF"))
show()



