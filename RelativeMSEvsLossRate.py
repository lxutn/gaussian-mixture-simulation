# compare the average RMSE of different estimators
# OLSETKF, MMSE estimator, finite-memory estimator, particle filter
# with respect to the oracle_estimator 

from numpy import *
from numpy.linalg import *
from numpy.random import *
from scipy.linalg import sqrtm
from scipy.io import savemat
from matplotlib.pyplot import plot, savefig, show, legend
from estimators import plant, OLSETKF_estimator, oracle_estimator, MMSE_estimator, gpb_estimator, particle_filter, particle_filter_optimal

simLength=10

A=diag([0.8,0.95],0)
C=array([[1,1]])
Q=diag([1,1],0)
R=1
Sigma0=diag([1,1],0)

Y=1

GPBHistoryLength=2
particleNumbers=20

pNumbers=11
rounds=1500

EmpiricalRelativeMSE=zeros((pNumbers,4))

for pindex in range(pNumbers):

    p=1/(pNumbers-1)*pindex

    for round in range(rounds):

        print(p, round, sep=' : ')

        [x,y,s,gamma]=plant(A,C,Q,R,Sigma0, Y, p, simLength)

        oracle_estimate=oracle_estimator(A, C, Q, R, Sigma0, Y, simLength, gamma, s, y)
        OLSETKF_estimate=OLSETKF_estimator(A, C, Q, R, Sigma0, Y, simLength, gamma*s, gamma*s*y)
        MMSE_estimate=MMSE_estimator(A, C, Q, R, Sigma0, Y, p, simLength, gamma*s, gamma*s*y)
        GPB_estimate=gpb_estimator(A, C, Q, R, Sigma0, Y, p, GPBHistoryLength, simLength,  gamma*s, gamma*s*y)
        PF_estimate=particle_filter_optimal(A, C, Q, R, Sigma0, Y, p, particleNumbers, simLength, gamma*s, gamma*s*y)

        EmpiricalRelativeMSE[pindex,0]= EmpiricalRelativeMSE[pindex,0]+ sum(norm(OLSETKF_estimate-x,2,0))/sum(norm(oracle_estimate-x,2,0))
        EmpiricalRelativeMSE[pindex,1]= EmpiricalRelativeMSE[pindex,1]+ sum(norm(MMSE_estimate-x,2,0))/sum(norm(oracle_estimate-x,2,0))
        EmpiricalRelativeMSE[pindex,2]= EmpiricalRelativeMSE[pindex,2]+ sum(norm(GPB_estimate-x,2,0))/sum(norm(oracle_estimate-x,2,0))
        EmpiricalRelativeMSE[pindex,3]= EmpiricalRelativeMSE[pindex,3]+ sum(norm(PF_estimate-x,2,0))/sum(norm(oracle_estimate-x,2,0))

    EmpiricalRelativeMSE[pindex,:]=EmpiricalRelativeMSE[pindex,:]/rounds

## save and plot

savemat('EmpiricalRelativeMSEdata.mat', mdict={'EmpiricalRelativeMSE': EmpiricalRelativeMSE})

for i in range(4):
    plot(arange(0,1.00001,1/(pNumbers-1)), EmpiricalRelativeMSE[:,i])
legend(("OLSETKF","MMSE","GPB","PF"))
show()






