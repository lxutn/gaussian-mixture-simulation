from numpy import *
from numpy.linalg import *
from numpy.random import *
from scipy.linalg import sqrtm
from matplotlib.pyplot import plot, savefig, show

## plant
def plant(A,C,Q,R, Sigma0, Y, p, simLength):

    x=-888*ones((2,1, simLength))
    y=-888*ones(simLength)

    w = multivariate_normal([0,0], Q, (simLength,1)).T
    v = normal(0,R,simLength)

    s=zeros(simLength)
    zeta=rand(simLength)
    gamma= binomial(1,1-p, simLength)

    for i in range(simLength):

        if i==0:
            x[:,:,0] = multivariate_normal([0,0], Sigma0, 1).T 
        else:
            x[:,:,i]=A@x[:,:,i-1]+w[:,:,i-1]

        y[i]=C@x[:,:,i]+v[i]

        if zeta[i]<= exp(-1/2*y[i]*Y*y[i]):
            s[i]=0
        else:
            s[i]=1
    return x,y,s,gamma

## OLSET-KF estimator
def OLSETKF_estimator(A, C, Q, R, Sigma0, Y, simLength, gammaTimeS, gammaTimeSTimey):

    x_predict=-888*ones((2,1,simLength))
    P_predict=-888*ones((2,2,simLength))
    x_estimate=-888*ones((2,1,simLength))
    P_estimate=-888*ones((2,2,simLength))
    K=-888*ones((2,1,simLength))

    for i in range(simLength):
        
        if i==0:
            x_predict[:,:,i]=array([[0],[0]])
            P_predict[:,:,i]=Sigma0
        else:
            x_predict[:,:,i]=A@x_estimate[:,:,i-1]
            P_predict[:,:,i]=A@P_estimate[:,:,i-1]@A.T+Q
        
        K[:,:,i]=P_predict[:,:,i]@C.T/(C@P_predict[:,:,i]@C.T+R+(1-gammaTimeS[i])/Y)
        x_estimate[:,:,i]=(eye(2)-K[:,:,i]@C)@x_predict[:,:,i]+K[:,:,i]*gammaTimeSTimey[i]
        P_estimate[:,:,i]=(eye(2)-K[:,:,i]@C)@P_predict[:,:,i]
    
    return x_estimate


# Oracle estimator
def oracle_estimator(A, C, Q, R, Sigma0, Y, simLength, gamma, s, y):

    x_predict=-888*ones((2,1,simLength))
    P_predict=-888*ones((2,2,simLength))
    x_estimate=-888*ones((2,1,simLength))
    P_estimate=-888*ones((2,2,simLength))
    K=-888*ones((2,1,simLength))

    x_predict[:,:,0]= array([[0],[0]])
    P_predict[:,:,0]=Sigma0

    for i in range(simLength):

        if gamma[i]==0:
            x_estimate[:,:,i]=x_predict[:,:,i]
            P_estimate[:,:,i]=P_predict[:,:,i]
        else:
            K[:,:,i]=P_predict[:,:,i]@C.T/(C@P_predict[:,:,i]@C.T+R+(1-s[i])/Y)
            x_estimate[:,:,i]=(eye(2)-K[:,:,i]@C)@x_predict[:,:,i]+s[i]*K[:,:,i]*y[i]
            P_estimate[:,:,i]=(eye(2)-K[:,:,i]@C)@P_predict[:,:,i]

        if i+1<= simLength-1: 
            x_predict[:,:,i+1]=A@x_estimate[:,:,i]
            P_predict[:,:,i+1]=A@P_estimate[:,:,i]@A.T+Q

    return x_estimate

# MMSE estimator
def MMSE_estimator(A, C, Q, R, Sigma0, Y, p, simLength, gammaTimeS, gammaTimeSTimey):

    N=simLength+1

    m_kConkm1=-888*ones((2,1, simLength, 2**(N-1))) # at time k, 0 <= i <= 2^k-1 for prediction
    P_kConkm1=-888*ones((2,2, simLength, 2**(N-1)))

    m_kConk=-888*ones((2,1, simLength, 2**N))# at time k, 0 <= i <= 2^(k+1)-1 for prediction
    P_kConk=-888*ones((2,2, simLength, 2**N))

    alpha_kConkm1=-888*ones((simLength,2**N))
    alpha_kConk=-888*ones((simLength, 2**N))

    Pr_skgammakCon=-888*ones((simLength, 2**N))

    #S_k=-888*ones(2,2,simLength+1, 2**(N-1))
    #n_k=-888*ones(2,simLength+1,2**(N-1))

    OptimalEstimate=-888*ones((2,1,simLength))
    #OptimalError=-888*ones((2,2,simLength+1,2**N))


    # initialization for the hypothesis prediction
    m_kConkm1[:,:,0,0]=array([[0],[0]])
    P_kConkm1[:,:,0,0]=Sigma0

    for k in range(simLength):
            # hypothesis state update
            for i in range(2**(k+1)):
                if i<2**k:
                    m_kConk[:,:,k,i]=m_kConkm1[:,:,k,i]
                    P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i]
                else:
                    i_pre=i-2**k
                    K=P_kConkm1[:,:,k,i_pre]@C.T/(C@P_kConkm1[:,:,k,i_pre]@C.T+R+(1-gammaTimeS[k])/Y)
                    m_kConk[:,:,k,i]= (eye(2)-K@C)@m_kConkm1[:,:,k,i_pre]+K*gammaTimeSTimey[k]
                    P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i_pre]-K@C@P_kConkm1[:,:,k,i_pre]

                if k+1<=simLength-1:
                    m_kConkm1[:,:,k+1,i]=  A@m_kConk[:,:,k,i]
                    P_kConkm1[:,:,k+1,i]= A@P_kConk[:,:,k,i]@A.T+Q


            # hypothesis probability update: time update
            for i in range(2**(k+1)): 
                if k==0:
                    alpha_kConkm1[k,0]=p
                    alpha_kConkm1[k,1]=1-p
                else:
                    if i< 2**k:
                        alpha_kConkm1[k,i]= p*alpha_kConk[k-1,i]
                    else:
                        i_pre=i-2**k
                        alpha_kConkm1[k,i] =(1-p)*alpha_kConk[k-1,i_pre]

            # hypothesis probability update: measurement update
            for j in range(2**(k+1)):
                if j< 2**k:
                    Pr_skgammakCon[k,j]= 1-gammaTimeS[k]
                else:
                    j_pre=j-2**k 
                    Pr_skgammakCon[k,j] =gammaTimeS[k] + (1-2*gammaTimeS[k])/(sqrt(det((C@P_kConkm1[:,:,k,j_pre]@C.T+R)*Y+1)))* exp(-1/2*(C@m_kConkm1[:,:,k,j_pre]).T@C@m_kConkm1[:,:,k,j_pre]/(1/Y+(C@P_kConkm1[:,:,k,j_pre]@C.T+R)))
            
            tempsum=0
            for j in range(2**(k+1)):
                tempsum=tempsum+ Pr_skgammakCon[k,j]*alpha_kConkm1[k,j]

            for i in range(2**(k+1)):
                alpha_kConk[k,i]= (Pr_skgammakCon[k,i]*alpha_kConkm1[k,i])/(tempsum)
            
            # Final estimate
            OptimalEstimate[:,:,k]=0
            # OptimalError[:,:,k]=0

            for i in range(2**(k+1)):
                OptimalEstimate[:,:,k]= OptimalEstimate[:,:,k]+ alpha_kConk[k,i]*m_kConk[:,:,k,i]
            

            # for i in range(2**(k+1)-1):
            #     OptimalError[:,:,k]=OptimalError[:,:,k)+ alpha_kConk[k,i]*(P_kConk[:,:,k,i]+ (m_kConk[:,:,k,i]-OptimalEstimate[:,:,k])@(m_kConk[:,:,k,i]-OptimalEstimate[:,:,k]).T)

    return OptimalEstimate


# gpb estimator (finite-memory estimator)
def gpb_estimator(A, C, Q, R, Sigma0, Y, p, GPBHistoryLength, simLength, gammaTimeS, gammaTimeSTimey):

    N=GPBHistoryLength

    # when k\le N-1, the m_kConkm1 only requires 2**k to storage, which corresponding to x_{k|k-1}(\gamma_{k-1}, \ldots, \gamma_{0})
    # therefore, when k\le N-1, the maximal index for m_kConkm1 is 2**(N-1).
    # However, when k\ge N, the meaning of m_kConkm1 changes, which represents the time update from n_k. 
    # and should have the same number of index as m_kConk, which therefore requires 2**N storages.
    m_kConkm1=-999*ones((2,1, simLength, 2**N))
    P_kConkm1=-999*ones((2,2, simLength, 2**N))

    m_kConk=-999*ones((2,1, simLength, 2**N))
    P_kConk=-999*ones((2,2, simLength, 2**N))

    alpha_kConkm1=-999*ones((simLength,2**N))
    alpha_kConk=-999*ones((simLength, 2**N))

    Pr_skgammakCon=-999*ones((simLength, 2**N))

    # hypothesis mixing result
    n_k=-999*ones((2,1,simLength, 2**(N-1))) # represents hat{x}_k(\gamma_k, \ldots, \gamma_{k-N+2})
    S_k=-999*ones((2,2,simLength, 2**(N-1))) # represent P_k(\gamma_k, \ldots, \gamma_{k-N+2})
    beta_k=-999*ones((simLength, 2**N)) # represent Pr(\gamma_k, \ldots, \gamma_{k-N+2}|\mathcal{I}_k)

    GPBEstimate=-999*ones((2,1,simLength))
    # GPBError=-999*ones((2,2,simLength,2**(N)))

    m_kConkm1[:,:,0,0]=array([[0],[0]])
    P_kConkm1[:,:,0,0]=Sigma0

    for k in range(simLength):
        # initialization
        if k <= N-1:
            for i in range(2**(k+1)):
                #[\# particle state update: initialization]
               
                if i<2**k:
                    m_kConk[:,:,k,i]=m_kConkm1[:,:,k,i]
                    P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i]
                else:
                    i_pre=i-2**k
                    K=P_kConkm1[:,:,k, i_pre]@C.T/(C@P_kConkm1[:,:,k,i_pre]@C.T+R+(1-gammaTimeS[k])/Y)
                    m_kConk[:,:,k,i]= (eye(2)-K@C)@m_kConkm1[:,:,k,i_pre]+K*gammaTimeSTimey[k]
                    P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i_pre]-K@C@P_kConkm1[:,:,k,i_pre]
                
                if k+1<=N-1:
                    m_kConkm1[:,:,k+1,i]=  A@m_kConk[:,:,k,i]
                    P_kConkm1[:,:,k+1,i]= A@P_kConk[:,:,k,i]@A.T+Q

            # lossy probability update: time update
            for i in range(2**(k+1)): 
                if k==0:
                    alpha_kConkm1[k,0]=p
                    alpha_kConkm1[k,1]=1-p
                else:
                    if i< 2**k:
                        alpha_kConkm1[k,i]= p*alpha_kConk[k-1,i]
                    else:
                        i_pre=i-2**k
                        alpha_kConkm1[k,i] =(1-p)*alpha_kConk[k-1,i_pre]
            
            # lossy probability update: measurement update
            for j in range(2**(k+1)): 
                if j< 2**k:
                    Pr_skgammakCon[k,j]= 1-gammaTimeS[k]
                else:
                    j_pre=j-2**k
                    Pr_skgammakCon[k,j] =gammaTimeS[k] + (1-2*gammaTimeS[k])/(sqrt(det((C@P_kConkm1[:,:,k,j_pre]@C.T+R)*Y+1)))*exp(-1/2*(C@m_kConkm1[:,:,k,j_pre]).T/(1/Y+(C@P_kConkm1[:,:,k,j_pre]@C.T+R))@C@m_kConkm1[:,:,k,j_pre])
                    
            tempsum=0
            for j in range(2**(k+1)):
                tempsum=tempsum+ Pr_skgammakCon[k,j]*alpha_kConkm1[k,j]
            
            for i in range(2**(k+1)):
                alpha_kConk[k,i]= (Pr_skgammakCon[k,i]*alpha_kConkm1[k,i])/(tempsum)
            
            # Final estimate for k\le N-1

            GPBEstimate[:,:,k]=0
            # GPBError[:,:,k]=0
            
            for i in range(2**(k+1)):
                GPBEstimate[:,:,k]= GPBEstimate[:,:,k]+ alpha_kConk[k,i]*m_kConk[:,:,k,i]
            
            # for i in range(2**(k+1))
            #     GPBError(:,:,k)=GPBError(:,:,k)+ alpha_kConk[k,i]*(P_kConk[:,:,k,i]+ (m_kConk[:,:,k,i]-GPBEstimate(:,k))*(m_kConk[:,:,k,i]-GPBEstimate(:,k) )')

        # when k\ge N
        else: 
            # particle state update: mixing
            for i in range(2**(N-1)):
                i_twice=2*i 
                if (alpha_kConk[k-1,i_twice]+alpha_kConk[k-1,i_twice+1])==0:
                    n_k[:,:,k-1,i]=array([[0],[0]])
                    S_k[:,:,k-1,i]=zeros((2,2))
                else:
                    n_k[:,:,k-1,i]=(m_kConk[:,:,k-1,i_twice]*alpha_kConk[k-1,i_twice]+m_kConk[:,:,k-1,i_twice+1]*alpha_kConk[k-1,i_twice+1])/(alpha_kConk[k-1,i_twice]+alpha_kConk[k-1,i_twice+1])
                    S_k[:,:,k-1,i]= 1/(alpha_kConk[k-1,i_twice]+alpha_kConk[k-1,i_twice+1])*(alpha_kConk[k-1,i_twice]*(P_kConk[:,:,k-1,i_twice]+(m_kConk[:,:,k-1,i_twice]-n_k[:,:,k-1,i])@(m_kConk[:,:,k-1,i_twice]-n_k[:,:,k-1,i]).T)+ alpha_kConk[k-1,i_twice+1]*(P_kConk[:,:,k-1,i_twice+1]+(m_kConk[:,:,k-1,i_twice+1]-n_k[:,:,k-1,i])@(m_kConk[:,:,k-1,i_twice+1]-n_k[:,:,k-1,i]).T) )

            # particle state update: time and measurement update
            for i in range(2**N):
                if i<2**(N-1):
                    m_kConkm1[:,:,k,i]=  A@n_k[:,:,k-1,i]
                    P_kConkm1[:,:,k,i]= A@S_k[:,:,k-1,i]@A.T+Q
                    
                    m_kConk[:,:,k,i]=m_kConkm1[:,:,k,i] 
                    P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i] 
                else:
                    m_kConkm1[:,:,k,i]=  A@n_k[:,:,k-1,i-2**(N-1)]
                    P_kConkm1[:,:,k,i]= A@S_k[:,:,k-1,i-2**(N-1)]@A.T+Q
                    
                    K=P_kConkm1[:,:,k,i-2**(N-1)]@C.T/(C@P_kConkm1[:,:,k,i-2**(N-1)]@C.T+R+(1-gammaTimeS[k])/Y)
                    m_kConk[:,:,k,i]= (eye(2)-K@C)@m_kConkm1[:,:,k,i-2**(N-1)]+ K*gammaTimeSTimey[k]
                    P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i-2**(N-1)]-K@C@P_kConkm1[:,:,k,i-2**(N-1)]
                    
            # lossy probability: mixing
            for i in range(2**(N-1)):
                i_twice=2*i 
                beta_k[k-1,i]=alpha_kConk[k-1,i_twice]+alpha_kConk[k-1,i_twice+1]

            # lossy probability update: time and measurement update
            for i in range(2**N): 
                if i< 2**(N-1):
                    alpha_kConkm1[k,i]= p*beta_k[k-1,i]
                else: 
                    alpha_kConkm1[k,i] =(1-p)*beta_k[k-1,i-2**(N-1)]
                    
            for j in range(2**N):
                if j< 2**(N-1):
                    Pr_skgammakCon[k,j]= 1-gammaTimeS[k]
                else:
                    Pr_skgammakCon[k,j] =gammaTimeS[k] + (1-2*gammaTimeS[k])/(sqrt((C@P_kConkm1[:,:,k,j-2**(N-1)]@C.T+R)*Y+1))* exp(-1/2*(C@m_kConkm1[:,:,k,j-2**(N-1)]).T/(1/Y+(C@P_kConkm1[:,:,k,j-2**(N-1)]@C.T+R))@C@m_kConkm1[:,:,k,j-2**(N-1)])
                    
            tempsum=0
            for j in range(2**N):
                tempsum=tempsum+ Pr_skgammakCon[k,j]*alpha_kConkm1[k,j]
            
            for i in range(2**N):
                alpha_kConk[k,i]= (Pr_skgammakCon[k,i]*alpha_kConkm1[k,i])/(tempsum)
            
            # Final estimate for k\ge N
            
            GPBEstimate[:,:,k]=0
            #GPBError(:,:,k)=0
            
            for i in range(2**N):
                GPBEstimate[:,:,k]= GPBEstimate[:,:,k]+ alpha_kConk[k,i]*m_kConk[:,:,k,i]
            
            # for i=0:(2**N-1)
                
            #     k=k+1 i=i+1
            #     GPBError(:,:,k)=GPBError(:,:,k)+ alpha_kConk[k,i]*(P_kConk[:,:,k,i]+ (m_kConk[:,:,k,i]-GPBEstimate(:,k))*(m_kConk[:,:,k,i]-GPBEstimate(:,k) )')
            #     k=k-1 i=i-1
                
            # 
            # {\& Final estimate}
    return GPBEstimate


# particle filter with prior as importance density
def particle_filter(A, C, Q, R, Sigma0, Y, p, particleNumbers, simLength, gammaTimeS, gammaTimeSTimey):

    M=particleNumbers

    m_kConkm1=-999*ones((2,1, simLength, M))
    P_kConkm1=-999*ones((2,2, simLength, M))

    m_kConk=-999*ones((2,1, simLength, M))
    P_kConk=-999*ones((2,2, simLength, M))

    w_k=-999*ones((simLength, M))

    particle_gamma=-999*ones((simLength, M))
    Pr_skgammakCon=-999*ones((simLength, M))

    ParticleFilterEstimate=-999*ones((2,1,simLength))
    # ParticleFilterError=-999*ones((2,2,simLength))

    for k in range(simLength):
        
        for i in range(M):

            # generate new particle
            if gammaTimeS[k]==1:
                particle_gamma[k,i]=1
            else:
                particle_gamma[k,i]= binomial(1,1-p)
            
            if k==0:
                m_kConkm1[:,:,k,i]=array([[0],[0]])
                P_kConkm1[:,:,k,i]=Sigma0
            
            if particle_gamma[k,i]==0:
                Pr_skgammakCon[k,i]= 1-gammaTimeS[k]
            else:
                Pr_skgammakCon[k,i] =gammaTimeS[k] + (1-2*gammaTimeS[k])/(sqrt(det((C@P_kConkm1[:,:,k,i]@C.T+R)*Y+1)))* exp(-1/2*(C@m_kConkm1[:,:,k,i]).T/(1/Y+(C@P_kConkm1[:,:,k,i]@C.T+R))@C@m_kConkm1[:,:,k,i])
            
            
        # particle weights
        if k==0:
            tempsum=0
            for i in range(M):
                tempsum=tempsum+ Pr_skgammakCon[k,i]
            
            for i in range(M):
                w_k[k,i]= (Pr_skgammakCon[k,i])/(tempsum)
            
        else:
            tempsum=0
            for i in range(M):
                tempsum=tempsum+ Pr_skgammakCon[k,i]*w_k[k-1,i]
            
            for i in range(M):
                w_k[k,i]= (Pr_skgammakCon[k,i]*w_k[k-1,i])/(tempsum)
        
        # particle measurement update
        for i in range(M):
            if particle_gamma[k,i]==0:
                m_kConk[:,:,k,i]=m_kConkm1[:,:,k,i]
                P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i]
            else:
                K=P_kConkm1[:,:,k, i]@C.T/(C@P_kConkm1[:,:,k,i]@C.T+R+(1-gammaTimeS[k])/Y)
                m_kConk[:,:,k,i]= (eye(2)-K@C)@m_kConkm1[:,:,k,i]+K*gammaTimeSTimey[k]
                P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i]-K@C@P_kConkm1[:,:,k,i]
                
        # generate state estimate
        
        ParticleFilterEstimate[:,:,k]=0
        # ParticleFilterError[:,:,k]=0
        for i in range(M):
            ParticleFilterEstimate[:,:,k]= ParticleFilterEstimate[:,:,k]+ w_k[k,i]*m_kConk[:,:,k,i]
        
        # for i=1:M
        #     ParticleFilterError[:,:,k]=ParticleFilterError[:,:,k]+ w_k[k,i]*(P_kConk[:,:,k,i]+ (m_kConk[:,k,i]-ParticleFilterEstimate[:,k]]*(m_kConk[:,k,i]-ParticleFilterEstimate[:,k] )')
        
        # particle time update
        if k+1<= simLength-1:
            for i in range(M):
                m_kConkm1[:,:,k+1,i]=  A@m_kConk[:,:,k,i]
                P_kConkm1[:,:,k+1,i]= A@P_kConk[:,:,k,i]@A.T+Q

    return ParticleFilterEstimate
        



# particle filter with optimal importance density (recursively sampling from the posterior) 
def particle_filter_optimal(A, C, Q, R, Sigma0, Y, p, particleNumbers, simLength,  gammaTimeS, gammaTimeSTimey):
    # this is the particle filter with optimal importance density, i.e., the importance density selected as $Pr(\Gamma_k^i|\mathcal{I})$

    M=particleNumbers

    m_kConkm1=-999*ones((2,1, simLength, M))
    P_kConkm1=-999*ones((2,2, simLength, M))

    m_kConk=-999*ones((2,1, simLength, M))
    P_kConk=-999*ones((2,2, simLength, M))

    particle_gamma=-999*ones((simLength, M))

    Pr_skgammakCon=-999*ones((simLength, M))

    ParticleFilterEstimate=-999*ones((2,1,simLength))
    # ParticleFilterError=-999*ones((2,2,simLength))

    for k in range(simLength):
        
        for i in range(M):

            # particle time update
            if k==0:
                m_kConkm1[:,:,k,i]=array([[0],[0]])
                P_kConkm1[:,:,k,i]=Sigma0
            else:
                m_kConkm1[:,:,k,i]=  A@m_kConk[:,:,k-1,i]
                P_kConkm1[:,:,k,i]= A@P_kConk[:,:,k-1,i]@A.T+Q

            # generate new particle
            if gammaTimeS[k]==1:
                particle_gamma[k,i]=1
            else:
                Pr_skgammakCon[k,i] =(1-p)/(sqrt(det((C@P_kConkm1[:,:,k,i]@C.T+R)*Y+1)))* exp(-1/2*(C@m_kConkm1[:,:,k,i]).T/(1/Y+(C@P_kConkm1[:,:,k,i]@C.T+R))@C@m_kConkm1[:,:,k,i])
 
                particle_gamma[k,i]= binomial(1,Pr_skgammakCon[k,i]/(p+Pr_skgammakCon[k,i]))
            
            # measurement update for particles
            if particle_gamma[k,i]==0:
                m_kConk[:,:,k,i]=m_kConkm1[:,:,k,i]
                P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i]
            else:
                K=P_kConkm1[:,:,k, i]@C.T/(C@P_kConkm1[:,:,k,i]@C.T+R+(1-gammaTimeS[k])/Y)
                m_kConk[:,:,k,i]= (eye(2)-K@C)@m_kConkm1[:,:,k,i]+K*gammaTimeSTimey[k]
                P_kConk[:,:,k,i]=P_kConkm1[:,:,k,i]-K@C@P_kConkm1[:,:,k,i]
                
        # generate state estimate
        
        ParticleFilterEstimate[:,:,k]=0
        # ParticleFilterError[:,:,k]=0
        for i in range(M):
            ParticleFilterEstimate[:,:,k]= ParticleFilterEstimate[:,:,k]+ 1/M*m_kConk[:,:,k,i]
        
        # for i=1:M
        #     ParticleFilterError[:,:,k]=ParticleFilterError[:,:,k]+ w_k[k,i]*(P_kConk[:,:,k,i]+ (m_kConk[:,k,i]-ParticleFilterEstimate[:,k]]*(m_kConk[:,k,i]-ParticleFilterEstimate[:,k] )')
        
    return ParticleFilterEstimate
 


