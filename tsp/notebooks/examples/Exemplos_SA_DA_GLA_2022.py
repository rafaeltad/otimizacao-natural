################################################################################################################################################################

# SA with logbook and global optimum

################################################################################################################################################################

# 20200713 gabriel@pads.ufrj.br

# (conda install scipy)

import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d

# Data

np.random.seed(1); P=100; NC=8;
cluster_centers=np.random.normal(0,1,[2,NC])

aux=np.zeros([2,1]); aux[0]=cluster_centers[0,0]; aux[1]=cluster_centers[1,0]
data_vectors=0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))
for k in range(1,NC):
    aux=np.zeros([2,1]); aux[0]=cluster_centers[0,k]; aux[1]=cluster_centers[1,k]
    data_vectors=np.concatenate((data_vectors,0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))),axis=1)

# Given Cost Function

def J(X,Y):
    # data_vectors go into X. 
    # codebook, which is under optimization, goes into Y.
    D=0; M,N=np.shape(X); M,K=np.shape(Y)
    aux=np.zeros([2,1]); p=np.zeros([K,1]);
    for n in range(0,N):
        aux[0]=X[0,n]; aux[1]=X[1,n];
        d=np.sum(np.power(np.tile(aux,(1,K))-Y,2),axis=0)
        # k=np.where(d==np.min(d))
        # p[k[0]]+=1
        D+=np.min(d)
    Dout=D/N
    # if np.min(p)==0: Dout=100
    return Dout

# Definitions                                Let empty cells, X~=Xmin        Let empty cells, X=Xmin             No empty cell, X=Xmin               No empty cell, X~=Xmin              No empty cell, X=Xmin, Cauchy

# N=int(1e4); K=8;   T0=5e-3; epsilon=0.05 # (0.06943) 20200715 03:01 05:50; (0.06877) 20200717 02:34 05:31;     (0.02168) 20200716 04:13 09:19; (!) (0.02234) 20200716 12:51 17:23; (!) 
# N=int(1e3); K=8;   T0=5e-3; epsilon=0.05 # (0.06939) 20200715 01:32 01:49; (0.07002) 20200718 16:57 17:22;     (0.06942) 20200718 17:28 18:14;     (0.06996) 20200718 18:24 19:09;
# N=int(1e2); K=4;   T0=0.5;  epsilon=0.1  # (0.56384) 20200718 19:13 19:15; (0.30380) 20200718 19:17 19:19;     (0.37816) 20200718 19:21 19:23;     (0.64658) 20200718 19:26 19:28;     (0.30756) 20200720 01:53 01:55
# N=int(1e2); K=16;  T0=0.5;  epsilon=0.1  # (0.38816) 20200723 01:36 01:40;                                                                                                             (0.09264) 20200720 01:59 02:06
# N=int(1e2); K=32;  T0=0.5;  epsilon=0.1  # (0.32721) 20200723 01:25 01:33; (0.11954) 20200721 01:51 02:00;                                                                             (0.07123) 20200720 02:16 02:30; (*)
N=int(1e2);   K=64;  T0=0.5;  epsilon=0.1  # (0.32721) 20200721 00:50 01:08; (0.09565) 20200721 01:19 01:38;                                                                             (0.03852) 20200720 02:36 03:04; (*)
# N=int(1e3); K=2;   T0=0.2;  epsilon=0.05 # (0.19230) 20200716 00:59 01:03; (0.25565) 20200716 01:41 01:45;     (0.28089) 20200716 01:47 01:54;     (0.31044) 20200718 19:33 19:45;
# N=int(1e3); K=4;   T0=0.5;  epsilon=0.1  # (0.56046) 20200723 02:01 02:14; 
# N=int(1e3); K=4;   T0=0.1;  epsilon=0.1  # (0.16800) 20200716 02:40 02:48; (0.12055) 20200716 02:50 02:58;     (0.10352) 20200716 03:56 04:
# N=int(1e3); K=16;  T0=0.1;  epsilon=0.1  # (0.11934) 20200718 01:10 01:40; (0.03077) 20200718 02:22 02:53 (!); (0.07149) 20200218 02:57 04:01
# N=int(1e3); K=20;  T0=0.1;  epsilon=0.2  # (0.13637) 20200718 22:54 23:40; (0.13769) 20200719 00:08 00:55;     (0.05781) 20200719 01:09 02:39 (*); (0.06894) 20200719 02:55 04:27; (*)
# N=int(1e3); K=20;  T0=0.2;  epsilon=0.5  # (0.52746) 20200723 02:18 xx:xx;                                                                         (0.18500) 20200719 04:32 06:00;
# N=int(1e3); K=20;  T0=0.1;  epsilon=0.5  # (0.31332) 20200715 14:44 15:24; (0.30313) 20200715 16:54 17:35
# N=int(1e3); K=20;  T0=0.2;  epsilon=0.2  # (0.28410)
# N=int(1e3); K=20;  T0=5e-3; epsilon=0.2  # (0.07716) 20200715 11:27 12:11; 
# N=int(1e3); K=20;  T0=1;    epsilon=0.2  # (0.44194) 20200715 12:14 12:56;
# N=int(1e3); K=10;  T0=0.2;  epsilon=0.2  # (0.35083) 20200715 12:58 13:19; (0.15461) 20200715 14:06 14:26
# N=int(1e2); K=40;  T0=1;    epsilon=0.5; # (0.52746) 20200717 00:19 00:29; (0.22927) 20200715 17:44 17:53
# N=int(1e2); K=40;  T0=1;    epsilon=1;   # (0.79857) 20200717 00:30 00:40; (0.51304) 20200715 18:07 18:17 
# N=int(1e2); K=40;  T0=1;    epsilon=0.1; # (0.39133) 20200717 23:17 23:25; (0.12413) 20200715 18:22 18:30
# N=int(1e2); K=80;  T0=1;    epsilon=0.1; # (0.39133) 20200717 23:28 23:44; (0.08683) 20200715 18:53 19:10;     (0.09784) 20200715 23:45 00:14;    
# N=int(1e2); K=160; T0=1;    epsilon=0.1; # (0.39133) 20200717 00:42 01:20; (0.07638) 20200715 20:47 21:24;     (0.08162) 20200718 19:54 21:00;     (0.18537) 20200718 21:34 22:45;
# N=int(1e2); K=80;  T0=1;    epsilon=0.2; # (0.44194) 20200717 01:28 01:48; (0.10799) 20200715 21:49 22:05;     (0.08247) 20200716 00:21 00:49; (*) (0.22944) 20200717 01:51 02:25; ( ) (x.error) 20200720 01:49 ha:lt
# N=int(1e2); K=320; T0=1;    epsilon=0.1; # (0.39133) 20200718 00:01 01:03; (0.07638) 20200715 22:14 23:20;     (0.07625) 20200718 04:14 06:29;     (0.18537) 20200717 20:46 22:31;     (0.03161) 20200720 21:40 00:31 (*)
# N=int(2e2); K=1;   T0=10;   epsilon=1;   # (0.79857) 20200717 00:17 00:18;
# N=int(1e2); K=2; T=1e-2; epsilon=0.1

 # Remarks

# (!) Global minimum
# (*) Corret number of clusters - almost at global minimum - why has it not been reached?

# Initialize Main Loop

np.random.seed(0); X=np.random.normal(0,1,[2,NC])
fim=0; n=0; k=0; JX=J(data_vectors,X); Jmin=JX; Xmin=X; T=T0;
history_J=np.zeros([int(N*K),1]); history_T=np.zeros([int(N*K),1])

# Main Loop

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))

while not(fim):
    T=T0/np.log2(2+k)
    # T=T0/(k+1)
    for n in range(0,N):
        Xhat=X+epsilon*np.random.normal(0,1,np.shape(X))
        # Xhat=X+epsilon*np.random.standard_cauchy(np.shape(X))
        JXhat=J(data_vectors,Xhat) 
        if np.random.uniform()<np.exp((JX-JXhat)/T):
            X=Xhat; JX=JXhat
            if JX<Jmin:
                Jmin=JX; Xmin=X;
        history_J[k*N+n]=JX
        history_T[k*N+n]=T
        if np.remainder(n+1,100)==0:
            print([k,n+1,Jmin])
    k+=1
    # X=Xmin
    if k==K: fim=1

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))      
        
print(Jmin)
print(J(data_vectors,cluster_centers)) # 0.01987 (global minimum)
print(Xmin.T)

# Results

plt.rc('font',size=16,weight='bold')

plt.figure()
plt.subplot(211)
plt.plot(history_J)
plt.ylim(0,0.5)
plt.yticks(np.arange(0,0.5,0.1))
plt.grid()
plt.subplot(212)
plt.plot(history_T)
plt.grid()

vor=Voronoi(Xmin.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Xmin[0,:],Xmin[1,:],'r.',markersize=20)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.grid()
plt.show() 

################################################################################################################################################################

# DA with T0 = 10

################################################################################################################################################################

# 20200726 gabriel@pads.ufrj.br

# (conda install scipy)

import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d

# Data

np.random.seed(1); P=100; NC=8;
cluster_centers=np.random.normal(0,1,[2,NC])

aux=np.zeros([2,1]); aux[0]=cluster_centers[0,0]; aux[1]=cluster_centers[1,0]
data_vectors=0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))
for k in range(1,NC):
    aux=np.zeros([2,1]); aux[0]=cluster_centers[0,k]; aux[1]=cluster_centers[1,k]
    data_vectors=np.concatenate((data_vectors,0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))),axis=1)
    
# Main Loop

X=data_vectors; M,N=np.shape(X); K=NC; Y=np.random.normal(0,1,[M,K]); 
T=10; alpha=0.9; i=0; fim=0; epsilon=1e-6; delta=1e-3;
d=np.zeros([K,N])
p_ygivenx=np.zeros([K,N])
I=200
J=np.zeros(I)
D=np.zeros(I)
LocalT=np.zeros(I)
 
while not(fim):

    # Partition Condition
    for n in range(0,N):
        for k in range(0,K):
            d[k,n]=np.sum(np.power(X[:,n]-Y[:,k],2))
            p_ygivenx[k,n]=np.exp(-d[k,n]/T)
    Zx=np.sum(p_ygivenx,axis=0)   
    p_ygivenx=p_ygivenx/np.tile(Zx,(K,1))

    # Centroid Condition
    Y=np.zeros([M,K])
    for k in range(0,K):
        y=np.zeros(M)
        w=0
        for n in range(0,N):
            y+=p_ygivenx[k,n]*X[:,n]
            w+=p_ygivenx[k,n]
        Y[:,k]=y/w
        
    # Cost Function and Loop Control
    J[i]=-T/N*np.sum(np.log(Zx))
    D[i]=np.mean(np.sum(p_ygivenx*d,axis=0))
    LocalT[i]=T
    if i==34: Y34=Y # A few codebook examples at critical temperatures
    if i==45: Y45=Y
    if i==67: Y67=Y
    if i==90: Y90=Y
    if (i>0):
        if abs(J[i]-J[i-1])/abs(J[i-1])<delta:
            T=alpha*T
            Y=Y+epsilon*np.random.normal(0,1,np.shape(Y))
    print([i,J[i],D[i],LocalT[i]])   
    i+=1
    if (T<0.1)or(i==I): fim=1

plt.rc('font',size=16,weight='bold')

plt.figure()
plt.plot(-J,'r-',D,'k-',LocalT,'b-')
plt.ylim(0,5)
plt.grid()

plt.figure()
plt.plot(-J,'r-',D,'k-',LocalT,'b-')
plt.ylim(0,3)
plt.xlim(20,185)
plt.grid()

plt.figure()
plt.plot(-J,'r.-',D,'k.-',LocalT,'b.-')
plt.ylim(0.2,3)
plt.xlim(20,100)
plt.grid()

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y34[0,:],Y34[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y45[0,:],Y45[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y67[0,:],Y67[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y90[0,:],Y90[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y[0,:],Y[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 

################################################################################################################################################################

# GLA

################################################################################################################################################################

# 20200730 gabriel@pads.ufrj.br

# (conda install scipy)

import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d

# Data

np.random.seed(1); P=100; NC=8;
cluster_centers=np.random.normal(0,1,[2,NC])

aux=np.zeros([2,1]); aux[0]=cluster_centers[0,0]; aux[1]=cluster_centers[1,0]
data_vectors=0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))
for k in range(1,NC):
    aux=np.zeros([2,1]); aux[0]=cluster_centers[0,k]; aux[1]=cluster_centers[1,k]
    data_vectors=np.concatenate((data_vectors,0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))),axis=1)
    
# Main Loop

X=data_vectors; M,N=np.shape(X); K=NC; np.random.seed(5); Y=np.random.normal(0,1,[M,K]); 
i=0; fim=0; delta=1e-3; I=10; D=np.zeros(I); epsilon=1e-3;
 
while not(fim):

    # Partition Condition
    aux=np.zeros([2,1]); p=np.zeros([K,1]); Ynext=np.zeros([M,K])
    for n in range(0,N):
        aux[0]=X[0,n]; aux[1]=X[1,n];
        d=np.sum(np.power(np.tile(aux,(1,K))-Y,2),axis=0)
        k=np.where(d==np.min(d))
        Ynext[0,k[0]]+=aux[0];
        Ynext[1,k[0]]+=aux[1];
        p[k[0]]+=1
        D[i]+=np.min(d)
        
    D[i]=D[i]/N
        
    # Centroid Condition
    for k in range(0,K):
        if p[k]>0:
            Y[0,k]=Ynext[0,k]/p[k]
            Y[1,k]=Ynext[1,k]/p[k]
        else:
            m=np.where(p==np.max(p))
            Y[0,k]=Y[0,m[0]]+epsilon*np.random.normal(0,1)
            Y[1,k]=Y[1,m[0]]+epsilon*np.random.normal(0,1)
    
    # Loop Control
    if (i>0):
        if (abs(D[i]-D[i-1])/D[i-1])<delta:
            fim=1
    
    print([i,D[i]])
    i+=1

plt.rc('font',size=16,weight='bold')

plt.figure()
plt.plot(D,'k.-')
plt.xlim(0,5)
plt.grid()

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y[0,:],Y[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 

################################################################################################################################################################

# DA with T0 = 0.05

################################################################################################################################################################

# 20200730 gabriel@pads.ufrj.br

# (conda install scipy)

import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d

# Data

np.random.seed(1); P=100; NC=8;
cluster_centers=np.random.normal(0,1,[2,NC])

aux=np.zeros([2,1]); aux[0]=cluster_centers[0,0]; aux[1]=cluster_centers[1,0]
data_vectors=0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))
for k in range(1,NC):
    aux=np.zeros([2,1]); aux[0]=cluster_centers[0,k]; aux[1]=cluster_centers[1,k]
    data_vectors=np.concatenate((data_vectors,0.1*np.random.normal(0,1,[2,P])+np.tile(aux,(1,P))),axis=1)
    
# Main Loop

X=data_vectors; M,N=np.shape(X); K=NC; Y=np.random.normal(0,1,[M,K]); 
T=0.05; alpha=0.9; i=0; fim=0; epsilon=1e-6; delta=1e-3;
d=np.zeros([K,N])
p_ygivenx=np.zeros([K,N])
I=200
J=np.zeros(I)
D=np.zeros(I)
LocalT=np.zeros(I)
 
while not(fim):

    # Partition Condition
    for n in range(0,N):
        for k in range(0,K):
            d[k,n]=np.sum(np.power(X[:,n]-Y[:,k],2))
            p_ygivenx[k,n]=np.exp(-d[k,n]/T)
    Zx=np.sum(p_ygivenx,axis=0)   
    p_ygivenx=p_ygivenx/np.tile(Zx,(K,1))

    # Centroid Condition
    Y=np.zeros([M,K])
    for k in range(0,K):
        y=np.zeros(M)
        w=0
        for n in range(0,N):
            y+=p_ygivenx[k,n]*X[:,n]
            w+=p_ygivenx[k,n]
        Y[:,k]=y/w
        
    # Cost Function and Loop Control
    J[i]=-T/N*np.sum(np.log(Zx))
    D[i]=np.mean(np.sum(p_ygivenx*d,axis=0))
    LocalT[i]=T
    if (i>0):
        if abs(J[i]-J[i-1])/abs(J[i-1])<delta:
            T=alpha*T
            Y=Y+epsilon*np.random.normal(0,1,np.shape(Y))
    print([i,J[i],D[i],LocalT[i]])   
    i+=1
    if (T<0.005)or(i==I): fim=1

plt.rc('font',size=16,weight='bold')

plt.figure()
plt.plot(J,'r-',D,'k-',LocalT,'b-')
plt.ylim(0,0.2)
plt.xlim(0,56)
plt.grid()

vor=Voronoi(Y.T)
fig=voronoi_plot_2d(vor)
plt.plot(data_vectors[0,:],data_vectors[1,:],'k.')
plt.plot(Y[0,:],Y[1,:],'r.',markersize=20)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show() 