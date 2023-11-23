#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:00:29 2022

@author: chiara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:06:01 2022

@author: cmau
"""

import numpy as np
import scipy as sp
import time


def trainModel(trainImages,trainBasisFun,nLatentVars,costTol=1e-5, nItMin=10, nItMax=500):
    
    #costTol is tolerance for convergence of logml in the training
    #nItMin is minimum number of iterations during training 



    #compute causal weights W  
    
    D_W = np.matmul(np.transpose(trainImages),trainBasisFun)
    A_W = np.matmul(np.transpose(trainBasisFun),trainBasisFun)
    W = np.transpose(np.linalg.solve(np.transpose(A_W),np.transpose(D_W)))
    #w_star = W[:,1]
          
    nVoxels = np.shape(W)[0]
    nBasisFun = np.shape(W)[1]
    Ntrain = np.shape(trainBasisFun)[0]
          
    #inItialize parameters beta and V
        
    betaInit = np.zeros((nVoxels,))
    nLoops = np.ceil(nVoxels/20000)
    for i in np.arange(nLoops):
        indeces = np.arange(i*20000,np.amin([(i+1)*20000,nVoxels]),dtype=int)
        betaInit[indeces] = np.transpose(1/(np.var(trainImages[:,indeces],0))) 
        
    beta = np.reshape(betaInit,(nVoxels,1))
        
    #rng(n_inIt)
        
       
    VInit = np.random.normal(size=(nVoxels,nLatentVars)) #randomly sampled from standard Gaussian
    V = VInit
     
        
     
    logML = float('inf')
    logML_old = float('inf')
        
        
    logmlVector = np.zeros([nItMax,1])    
      
    
       
    
    #start training
    t = time.time()


    for nIt in np.arange(nItMax):
        if nIt > nItMin:
            if (np.abs((logML - logML_old)/logML) < costTol):  
                V = Vold
                beta = betaOld
                
                break
         
        
        print('it number:', nIt, '\n')
    
        #save old values
        Vold = V
        betaOld = beta
        logML_old = logML
          
        #save logML values for each iteration
        logmlVector[nIt] = logML
           
        #diagBetas = sp.sparse.spdiags(beta,0,nVoxels,nVoxels)
  
        
        
        #update of posterior mean and variance of latent variables
    
        #SigmaZ_inv=np.eye(nLatentVars)+np.linalg.multi_dot((np.transpose(V),diagBetas,V))
        SigmaZ_inv=np.eye(nLatentVars)+np.dot(np.transpose(V),beta*V)
        SigmaZ=np.linalg.inv(SigmaZ_inv)
            
        dimBlock = 8000
        nBlocksTot = np.ceil(Ntrain/dimBlock)
          
        muZ=np.zeros((Ntrain,nLatentVars))
           
        for n_block in np.arange(nBlocksTot):
            indeces_block = np.arange(n_block*dimBlock,np.amin([(n_block+1)*dimBlock,Ntrain]),dtype=int)
            aux = trainImages[indeces_block,:]-np.matmul(trainBasisFun[indeces_block,:],np.transpose(W))
            #muZ[indeces_block,:] = np.linalg.multi_dot((aux,diagBetas,V,SigmaZ))
            muZ[indeces_block,:] = np.linalg.multi_dot((aux,beta*V,SigmaZ))
               
          
        del aux
           
           
        #compute log marginal likelihood, with Cholesky decomposition
          
            
        U=np.linalg.cholesky(SigmaZ_inv)   #detSigmaZninv=det(U)^2;
        log_det_SigmaZ_inv=2*np.sum(np.log(np.diag(U)))
           
           
        
        dimBlock = 1000
        nBlocksTot = int(np.ceil(Ntrain/dimBlock))
        sumArray = np.zeros((nBlocksTot,))
                    
    
        for n_block in np.arange(nBlocksTot):
            indeces_block = np.arange(n_block*dimBlock,np.amin([(n_block+1)*dimBlock,Ntrain]),dtype=int)
            aux = trainImages[indeces_block,:]-np.matmul(trainBasisFun[indeces_block,:],np.transpose(W))
            #sum1 = np.trace(np.linalg.multi_dot((aux,diagBetas,np.transpose(aux))))
            sum1 = np.trace(np.dot(aux,beta*np.transpose(aux)))
            aux2 = np.dot(aux,beta*V)
            sum2 = np.trace( np.linalg.multi_dot((aux2,SigmaZ,np.transpose(aux2))))
            #probably I an use only one multidot 
            sumArray[n_block] = sum1-sum2
    
    
    
        del aux
        summ_logml=np.sum(sumArray)
            
        logML = -0.5*(Ntrain*nVoxels*np.log(2*np.pi)-Ntrain*np.sum(np.log(beta))+Ntrain*log_det_SigmaZ_inv+summ_logml)
          
        print('logML: ',logML,'\n')
        
          
            
        #update of V
            
        A = np.dot(np.transpose(muZ),muZ)+Ntrain*SigmaZ
        D = np.dot(np.transpose(trainImages),muZ)-np.linalg.multi_dot((W,np.transpose(trainBasisFun),muZ))
        V = np.transpose(np.linalg.solve(np.transpose(A),np.transpose(D)))
            
        #add other version!
        
        
        
        dimBlock=np.amax([np.floor(1500/(nBasisFun+nLatentVars)),1])
        #voxel_per_batch=1500
        nBlocksTot=np.ceil(nVoxels/dimBlock) #N. of batches used for updates during training
            
        
            
        #inizialize values for parallel loop
        # beta_par=cell(n_batches,1);
        # W_par=cell(n_batches,1);
        # V_par=cell(n_batches,1);
        # trainImages_par=cell(1,n_batches);
           
        # for i=1:n_batches
        #     voxelsBlock=1+(i-1)*voxel_per_batch:min(n_voxels,i*voxel_per_batch);%indixes of voxels of current batch
               
               
        #      trainImages_par{i}=trainImages(:,voxelsBlock); %train images of the batch (voxels in the batch)
        #      W_par{i}=W(voxelsBlock,:);
        #      V_par{i}=V(voxelsBlock,:);
                
           
            
           
            #parallel loop for computing betas
        # for i=1:n_batches
               
        # #compute elements of beta corresponding to the batch
                
            # beta_par{i}=Ntrain/
            #         (np.transpose(sum((trainImages_par{i}-trainBasisFun*np.transpose(W_par{i})-
            #         muZ*np.transpose(V_par{i})).^2))+diag(V_par{i}*Ntrain*SigmaZ*np.transpose(V_par{i})))
                
            
            
        # #update of betas
            
        # beta=cell2mat(beta_par)
           
           #%%
        beta = np.zeros([nVoxels,])
        for n_block in np.arange(nBlocksTot):
            voxelsBlock = np.arange(n_block*dimBlock,np.amin([(n_block+1)*dimBlock,nVoxels]),dtype=int)
                    
            beta[voxelsBlock] = Ntrain/(
                 np.transpose(np.sum( np.square(trainImages[:,voxelsBlock]-np.dot(trainBasisFun,np.transpose(W[voxelsBlock,:]))-
                 np.dot(muZ,np.transpose(V[voxelsBlock,:]))),axis=0))+np.diag(np.linalg.multi_dot((V[voxelsBlock,:],Ntrain*SigmaZ,np.transpose(V[voxelsBlock,:])))))
                   
        beta = np.reshape(beta,(nVoxels,1))
#%%

    nIterations = nIt - 1   
    timeTraining = time.time() - t
    print('number of iterations: ', nIterations,'\n')
    print('training time: ', timeTraining,' s \n')
        
    return W, V, beta, timeTraining, logmlVector, nIterations




