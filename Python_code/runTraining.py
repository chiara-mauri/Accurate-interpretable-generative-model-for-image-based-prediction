#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:49:48 2023

@author: cm1991
"""

import scipy as sp
import numpy as np
import hdf5storage
import pickle
import argparse
import matplotlib.pyplot as plt

from trainModel import *


parser = argparse.ArgumentParser(description="Script for training the model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

#parser.add_argument("-nLatentValues", required =  True, nargs="+", help="list with number of latent variables to try, e.g. -nLatentValues 20,50,100")
parser.add_argument("-nLat","--nLatentValues", required =  True, type=list_of_ints, help="list of number of latent variables to try, e.g. -nLatentValues 20,50,100")
parser.add_argument("-sp","--savePath", help="path where the trained model is saved", default=".")
parser.add_argument("-n","--nameSavedModel", help="name given to trained model",default="trainedModel")
parser.add_argument("-dp","--dataPath", help="path to data", default=".")
parser.add_argument("-th","--maskThreshold", type=float ,help="threshold for data mask", default=0.01)
parser.add_argument("-fig","--showFigures", action='store_true' ,help="Show figures")


args = parser.parse_args()
config = vars(args)


nLatentValues = config['nLatentValues']
saveModelPath = config['savePath']
nameSavedModel = config['nameSavedModel']
dataPath = config['dataPath']
maskThreshold = config['maskThreshold']
showFigures = config['showFigures']


#load training data

age = np.load(dataPath + "/trainAge.npy")
# array of size (# of subjects, 1) with age of all training subjects
allVolumes = np.load(dataPath + "/trainImages.npy")
# 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
# with images of all training subjects

targetTrain = age
N_train = len(targetTrain)
covariatesTrain = np.empty((N_train,0))


#train basis functions
trainBasisFunInit = np.concatenate((targetTrain,covariatesTrain),axis=1)
nBasisFun = np.shape(trainBasisFunInit)[1] #number of basis functions
  

trainBasisFunMean = np.mean(trainBasisFunInit,axis=0) #vector 1 x nBasisFun with means
trainBasisFunStd = np.std(trainBasisFunInit,axis=0) #vector 1 x nBasisFun with stds
trainBasisFun = (trainBasisFunInit-trainBasisFunMean)/trainBasisFunStd
 

   #%%

#compute mask of voxels 
avgVol=np.squeeze(np.mean(allVolumes,axis=0))
avgVolScaled=avgVol/np.max(avgVol)
indecesMask = (avgVolScaled>maskThreshold)
        
if showFigures:
    shape = np.shape(avgVol)
    sliceToShow = int(np.round(shape[2]/2))
    slice1 = avgVol[:,:,sliceToShow]
    plt.imshow(slice1, cmap='gray')
    plt.colorbar()
    plt.title("Average volume")
    plt.show()

    slice2 = avgVolScaled[:,:,sliceToShow]
    plt.imshow(slice2, cmap='gray')
    plt.colorbar()
    plt.title("Scaled average volume")
    plt.show()

    mask = np.zeros(np.shape(avgVol))
    mask[indecesMask] = 1
    slice3 = mask[:,:,sliceToShow]
    plt.imshow(slice3, cmap='gray')
    plt.colorbar()
    plt.title("Mask")
    plt.show()
#%%


trainImagesInit=np.zeros((N_train,np.sum(indecesMask)))
for n in np.arange(N_train):
    vol = np.squeeze(allVolumes[n,:,:,:])
    trainImagesInit[n,:] = np.transpose(vol[indecesMask])
del allVolumes
      

imTrainMean = np.mean(trainImagesInit,axis=0) 
imTrainStd = np.std(trainImagesInit,axis=0)
   
trainImages=(trainImagesInit-imTrainMean)/imTrainStd 
del trainImagesInit


#initializations
nLatentTot = len(nLatentValues)
nIterationsAll = np.zeros([nLatentTot,1])
timeTrainingAll = np.zeros([nLatentTot,1])
Vall = list()
betaAll = list()
logmlVectorAll = list()

#model training 

n = 0
for nLatentVars in nLatentValues:
    
    print("N. latent = " + str(nLatentVars) + "\n")
    (W, V, beta, timeTraining, logmlVector, nIterations) = trainModel(trainImages,trainBasisFun,nLatentVars)


    nIterationsAll[n] = nIterations
    Vall.append(V)
    betaAll.append(beta)
    timeTrainingAll[n] = timeTraining
    logmlVectorAll.append(logmlVector)
     
    n = n+1


#compute original weights (without standardization) and visualize them

W_original_0 = np.multiply(np.multiply(np.reshape(imTrainStd,[np.sum(indecesMask),1]),W),np.reshape(1/trainBasisFunStd,[1,nBasisFun]))
W_original_offset = np.reshape(imTrainMean,[np.sum(indecesMask),1])
W_original = np.concatenate((W_original_0, W_original_offset),axis=1)
#%%
if showFigures:
    #target weights
    sliceToShow = int(np.round(shape[2]/2))
    weights3D = np.zeros(np.shape(avgVol))
    weights3D[indecesMask] = W_original[:,0]
    sliceW = weights3D[:,:,sliceToShow]
    alphas = np.abs(sliceW)
    alphas = alphas/np.max(alphas)
    template = avgVol[:,:,sliceToShow]
    alphasTemplate = np.zeros(np.shape(avgVol))
    alphasTemplate[indecesMask] = 0.5
    plt.imshow(template, alpha = alphasTemplate[:,:,sliceToShow], cmap='gray')
    plt.imshow(sliceW, alpha = alphas)
    colorbar = plt.colorbar()
    #plt.colorbar.set_alpha(0.5)
    plt.title('target weights')
    plt.show()
    
    #offset
    offset3D = np.zeros(np.shape(avgVol))
    offset3D[indecesMask] = W_original[:,-1]
    slice_offset = offset3D[:,:,sliceToShow]
    plt.imshow(slice_offset,cmap='gray')
    colorbar = plt.colorbar()
    #plt.colorbar.set_alpha(0.5)
    plt.title('offset')
    plt.show()
    
    
#%%
model = {
  "W": W,
  "W_original": W_original,
  "Vall": Vall,
  "betaAll": betaAll,
  "timeTrainingAll": timeTrainingAll,
  "logmlVectorAll": logmlVectorAll,
  "nIterationsAll": nIterationsAll,
  "trainBasisFunMean": trainBasisFunMean,
  "trainBasisFunStd": trainBasisFunStd,
  "imTrainMean": imTrainMean,
  "imTrainStd": imTrainStd, 
  "indecesMask": indecesMask,
  "nLatentValues": nLatentValues 
}

 
#save trained model
modelPklFile = saveModelPath + "/" + nameSavedModel + ".pkl"  

with open(modelPklFile, 'wb') as file:  
    pickle.dump(model, file)
