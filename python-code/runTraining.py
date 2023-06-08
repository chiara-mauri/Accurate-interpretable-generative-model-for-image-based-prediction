#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:08:41 2022

@author: chiara
"""
#%%
import scipy as sp
import numpy as np
import hdf5storage
import pickle
import argparse

from trainModel import *


parser = argparse.ArgumentParser(description="Train the model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
#parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
#parser.add_argument("-B", "--block-size", help="checksum blocksize")
#parser.add_argument("--ignore-existing", action="store_true", help="skip files that exist")
#parser.add_argument("--exclude", help="files to exclude")
#parser.add_argument("src", help="Source location")
#parser.add_argument("dest", help="Destination location")



#parser.add_argument("-K","--n_latent_variables", help="number of latent variables.",type=int)
parser.add_argument("nLatentVars", help="number of latent variables.", type=int)
parser.add_argument("-sp","--savePath", help="path to save the trained model", default="./")
parser.add_argument("-n","--nameSavedModel", help="name of the saved model",default="trainedModel")




args = parser.parse_args()
config = vars(args)
print(config)

nLatentVars = config['nLatentVars']
saveModelPath = config['savePath']
nameSavedModel = config['nameSavedModel']


#dataPath = './'
dataPath =  '/dtu-compute/cmau/UKBiobank/final_data/healthy_general/T1/unbiased/nonlinear/final/down3/'
ageDict = sp.io.loadmat(dataPath + "age_uk_healthy_valid.mat")
age = ageDict["age_healthy_valid"]




T1validDict = hdf5storage.loadmat(dataPath + 'T1_nonlin_uk_healthy_valid_down3.mat')
T1valid = T1validDict["T1_nonlin_down3"]
del T1validDict


targetTrain = age
N_train = len(targetTrain)
covariatesTrain = []


#%%

   
#train basis functions
trainBasisFunInit = targetTrain
#trainBasisFunInit=np.concatenate((targetTrain,covariatesTrain))
     #matrix n_train_images x n_basis_fun with basis functions of all train subjects
nBasisFun = np.shape(trainBasisFunInit)[1] #number of basis functions
  

trainBasisFunMean = np.mean(trainBasisFunInit,axis=0) #vector 1 x nBasisFun with means
trainBasisFunStd = np.std(trainBasisFunInit,axis=0) #vector 1 x nBasisFun with stds
trainBasisFun = (trainBasisFunInit-trainBasisFunMean)/trainBasisFunStd
 

   


#cropped_images_train=T1valid[:,3:57,4:68,3:57]



#im_train_init=np.zeros((N_train,54*54*64))
#for n in np.arange(N_train):
 #   im_train_init[n,:]=np.transpose(np.reshape(cropped_images_train[n,:,:,:],(54*54*64,)))


#del cropped_images_train
#im_train_init = np.squeeze(im_train_init)


allVolumes = T1valid

avgVol=np.squeeze(np.mean(allVolumes,axis=0))
#imTrainStd=np.std(im_train_init,axis=0)

#indecesMask=np.asarray(np.where(avgVol>15))
indecesMask = (avgVol>15)
        
#im_train_init=np.zeros((N_train,len(indecesMask)))
im_train_init=np.zeros((N_train,np.sum(indecesMask)))
for n in np.arange(N_train):
    vol = np.squeeze(allVolumes[n,:,:,:])
    im_train_init[n,:] = np.transpose(vol[indecesMask])

   
#im_train_init=im_train_init[:,indecesMask]
#im_train_init = np.squeeze(im_train_init)

#imTrainMean=imTrainMean[indecesMask]
#imTrainStd=imTrainStd[indecesMask]
#n_voxels=size(im_train_init,2); %number of voxels in the mask
        

imTrainMean = np.mean(im_train_init,axis=0) 
imTrainStd = np.std(im_train_init,axis=0)
   
trainImages=(im_train_init-imTrainMean)/imTrainStd 
#clear('im_train_init')






(W, V, beta, timeTraining, logmlVector, nIterations) = trainModel(trainImages,trainBasisFun,nLatentVars)


model = {
  "W": W,
  "V": V,
  "beta": beta,
  "timeTraining": timeTraining,
  "logmlVector": logmlVector,
  "nIterations": nIterations,
  "trainBasisFunMean": trainBasisFunMean,
  "trainBasisFunStd": trainBasisFunStd,
  "imTrainMean": imTrainMean,
  "imTrainStd": imTrainStd, 
  "indecesMask": indecesMask 
}

#saveModelPath = './'
#modelPklFile = saveModelPath + "trainedModelNew.pkl"  

modelPklFile = saveModelPath + nameSavedModel + ".pkl"  

with open(modelPklFile, 'wb') as file:  
    pickle.dump(model, file)

