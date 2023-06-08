#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:48:12 2023

@author: cmau
"""

import scipy as sp
import numpy as np
import hdf5storage
import pickle
import argparse

from testModel import *


parser = argparse.ArgumentParser(description="Test the model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-mp","--modelPath", help="path where the trained model is saved", default="./")
parser.add_argument("-n","--nameSavedModel", help="name of the saved model",default="trainedModel")




args = parser.parse_args()
config = vars(args)
print(config)

modelPath = config['modelPath']
nameSavedModel = config['nameSavedModel']


#dataPath = './'
dataPath =  '/dtu-compute/cmau/UKBiobank/final_data/healthy_general/T1/unbiased/nonlinear/final/down3/'


#load model

modelPklFile = modelPath + nameSavedModel + ".pkl"  
with open(modelPklFile, 'rb') as file:  
    model = pickle.load(file)
    
W = model["W"]
V = model["V"] 
beta = model["beta"]
trainBasisFunMean = model["trainBasisFunMean"]
trainBasisFunStd = model["trainBasisFunStd"]
imTrainMean = model["imTrainMean"]
imTrainStd = model["imTrainStd"]
indecesMask = model["indecesMask"]


ageDict = sp.io.loadmat(dataPath + "age_uk_healthy_test.mat")
age = ageDict["age_healthy_test"]


T1testDict = hdf5storage.loadmat(dataPath + 'T1_nonlin_uk_healthy_test_down3.mat')
T1test = T1testDict["T1_nonlin_down3"]
del T1testDict


testTarget = age
N_test = len(testTarget)
covariatesTest = []


   
#  test basis functions
 
testBasisFun = testTarget
#testBasisFun=[test_target,covariates_test];
stTestBasisFun=(testBasisFun-trainBasisFunMean)/trainBasisFunStd
 
 
allVolumes = T1test



im_test=np.zeros((N_test,np.sum(indecesMask)))
for n in np.arange(N_test):
    vol = np.squeeze(allVolumes[n,:,:,:])
    im_test[n,:] = np.transpose(vol[indecesMask])
 

#cropped_images_test=T1test[:,3:57,4:68,3:57]


#im_test=np.zeros((N_test,54*54*64))
#for n in np.arange(N_test):
#    im_test[n,:]=np.transpose(np.reshape(cropped_images_test[n,:,:,:],(54*54*64,)))


#del cropped_images_test
#im_test = np.squeeze(im_test)

     
#im_test=im_test[:,indecesMask]
#im_test = np.squeeze(im_test)
stTestImages=(im_test-imTrainMean)/imTrainStd 

(targetPredictions,correlation,MAE,RMSE) = testModel(stTestImages,stTestBasisFun,testTarget,V,beta,W,trainBasisFunMean,trainBasisFunStd)

testResults = {
  "targetPredictions": targetPredictions,
  "correlation": correlation,
  "MAE": MAE,
  "RMSE": RMSE
}

testResultsPath = './'

testPklFile = testResultsPath + "testResultsNew.pkl"  

with open(testPklFile, 'wb') as file:  
    pickle.dump(testResults, file)
