#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import scipy as sp
import numpy as np
import hdf5storage
import pickle
import argparse
import os

from testModel import *


parser = argparse.ArgumentParser(description="Test the model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-mp","--modelPath", help="path where the trained model is saved", default=".")
parser.add_argument("-n","--nameSavedModel", help="name of the saved model",default="trainedModel.pkl")
parser.add_argument("-dp","--dataPath", help="path to test data", default=".")
parser.add_argument("-nLat","--nLatent", required =  True, help="n. of latent variables to use for testing", type = int)
parser.add_argument("-sp","--savePath", help="path where the test results are saved", default=".")


args = parser.parse_args()
config = vars(args)

modelPath = config['modelPath']
nameSavedModel = config['nameSavedModel']
dataPath = config['dataPath']
nLatentVars = config['nLatent']
savePath = config['savePath']


if not os.path.exists(savePath):
    os.mkdir(savePath)  



#load trained model

modelPklFile = modelPath + '/' + nameSavedModel 
with open(modelPklFile, 'rb') as file:  
    model = pickle.load(file)
    
W = model["W"]
Vall = model["Vall"] 
betaAll = model["betaAll"]
trainBasisFunMean = model["trainBasisFunMean"]
trainBasisFunStd = model["trainBasisFunStd"]
imTrainMean = model["imTrainMean"]
imTrainStd = model["imTrainStd"]
indecesMask = model["indecesMask"]
nLatentValues = np.asarray(model["nLatentValues"])

latentIndex = np.where(nLatentValues == nLatentVars)
latentIndex = latentIndex[0][0]
V = Vall[latentIndex]
beta = betaAll[latentIndex]


#load test data

age = np.load(dataPath + '/testAge.npy')
# array of size (# of subjects, 1) with age of all test subjects
allVolumes = np.load( dataPath + '/testImages.npy')
# 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
# with images of all test subjects


testTarget = age
Ntest = len(testTarget)
covariatesTest = np.empty((Ntest,0))


   
#  test basis functions
 
testBasisFun = np.concatenate((testTarget,covariatesTest),axis=1)
stTestBasisFun=(testBasisFun-trainBasisFunMean)/trainBasisFunStd
 
 


im_test=np.zeros((Ntest,np.sum(indecesMask)))
for n in np.arange(Ntest):
    vol = np.squeeze(allVolumes[n,:,:,:])
    im_test[n,:] = np.transpose(vol[indecesMask])
del allVolumes 

stTestImages=(im_test-imTrainMean)/imTrainStd 


#test model
(targetPredictions,correlation,MAE,RMSE) = testModel(stTestImages,stTestBasisFun,testTarget,V,beta,W,trainBasisFunMean,trainBasisFunStd)

testResults = {
  "targetPredictions": targetPredictions,
  "correlation": correlation,
  "MAE": MAE,
  "RMSE": RMSE
}

#save test results


testPklFile = savePath + '/' + "testResults.pkl"  

with open(testPklFile, 'wb') as file:  
    pickle.dump(testResults, file)
