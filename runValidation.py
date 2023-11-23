#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import scipy as sp
import numpy as np
import hdf5storage
import pickle
import argparse

from testModel import *


parser = argparse.ArgumentParser(description="Validate the model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-mp","--modelPath", help="path where the trained model is saved", default="./")
parser.add_argument("-n","--nameSavedModel", help="name of the saved model",default="trainedModel.pkl")
parser.add_argument("-dp","--dataPath", help="path to validation data", default="./")



args = parser.parse_args()
config = vars(args)
print(config)

modelPath = config['modelPath']
nameSavedModel = config['nameSavedModel']
dataPath = config['dataPath']



#load trained model

modelPklFile = modelPath + nameSavedModel   
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
nLatentValues = model["nLatentValues"]


#load validation data

age = np.load(dataPath + 'validAge.npy')
# array of size (# of subjects, 1) with age of all validation subjects
allVolumes = np.load( dataPath + 'validImages.npy')
# 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
# with images of all validation subjects


validTarget = age
Nvalid = len(validTarget)
covariatesValid = []


   
#  test basis functions
 
validBasisFun = validTarget
stValidBasisFun=(validBasisFun-trainBasisFunMean)/trainBasisFunStd
 
 


imValid=np.zeros((Nvalid,np.sum(indecesMask)))
for n in np.arange(Nvalid):
    vol = np.squeeze(allVolumes[n,:,:,:])
    imValid[n,:] = np.transpose(vol[indecesMask])
del allVolumes 

stValidImages=(imValid-imTrainMean)/imTrainStd 


#validation of model

nLatentTot = len(nLatentValues)
correlations = np.zeros([nLatentTot,1])
MAEs = np.zeros([nLatentTot,1])
RMSEs = np.zeros([nLatentTot,1])
targetPredictionsAll = list()

n = 0
for nLatentVars in nLatentValues:
    
    print("N. latent = " + str(nLatentVars) + "\n")
    V = Vall[n]
    beta = betaAll[n]
    
    (targetPredictions,correlation,MAE,RMSE) = testModel(stValidImages,stValidBasisFun,validTarget,V,beta,W,trainBasisFunMean,trainBasisFunStd)
    
    correlations[n] = correlation
    MAEs[n] = MAE
    RMSEs[n] = RMSE 
    targetPredictionsAll.append(targetPredictions)
    
    n = n+1
    
    
    
validResults = {
  "targetPredictionsAll": targetPredictionsAll,
  "correlations": correlations,
  "MAEs": MAEs,
  "RMSEs": RMSEs
}

#save test results

validResultsPath = './'
validPklFile = validResultsPath + "validationResults.pkl"  

with open(validPklFile, 'wb') as file:  
    pickle.dump(validResults, file)

