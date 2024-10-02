# Interpretable-subject-level-prediciton

This repository contains both the Matlab and the Python version of the code. 


##Installation

##Preprocessing

- the images needs to be nonlinearly registered to a common space. This can be done e.g. using Freesurfer...
- the experiments in the paper have been performed with downsampling the images.. this can be done with..

##Training

Training the model is performed by:

python runTraining.py -nLatentValues <LatentValues>

where 

-LatentValues contains all the values for the number of latent variables that you want to try. E.g. -nLatentValues 20,50,100

Optional parameters:
- sp
- n
- dp
- th


The code assumes that training data are provided in the following format: 
- trainAge.mat (for Matlab) or trainAge.npy (for Python) is an array of size (# of subjects, 1) with age of all training subjects;
- trainImages.mat (for Matlab) or trainImages.npy (for Python) is 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3) with images of all training subjects (nonlinearly registered to a common space).

##Validation and testing



Validation and testing data should be provided in the same way. 

NOTE: the Python code is now slower than the Matlab version, but it will be updated.

## To cite 
Please cite:

A Lightweight Causal Model for Interpretable Subject-level Prediction. Mauri, C., Cerri, S., Puonti, O., Mühlau, M. and Van Leemput, K., 2023. arXiv preprint arXiv:2306.11107
https://arxiv.org/pdf/2306.11107.pdf


  

  
