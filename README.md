# Interpretable-subject-level-prediciton

In this repository, we present a model for image-based single-subject prediction that is inherently interpretable. We provide both the Matlab and the Python version of the code. The repository does not contain a pre-trained model, but it can be used to train a model on your own data. Details about the model can be found in:


**Accurate and explainable image-based prediction using a lightweight generative model**. Mauri, C., Cerri, S., Puonti, O., Mühlau, M. and Van Leemput, K., 2022. In ‘International Conference on Medical Image Computing and
Computer-Assisted Intervention’, Springer, pp. 448–458. https://link.springer.com/chapter/10.1007/978-3-031-16452-1_43

**A Lightweight Causal Model for Interpretable Subject-level Prediction**. Mauri, C., Cerri, S., Puonti, O., Mühlau, M. and Van Leemput, K., 2023. arXiv preprint arXiv:2306.11107
https://arxiv.org/pdf/2306.11107.pdf


## Installation

1. Clone this repository

2. For Python code, create a virtual environment (e.g. with conda) and install the required packages:

```
conda create -n gen_env python=3.8 scipy nibabel matplotlib hdf5storage  -c anaconda -c conda-forge

```



## Preprocessing

- the images needs to be nonlinearly registered to a common space. This can be done e.g. using Freesurfer...

- the experiments in the paper have been performed with downsampling the images.. this can be done with..

## Training

## Python:
You can train the model on your own data with the following command:


```
python runTraining.py -nLat < LatentValues > -dp < /path/to/data > -fig True
```

where 

- -nLat or --nLatentValues pecifies all the values for the number of latent variables that you want to try. E.g. -nLat 20,50,100

Optional parameters:
- -sp or --savePath specifies the path where the trained model is saved, default="."
- -n or --nameSavedModel specifies the name given to trained model, default="trainedModel"
- -dp or --dataPath specifies tha folder containing the training data, default=".". See below for requirements on the data format
- -th or --maskThreshold specifies the threshold to use for masking out the background before trianing the model. The threshold is applied to the average volume scaled by its maximum. default=0.01
- -fig or --showFigures specifies if figures are displayed when running, default=True


The code assumes that training data are provided in the following format: 
- trainAge.mat (for Matlab) or trainAge.npy (for Python) is an array of size (# of subjects, 1) with age of all training subjects;
- trainImages.mat (for Matlab) or trainImages.npy (for Python) is 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3) with images of all training subjects (nonlinearly registered to a common space).

## Matlab:

## Validation and testing



Validation and testing data should be provided in the same way. 

NOTE: the Python code is now slower than the Matlab version, but it will be updated.

## To cite 
Please cite:

**A Lightweight Causal Model for Interpretable Subject-level Prediction**. Mauri, C., Cerri, S., Puonti, O., Mühlau, M. and Van Leemput, K., 2023. arXiv preprint arXiv:2306.11107
https://arxiv.org/pdf/2306.11107.pdf


  

  
