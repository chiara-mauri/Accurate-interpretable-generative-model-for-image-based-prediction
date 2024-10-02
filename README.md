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
conda create -n gen_env python=3.8 scipy nibabel matplotlib hdf5storage scikit-learn -c anaconda -c conda-forge

```


## Preprocessing

- the images needs to be nonlinearly registered to a common space. This can be done e.g. using Freesurfer...

- the experiments in the paper have been performed with downsampling the images.. this can be done with..

## Training

### Python
You can train the model on your own data with the following command:


```
python runTraining.py -nLat < LatentValues > -dp < /path/to/training/data > -fig 
```

where 

- -nLat or --nLatentValues specifies all the values for the number of latent variables that you want to try. E.g. -nLat 20,50,70,100

Optional parameters:

- -sp or --savePath specifies the path where the trained model is saved, default="."
- -n or --nameSavedModel specifies the name given to trained model, default="trainedModel"
- -dp or --dataPath specifies tha folder containing the training data, default=".". See below for requirements on the data format
- -th or --maskThreshold specifies the threshold to use for masking out the background before trianing the model. The threshold is applied to the average volume scaled by its maximum. default=0.01
- -fig or --showFigures if specified, figures are displayed when running


The code assumes that training data are provided in the following format: 
- trainAge.npy is a numpy array of size (# of subjects, 1) with age of all training subjects;
- trainImages.npy is 4D numpy array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3) with images of all training subjects (nonlinearly registered to a common space).


The code saves a pickle containing the model with all the specified number of latent variables

### Matlab:

- trainAge.mat (for Matlab) or trainAge.npy (for Python) is an array of size (# of subjects, 1) with age of all training subjects;
- trainImages.mat (for Matlab) or trainImages.npy (for Python) is 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3) with images of all training subjects (nonlinearly registered to a common space).

## Validation and testing

### Python

We can apply the models we previously trained with different numbers of latent variables to a validation set, to select the optimal number of latent variables. This can be done with:

```
python runValidation.py  -dp < /path/to/validation/data > -n < nameSavedModel.pkl >
```
With optional parameters:

- -mp or --modelPath specifies the path where the trained model is saved, default="."
- -n or --nameSavedModel specifies the name of the pickle file we saved after training, containing the model with all the specified number of latent variables, default="trainedModel.pkl"
- -dp or --dataPath specifies the path to validation data, default=".", see below for requirements on the data format

The code assumes that validation data are provided in the following format: 
- validAge.npy is a numpy array of size (# of subjects, 1) with age of all validation subjects;
- validImages.npy is 4D numpy array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3) with images of all validation subjects (nonlinearly registered to a common space).

the code displays the test metrics on the validation set for all the specified number of latent variables, and save the corresponding information in a pickle file. This can be used to select the optimal number of latent variables.

We are now ready for the final evaluation of the model on a separate test set, using the optimal number of latent variables that we selected. This can be done with:

```
python runTest.py -nLat < OptimalNumberOfLatentVar > -dp < /path/to/test/data > -n < nameSavedModel.pkl >
```
where:

- -nLat or --nLatent specifies the number of latent variables that we want to use for testing (it should be the optimal value selected on the validation set)

With optional parameters:

- -mp or --modelPath specifies the path where the trained model is saved, default="."
- -n or --nameSavedModel specifies the name of the pickle file we saved after training, containing the model with all the specified number of latent variables, default="trainedModel.pkl"
- -dp or --dataPath specifies the path to test data, default=".", See below for requirements on the data format


The code assumes that test data are provided in the following format: 
- testAge.npy is a numpy array of size (# of subjects, 1) with age of all test subjects;
- testImages.npy is 4D numpy array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3) with images of all test subjects (nonlinearly registered to a common space).

The code displays the metrics on the test set with the final model and save the corresponding information in a pickle file.

### Matlab



NOTE: the Python code is now slower than the Matlab version, but it will be updated.

## To cite 
Please cite:

**A Lightweight Causal Model for Interpretable Subject-level Prediction**. Mauri, C., Cerri, S., Puonti, O., Mühlau, M. and Van Leemput, K., 2023. arXiv preprint arXiv:2306.11107
https://arxiv.org/pdf/2306.11107.pdf


  

  
