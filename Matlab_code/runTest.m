%run test

dataPath = './';
modelPath = './';
savePath = './';
nameSavedModel = 'trainedModel';

%load saved trained model
load([modelPath,nameSavedModel])


W = double(model.W);
Vall = model.Vall;
betaAll = model.betaAll;
trainBasisFunMean = model.trainBasisFunMean;
trainBasisFunStd = model.trainBasisFunStd;
imTrainMean = model.imTrainMean;
imTrainStd = model.imTrainStd;
indecesMask = model.indecesMask;
nLatentValues = model.nLatentValues;

nLatentVars = input('selected number of latent variables: \n');
latentIndex = find(nLatentValues==nLatentVars);
V = double(Vall{latentIndex,1});
beta = betaAll{latentIndex,1};

%load test data 


load([dataPath,'testAge.mat']) %load age_healthy_test
%array of size (# of subjects, 1) with age of all test subjects


testTarget=age_healthy_test;
nTest=size(testTarget,1);
covariatesTest=[];
clear('age_healthy_test')

%test basis functions

test_basis_fun=[testTarget,covariatesTest];
stTestBasisFun=(test_basis_fun-trainBasisFunMean)./trainBasisFunStd;


%test images

load([dataPath,'testImages.mat']) %load T1_nonlin_down3
%load 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
% with images of all test subjects

allVolumes = T1_nonlin_down3;
clear T1_nonlin_down3

imTest = zeros(nTest,numel(indecesMask));
for nSubj = 1:nTest
    vol = squeeze(allVolumes(nSubj,:,:,:));
    imTest(nSubj,:) = vol(indecesMask);
end
clear('allVolumes')


stTestImages = (imTest-imTrainMean)./imTrainStd;
clear('imTest')


% target predictions


[targetPredictions,correlation,MAE,RMSE] = testModel(stTestImages,stTestBasisFun,testTarget,V,beta,W,trainBasisFunMean,trainBasisFunStd);



testResults.nLatent=nLatentVars;
testResults.correlation=correlation;
testResults.MAE= MAE;
testResults.RMSE= RMSE;
testResults.targetPredictions=targetPredictions;


save([savePath,'testResults'],'testResults','-v7.3')

    
   
