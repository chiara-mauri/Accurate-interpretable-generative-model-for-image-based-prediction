%run test

function runTest(varargin)

p = inputParser;
addRequired(p,'nLatent',@(x) isnumeric(x));
addParameter(p,'dataPath','.',@(x) ischar(x) || isstring(x));
addParameter(p,'savePath','.',@(x) ischar(x) || isstring(x));
addParameter(p,'modelPath','.',@(x) ischar(x) || isstring(x));
addParameter(p,'nameSavedModel','trainedModel.mat',@(x) ischar(x) || isstring(x));



parse(p, varargin{:});
dataPath = p.Results.dataPath;
savePath = p.Results.savePath;
modelPath = p.Results.modelPath;
nameSavedModel = p.Results.nameSavedModel;
nLatent = p.Results.nLatent;

if ~exist(savePath, 'dir')
    % If it does not exist, create the folder
    mkdir(savePath);
end

%load saved trained model
load([modelPath,'/',nameSavedModel])


W = double(model.W);
Vall = model.Vall;
betaAll = model.betaAll;
trainBasisFunMean = model.trainBasisFunMean;
trainBasisFunStd = model.trainBasisFunStd;
imTrainMean = model.imTrainMean;
imTrainStd = model.imTrainStd;
indecesMask = model.indecesMask;
nLatentValues = model.nLatentValues;


latentIndex = find(nLatentValues==nLatent);
V = double(Vall{latentIndex,1});
beta = betaAll{latentIndex,1};

%load test data 


load([dataPath,'/testAge.mat']) %load age_healthy_test
%array of size (# of subjects, 1) with age of all test subjects


testTarget=testAge;
nTest=size(testTarget,1);
covariatesTest=[];
clear('testAge')

%test basis functions

test_basis_fun=[testTarget,covariatesTest];
stTestBasisFun=(test_basis_fun-trainBasisFunMean)./trainBasisFunStd;


%test images

load([dataPath,'/testImages.mat']) %load T1_nonlin_down3
%load 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
% with images of all test subjects
allVolumes = testImages;
clear testImages

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



testResults.nLatent=nLatent;
testResults.correlation=correlation;
testResults.MAE= MAE;
testResults.RMSE= RMSE;
testResults.targetPredictions=targetPredictions;


save([savePath,'/testResults'],'testResults','-v7.3')

    
   
