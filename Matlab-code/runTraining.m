clear all, close all


dataPath = './';
showFigures=1;

%select number of latent variables to try
nLatentValues=[ 0 50 100 200];
%nLatentValues = [100];
nLatentTot=numel(nLatentValues); 




%load target and covariates

load([dataPath,'trainAge_N1000.mat']) %load age
%array of size (# of subjects, 1) with age of all training subjects

targetTrain = age;
Ntrain=size(targetTrain,1);
covariatesTrain=[];
clear('age')


%train basis functions

trainBasisFunInit=[targetTrain,covariatesTrain];

trainBasisFunMean=mean(trainBasisFunInit); 
trainBasisFunStd=std(trainBasisFunInit); 
trainBasisFun=(trainBasisFunInit-trainBasisFunMean)./trainBasisFunStd;


% training images

load([dataPath,'trainImages_N1000.mat']) %load im_healthy_train
%load 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
% with images of all training subjects


allVolumes = im_healthy_train;
clear im_healthy_train

%compute mask
avgVol = squeeze(mean(allVolumes,1));
indecesMask = find(avgVol>15)';
        

trainImagesInit = zeros(Ntrain,numel(indecesMask));
for nSubj = 1:Ntrain
    vol = squeeze(allVolumes(nSubj,:,:,:));
    trainImagesInit(nSubj,:) = vol(indecesMask);
end
   
clear allVolumes
        
%standardization
imTrainMean = mean(trainImagesInit);
imTrainStd = std(trainImagesInit);
trainImages=(trainImagesInit-imTrainMean)./imTrainStd;

clear('trainImagesInit')



%inizialization

nIterationsAll = zeros(nLatentTot,1);
Vall = cell(nLatentTot,1);
betaAll = cell(nLatentTot,1);
timeTrainingAll = zeros(nLatentTot,1);
logMLVectorAll = cell(nLatentTot,1);


%loop over latent variables
for n = 1:nLatentTot

    nLatentVars=nLatentValues(n);
    fprintf('n.latent = %d\n',nLatentVars)
    
    costTol = 1e-5;
    nItMax = 500;
    nItMin = 3;

    %model training
    [W, V, beta, timeTraining, logMLVector, nIterations] = trainingModel(trainImages, ...
        trainBasisFun, nLatentVars, costTol, nItMax, nItMin);


    nIterationsAll(n) = nIterations;
    Vall{n} = single(V);
    betaAll{n}=beta;
    timeTrainingAll(n) = timeTraining;
    logMLVectorAll{n} = logMLVector;


    %plot of logML across iterations
    if showFigures

        figure,
        plot(1:nIterationsAll(n)-1,logMLVector(1,2:nIterationsAll(n)))
        xlabel('iteration number')
        title('logML')

    end

end


%save trained model

model.W=single(W);
model.Vall=Vall;
model.betaAll=betaAll;
model.indecesMask=indecesMask;
model.timeTrainingAll=timeTrainingAll;
model.logMLVectorAll=logMLVectorAll;
model.nIterationsAll=nIterationsAll;

model.nLatentValues=nLatentValues;

model.trainBasisFunMean=trainBasisFunMean;
model.trainBasisFunStd=trainBasisFunStd;
model.imTrainMean=imTrainMean;
model.imTrainStd=imTrainStd;


saveModelPath='./';
nameSavedModel = 'trainedModel';
save([saveModelPath, nameSavedModel, '.mat'],'model','-v7.3')




