%run validation

dataPath = './';
modelPath = './';
savePath = './';
nameSavedModel = 'trainedModel_N1000';

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
nLatentTot = numel(nLatentValues);


%load validation data 

load([dataPath,'age_uk_healthy_valid.mat']) %load age_healthy_valid
%array of size (# of subjects, 1) with age of all validation subjects

targetValid=age_healthy_valid;
nValid=size(targetValid,1);
covariatesValid=[];
clear('age_healthy_valid')


%validation basis functions

validBasisFun=[targetValid,covariatesValid];
stValidBasisFun=(validBasisFun-trainBasisFunMean)./trainBasisFunStd;


%validation images

load([dataPath,'T1_nonlin_uk_healthy_valid_down3.mat']) %load T1_nonlin_down3
%load 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
% with images of all validation subjects

allVolumes = T1_nonlin_down3;
clear T1_nonlin_down3

imValid = zeros(nValid,numel(indecesMask));
for nSubj = 1:nValid
    vol = squeeze(allVolumes(nSubj,:,:,:));
    imValid(nSubj,:) = vol(indecesMask);
end
clear('allVolumes')


stValidImages = (imValid-imTrainMean)./imTrainStd;
clear('imValid')




%initialization
correlations=zeros(nLatentTot,1);
MAEs=zeros(nLatentTot,1);
RMSEs=zeros(nLatentTot,1);
predictions=cell(nLatentTot,1);


% target predictions

   
   for n = 1:nLatentTot

    nLatentVars=nLatentValues(n);
    fprintf('n.latent = %d\n',nLatentVars)
    V = double(Vall{n,1});
    beta = betaAll{n,1};
  
   
    [targetPredictions,correlation,MAE,RMSE] = testModel(stValidImages,stValidBasisFun,targetValid,V,beta,W,trainBasisFunMean,trainBasisFunStd);

  

    %compute correlation and error

    correlations(n)=correlation;
    RMSEs(n)=RMSE;
    MAEs(n)=MAE;
    predictions{n}=single(targetPredictions);

   end




   validResults.nLatentValues=nLatentValues;
   validResults.correlations=correlations;
   validResults.MAEs= MAEs;
   validResults.RMSEs= RMSEs;
   validResults.predictions=predictions;


   save([savePath,'validResults'],'validResults','-v7.3')

   