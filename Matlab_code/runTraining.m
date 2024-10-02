function runTraining(varargin)

p = inputParser;
addRequired(p,'nLatentValues',@(x) isnumeric(x) && isvector(x));
addParameter(p,'dataPath','.',@(x) ischar(x) || isstring(x));
addParameter(p,'savePath','.',@(x) ischar(x) || isstring(x));
addParameter(p,'nameSavedModel','trainedModel',@(x) ischar(x) || isstring(x));
addParameter(p,'maskThreshold',0.01, @(x) isnumeric(x));
addParameter(p,'showFigures', true, @(x) islogical(x));






parse(p, varargin{:});
nLatentValues = p.Results.nLatentValues;
dataPath = p.Results.dataPath;
saveModelPath = p.Results.savePath;
nameSavedModel = p.Results.nameSavedModel;
maskThreshold = p.Results.maskThreshold;
showFigures = p.Results.showFigures;

if ~exist(saveModelPath, 'dir')
    % If it does not exist, create the folder
    mkdir(saveModelPath);
end

nLatentTot = numel(nLatentValues); 

%load target and covariates

load([dataPath,'/trainAge.mat']) %load age
%array of size (# of subjects, 1) with age of all training subjects

targetTrain = age;
Ntrain = size(targetTrain,1);
covariatesTrain = [];
clear('age')


%train basis functions

trainBasisFunInit = [targetTrain,covariatesTrain];

trainBasisFunMean = mean(trainBasisFunInit); 
trainBasisFunStd = std(trainBasisFunInit); 
trainBasisFun = (trainBasisFunInit-trainBasisFunMean)./trainBasisFunStd;


% training images

load([dataPath,'/trainImages.mat']) %load im_healthy_train
%load 4D array of size (# of subjects, image dimension 1, image dimension 2, image dimension 3)
% with images of all training subjects


allVolumes = im_healthy_train;
clear im_healthy_train

%compute mask
avgVol = squeeze(mean(allVolumes,1));
avgVolScaled=avgVol/max(avgVol(:));
indecesMask = find(avgVolScaled>maskThreshold)';

if showFigures
    shape = size(avgVol);
    sliceToShow = round(shape(3)/2);
    slice1 = avgVol(:,:,sliceToShow);
    figure, imagesc(imrotate(slice1,90)), colormap gray, colorbar
    axis equal
    title('average volume')
    
    slice2 = avgVolScaled(:,:,sliceToShow);
    figure, imagesc(imrotate(slice2,90)), colormap gray, colorbar
    axis equal
    title('scaled average volume')
    
    mask = zeros(size(avgVol));
    mask(indecesMask) = 1;
    slice3 = mask(:,:,sliceToShow);
    figure, imagesc(imrotate(slice3,90)), colormap gray, colorbar
    axis equal
    title('mask')
end
        

trainImagesInit = zeros(Ntrain,numel(indecesMask));
for nSubj = 1:Ntrain
    vol = squeeze(allVolumes(nSubj,:,:,:));
    trainImagesInit(nSubj,:) = vol(indecesMask);
end
   
clear allVolumes
        
%standardization
imTrainMean = mean(trainImagesInit);
imTrainStd = std(trainImagesInit);
trainImages = (trainImagesInit-imTrainMean)./imTrainStd;

clear('trainImagesInit')



%inizialization

nIterationsAll = zeros(nLatentTot,1);
Vall = cell(nLatentTot,1);
betaAll = cell(nLatentTot,1);
timeTrainingAll = zeros(nLatentTot,1);
logMLVectorAll = cell(nLatentTot,1);

delete(gcp('nocreate'))
parpool(16);
%loop over latent variables
for n = 1:nLatentTot

    nLatentVars = nLatentValues(n);
    fprintf('n.latent = %d\n',nLatentVars)
    
    costTol = 1e-5;
    nItMax = 500;
    nItMin = 3;

    
    %model training
    [W, V, beta, timeTraining, logMLVector, nIterations] = trainingModel(trainImages, ...
        trainBasisFun, nLatentVars, costTol, nItMax, nItMin);


    nIterationsAll(n) = nIterations;
    Vall{n} = single(V);
    betaAll{n} = beta;
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

%% compute original weights (without standardization) and visualize them

W_original_0 = spdiags(imTrainStd',0,numel(indecesMask),numel(indecesMask))*W*...
    spdiags(1./trainBasisFunStd',0,size(W,2),size(W,2));
W_original_offset=imTrainMean';
W_original=[full(W_original_0), W_original_offset];

%display offset
if showFigures  

    offset3D = zeros(size(avgVol));
    offset3D(indecesMask)=  W_original_offset;
        
   
    shape = size(avgVol);
    sliceToShow = round(shape(3)/2)-4;
    figure,
    imagesc(imrotate(squeeze(offset3D(:,:,sliceToShow)),90)), colormap gray, 
    title('Offset')
    colorbar('FontSize',13)
    axis equal
    axis off
end

%display target weights

if showFigures  

    targetWeights3D = zeros(size(avgVol));
    targetWeights3D(indecesMask) = W_original(:,1);
    template = squeeze(avgVol(:,:,sliceToShow));
    
    transparencyMask = zeros(size(avgVol));
    transparencyMask(indecesMask)=0.5;
    alpha_template=squeeze(transparencyMask(:,:,sliceToShow));
    
    
    figure;
    ax1 = axes;
    imagesc(imrotate(template,90),'alphadata',imrotate(alpha_template,90));
    axis equal
    axis off
    colormap(ax1,'gray');
    weights=squeeze(targetWeights3D(:,:,sliceToShow)); 
    ax2 = axes;
    alphamap = abs(weights);
    %max(alphamap(:))
    alphamap = alphamap./max(alphamap(:));
    imagesc(ax2,imrotate(weights,90),'alphadata',imrotate(alphamap,90));
    title('Target weights')
    axis equal
    axis off
    %caxis([min_2d_for_axis   max_2d_for_axis]); 
    set(gcf,'Visible','on')
    colormap(ax2,'parula');
    %colormap(ax2,'turbo');
    linkprop([ax1 ax2],'Position');
    c = colorbar();
    drawnow
    
    % Get the color data of the object that correponds to the colorbar
    cdata = c.Face.Texture.CData;
    
    colorbar_values=linspace(c.Limits(1),c.Limits(2),256);
    max_abs_colorbar_values=max(abs(c.Limits(1)),abs(c.Limits(2)));
    cdata(4,:)=uint8((abs(colorbar_values)./max_abs_colorbar_values).*255);
    % Ensure that the display respects the alpha channel
    c.Face.Texture.ColorType = 'truecoloralpha';
    % Update the color data with the new transparency information
    c.Face.Texture.CData = cdata;
    
    drawnow
    
    % Make sure that the renderer doesn't revert your changes
    c.Face.ColorBinding = 'discrete';
    drawnow
    pause(1)
    clear('c')

end
 

%% save trained model

model.W=single(W);
model.W_original = single(W_original);
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




save([saveModelPath, '/', nameSavedModel, '.mat'],'model','-v7.3')

return




