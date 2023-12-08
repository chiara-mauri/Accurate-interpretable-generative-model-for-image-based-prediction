function [W, V, beta, timeTraining, logMLVector, nIterations] = trainingModel(trainImages, trainBasisFun, nLatentVars, costTol, nItMax, nItMin)


Ntrain = size(trainImages,1);
nVoxels = size(trainImages,2);
nBasisFun = size(trainBasisFun,2);

%compute W

D_W = trainImages'*trainBasisFun;
A_W = trainBasisFun'*trainBasisFun;
W = (A_W'\D_W')';

%initialization of beta and V

betaInit = zeros(nVoxels,1);
nLoop = ceil(nVoxels/20000);
for i = 1:nLoop
    indeces= 1+(i-1)*20000:min(i*20000,nVoxels);
    betaInit(indeces)=(1./(var(trainImages(:,indeces))))';
end

rng(1)
Vinit=randn(nVoxels,nLatentVars); %randomly sampled from standard Gaussian;


beta = betaInit;
V=Vinit;


logML = inf;
logMLOld = inf;

logMLVector = zeros(1,nItMax);

tic
for nIt = 1:nItMax
    if (nIt > nItMin)
        if (abs((logML - logMLOld)/logML) < costTol)
            V = Vold;
            beta = betaOld;

            break;

        end
    end
    fprintf('it number: %d\n', nIt)

    %save old values

    Vold = V;
    betaOld = beta;
    logMLOld = logML;

    logMLVector(nIt) = logML;

    Delta = spdiags(beta,0,nVoxels,nVoxels);
    
 
    %%

    %compute posterior mean and variance over latent variables
    
    sigmaZinv = eye(nLatentVars)+V'*Delta*V;
    sigmaZ = inv(sigmaZinv);

    dimBlock = 4000;
    nBlocksTot = ceil(Ntrain/dimBlock);
   
    muZ = zeros(Ntrain,nLatentVars);
    for nBlock = 1:nBlocksTot
        indecesBlock = (nBlock-1)*dimBlock+1:min(nBlock*dimBlock,Ntrain);
        Mat = trainImages(indecesBlock,:)-trainBasisFun(indecesBlock,:)*W';
        muZ(indecesBlock,:) = Mat*Delta*V*sigmaZ;

    end
    clear('Mat')


    %compute log marginal likelihood with Cholesky decomposition

    U = chol(sigmaZinv); %detSigmaZninv=det(U)^2;
    logDetSigmaZinv = 2*sum(log(diag(U)));
   
    
    dimBlock = 1000;
    nBlocksTot = ceil(Ntrain/dimBlock);
    sumCell = cell(nBlocksTot,1);

    for nBlock = 1:nBlocksTot
        indecesBlock = (nBlock-1)*dimBlock+1:min(nBlock*dimBlock,Ntrain);
        Mat = trainImages(indecesBlock,:)-trainBasisFun(indecesBlock,:)*W';
        sum1 = trace(Mat*Delta*Mat');
        sum2 = trace((Mat*Delta*V)*sigmaZ*(V'*Delta*Mat'));
        sumCell{nBlock} = sum1-sum2;
    end
    clear('Mat')
    sumLogML = sum(cell2mat(sumCell));

    logML = -0.5*(Ntrain*nVoxels*log(2*pi)-Ntrain*sum(log(beta))+Ntrain*logDetSigmaZinv+sumLogML);
    fprintf('logML: %.6e\n',logML)

  

    %update of V
    
    A = muZ'*muZ+Ntrain.*sigmaZ;
    
    %D = (trainImages'-W*trainBasisFun')*muZ;
    %V = transpose(A'\D');
    
    dimBlock = 20000;
    nBlocks = ceil(nVoxels/dimBlock);
    V = zeros(nVoxels,nLatentVars);
    for i = 1:nBlocks
        indecesBlock=1+(i-1)*dimBlock:min(i*dimBlock,nVoxels);
        D_partial=(trainImages(:,indecesBlock)'-W(indecesBlock,:)*trainBasisFun')*muZ;
        V(indecesBlock,:)=transpose(A'\D_partial');
        
    end
    clear('A')
    clear('D_partial')

    %update of beta (parallel loop)
 

    %inizialize values for parallel loop
    voxelPerBatch = max(floor(1500/(nBasisFun+nLatentVars)),1);
    nBatches = ceil(nVoxels/voxelPerBatch); %n. of batches used for updates during training
    betaPar = cell(nBatches,1);
    Wpar = cell(nBatches,1);
    Vpar = cell(nBatches,1);
    imTrainPar = cell(1,nBatches);

    for i = 1:nBatches
        voxelInd = 1+(i-1)*voxelPerBatch:min(nVoxels,i*voxelPerBatch);%indixes of voxels of current batch
        imTrainPar{i} = trainImages(:,voxelInd); %train images of the batch (voxels in the batch)
        Wpar{i} = W(voxelInd,:);
        Vpar{i} = V(voxelInd,:);

    end



    %parallel loop for computing beta
    parfor i = 1:nBatches

        %compute elements of beta in each batch
        betaPar{i} = Ntrain./...
            ((sum((imTrainPar{i}-trainBasisFun*Wpar{i}'-...
            muZ*Vpar{i}').^2))'+diag(Vpar{i}*Ntrain*sigmaZ*Vpar{i}'));

    end
    beta = cell2mat(betaPar);



end 

timeTraining = toc;
nIterations = nIt-1;

fprintf('number of iterations: %d\n', nIterations)
fprintf('training time: %.2f s\n', timeTraining)
        
return 




