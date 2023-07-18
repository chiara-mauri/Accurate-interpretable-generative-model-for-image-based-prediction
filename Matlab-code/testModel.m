function [targetPredictions,correlation,MAE,RMSE] = testModel(testImages,testBasisFun,testTarget,V,beta,W,trainBasisFunMean,trainBasisFunStd)


    showFigures = 1;

    targetTrainMean = trainBasisFunMean(1);
    targetTrainStd = trainBasisFunStd(1);
    nLatentVars = size(V,2);
    nVoxels = size(V,1);
    nTest = size(testImages,1);
    
    wStar = W(:,1);
    
    Delta=spdiags(beta,0,nVoxels,nVoxels);
    SigmaZinv=eye(nLatentVars)+V'*Delta*V;
    SigmaZ=inv(SigmaZinv);
    
    
    
    DeltaTimesWstar = Delta*wStar;
    aux = V'*DeltaTimesWstar;
    %posteriorVarTarget=1/(1/priorVarTarget+wStar'*DeltaTimesWstar-aux'*SigmaZ*aux);
    posteriorVarTarget=1/(wStar'*DeltaTimesWstar-aux'*SigmaZ*aux);

    gap = testImages'-W(:,2:end)*testBasisFun(:,2:end)';
    aux2 = V'*Delta*gap;
    standTargetPredictions = posteriorVarTarget.*(DeltaTimesWstar'*gap-aux'*SigmaZ*aux2)';
   

    %take predictions back to the original target space
    targetPredictions = standTargetPredictions.*targetTrainStd+targetTrainMean;
    
    correlation=corr(testTarget,targetPredictions);
    RMSE=sqrt(sum((testTarget-targetPredictions).^2)/nTest);
    MAE=mean(abs(testTarget-targetPredictions));

    fprintf('test correlation : %.4f\n', correlation)
    fprintf('test RMSE : %.4f\n', RMSE)
    fprintf('test MAE : %.4f\n', MAE)
    fprintf('\n')

    if showFigures
        figure, scatter(testTarget,targetPredictions)
        hold on
        plot(testTarget,testTarget)
        xlabel('true score')
        ylabel('predicited score')
        title('Predictions')
        hold off
    end
   
    return

