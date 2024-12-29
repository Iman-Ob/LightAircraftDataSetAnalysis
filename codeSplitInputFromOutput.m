y = downsample(trainMat,2);
y=trainMat;
%with standaradization
mu = mean(y);
sig = std(y);
trainMatStd = (y - mu) ./ sig;

%without stanardization
inTrain=y(:,[1,3,4]);
outTrain=y(:,2);

numFeatures = 3;
numResponses = 1;
numHiddenUnits = 200;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% maxEpochs = 100;
% miniBatchSize = 100;
% 
% options = trainingOptions('adam', ...
%     'MaxEpochs',maxEpochs, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'InitialLearnRate',0.01, ...
%     'GradientThreshold',1, ...
%     'Shuffle','never', ...
%     'Plots','training-progress',...
%     'Verbose',0);

net = trainNetwork(inTrain',outTrain',layers,options);

%=============================================
t=nstep3;
mut = mean(t);
sigt = std(t);
dataTestStandardized = (t - mut) ./ sigt;

% inTest=dataTestStandardized(1:400,[1,3,4]);
% outTest=dataTestStandardized(1:400,2);

inTest=t(1:400,[1,3,4]);
outTest=t(1:400,2);


[net,YPred] = predictAndUpdateState(net,inTest(1,:)');
YPred = predict(net,inTest','MiniBatchSize',1);
rmse = sqrt(mean((YPred'-outTest).^2));
anomal=(rmse> 0.1);
YPred=YPred';
figure
subplot(2,1,1)
plot(outTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("teta")
title("Forecast Normal Step")
subplot(2,1,2)
stem(YPred - outTest)
xlabel("ms")
ylabel("Error")
title("RMSE = " + rmse)