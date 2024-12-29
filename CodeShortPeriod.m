global net;
load('D:\Study\Code Matlab\ex3\data training\net.mat');
load('D:\Study\Code Matlab\ex2\data training\trainMat.mat');
fWindow=trainMat;
%divide data into train and test data
numTimeStepsTrain = floor(0.9*size(fWindow,1));
dataTrain = fWindow(1:numTimeStepsTrain+1,:);
dataTest = fWindow(numTimeStepsTrain+1:end,:);

%standardize training data
m=max(dataTrain);
dataTrainStandardized=dataTrain ./ m;
% mu = mean(dataTrain);
% sig = std(dataTrain);
%dataTrainStandardized = (dataTrain - mu) ./ sig;

XTrain = dataTrainStandardized(1:end-1,:);
YTrain = dataTrainStandardized(2:end,:);
 
%standardize testing data using the same parameters mu, sig
% dataTestStandardized = (dataTest - mu) ./ sig; 
dataTestStandardized=dataTest./m;
XTest = dataTestStandardized(1:end-1,:);
YTest = dataTest(2:end,:);
%dataTestStandardized = dataTest
%Network Architecture
numFeatures = 4;
numResponses = 4;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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
    
%Train Network
[net,info] = trainNetwork(XTrain',YTrain',layers,options);
%Initializing the network using the training data
net = predictAndUpdateState(net,XTrain');
%Predict using the last value in the predicted training data
[net,YPred] = predictAndUpdateState(net,YTrain(end,:)');
   
%Prediction when we have access to actual observed data
%net = resetState(net);
net = predictAndUpdateState(net,XTrain');
YPred = [];
numTimeStepsTest = size(XTest,1);
XTest=XTest';
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
    %h(:,i)=net.Layers(2,1).HiddenState;
    %w(:,i)=net.Layers(3,1).Weights;
end
   
%YPred = sig.*YPred' + mu;
%YPred=m .* YPred;
rmse = sqrt(mean((YPred'-YTest).^2));
e=abs(YPred-YTest)./YTest;
anomal=(e> YTest.*0.5);
anomaly=YTest.*0.1;

YPred=YPred';
figure
subplot(2,1,1)
plot(YTest(:,2))
hold on
plot(YPred(:,2),'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("teta")
title("Forecast with Updates")
subplot(2,1,2)
stem(YPred(:,2) - YTest(:,2))
xlabel("ms")
ylabel("Error")
title("RMSE = " + rmse(2))

%==================================================================
trainMat=[psine5; psine3; nsine5; pstep5; pstep3; nstep5; nstep3]
%test on SineWithNoiseGap New Data

step=[teta.data(:,1) delta.data(:,1) vel.data(:,1) acc.data(:,1)]

testMat=psine5;
update=testMat(1:300,:);
test=testMat(301:400,:);

mut = mean(testMat);
sigt = std(testMat);
dataTestStandardized = (test - mut) ./ sigt;

dataupdateStandardized = (update - mut) ./ sigt;

net = predictAndUpdateState(net,dataupdateStandardized');
%net = predictAndUpdateState(net,XTrain');

YPred = [];
numTimeStepsTest = size(dataTestStandardized,1);
XTest=dataTestStandardized';

%test with access to the test data
% for i = 1:numTimeStepsTest
%     [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
% end

%test without access to the test data
[net,YPred] = predictAndUpdateState(net,update(end,:)');
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

rmse = sqrt(mean((YPred'-test).^2));
anomal=(rmse> 1);
% e=abs(YPred-test)./test;
% anomal=(e> test.*0.3);
% anomaly=test.*0.1;
YPred=YPred';
figure
subplot(2,1,1)
plot(test(:,2))
hold on
plot(YPred(:,2),'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("teta")
title("Forecast Normal Step With Gap")
subplot(2,1,2)
stem(YPred(:,2) - test(:,2))
xlabel("ms")
ylabel("Error")
title("RMSE = " + rmse(2))


figure
plot(x,n,'g',x,f,'b-.');
legend('normal','failure')