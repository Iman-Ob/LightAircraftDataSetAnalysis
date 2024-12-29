data=[delta.data(:,1) deltath.data(:,1) V.data(:,1) gamma.data(:,1) teta.data(:,1) q.data(:,1) alpha.data(:,1) alphadot.data(:,1) h.data(:,1) hdot.data(:,1) t.data(:,1)];
v2=data(:,3).^2;
tstep=data(2,end)-data(1,end);
qdot=diff(data(:,6))/(tstep);
data(1,:)=[];
v2(1,:)=[];
da=[data(:,1) data(:,3) v2 data(:,5) data(:,6) qdot data(:,7) data(:,8) data(:,9)];
y = detrend(data(:,9));
da(:,9)=y;
data=[da(2:end,:) da(1:end-1,:)];

numTimeStepsTrain = floor(0.8*size(data,1));
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);

mu = mean(data);
sig = std(data);
datastd=(data-mu) ./ sig;

%data=(data-min(data))./(max(data)-min(data));
XTrain = datastd(1:end-1,:);
YTrain = datastd(2:end,1:9);


numFeatures = 18;
numResponses = 9;
numHiddenUnits = 18;

 % fullyConnectedLayer(80)
  %  dropoutLayer(0.01)
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.01)
    fullyConnectedLayer(numResponses)
    regressionLayer];
%
maxEpochs = 200;
miniBatchSize = 500;
%solver adam, sgdm
%'MiniBatchSize',miniBatchSize, ...
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',0.01, ...
    'InitialLearnRate',0.01, ...
    'LearnRateDropFactor',0.1, ...
    'Shuffle','never', ...
    'Plots','training-progress');

net = trainNetwork(XTrain',YTrain',layers,options);
%dataTest=data;

%dataTest=(dataTest-min(datastd))./(max(datastd)-min(datastd));
dataTestStandardized = (dataTest - mu) ./ sig;
XTest = dataTestStandardized(1:end-1,:);
YTest = dataTestStandardized(2:end,1:9);

net = predictAndUpdateState(net,XTest(1:2,:)');
[net,YPred] = predictAndUpdateState(net,XTest(2,:)');

numTimeStepsTest = size(XTest,1);
for i = 2:numTimeStepsTest
   if mod(i,100)==0
       resetState(net);
       [net,YPred(:,i)] = predictAndUpdateState(net,XTest(i-1,:)');
   else     
       xt=[YPred(:,i-1)' XTest(i,10:end)];
       [net,YPred(:,i)] = predictAndUpdateState(net, xt','ExecutionEnvironment','cpu');
       %resetState(net);
   end
end
YPred=YPred';

rmse = sqrt(mean((YPred-YTest).^2));
stderror = std( YTest ) / sqrt( size(YTest,1) );
ForecastFI1 = YTest - 2*stderror;
ForecastFI2 = YTest + 2*stderror;
anomal=(abs(YPred-YTest)>abs(3*stderror));
text=["delta","v","v2","teta","q","qdot","alpha","alphadot","h"];

for j=1:9
figure
subplot(2,1,1)
plot(YTest(:,j))
hold on
plot(ForecastFI1(:,j),'k--');
hold on
plot(ForecastFI2(:,j),'k--');
hold on
plot(YPred(:,j),'r.-');
hold off
legend(["Observed" "conf1" "conf2" "Predicted"])
ylabel(text(j))
title("Forecast")
subplot(2,1,2)
%stem(YPred(:,j) - YTest(:,j))
plot(anomal(:,j).*abs((YPred(:,j) - YTest(:,j))))
xlabel("ms")
ylabel("anomaly")
title("RMSE = " + rmse(j))
end

inputWeights=net.Layers(2,1).InputWeights(1:65,:);
outputWeights=net.Layers(2,1).InputWeights(196:end,:);
fullyConnectedWeights=net.Layers(4,1).Weights';
%B = lasso(inputWeights,outputWeights(:,1));
PredictorNames={'delta','v','v2','teta','q','qdot','alpha','alphadot','h','delta1','v1','v21','teta1','q1','qdot1','alpha1','alphadot1','h1'};
[B,FitInfo] = lasso(inputWeights(:,[1:3,5:12,14:18]),inputWeights(:,4),'CV',10,'PredictorNames',PredictorNames(:,[1:3,5:12,14:18]));
idxLambdaMinMSE = FitInfo.IndexMinMSE;
minMSEModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0)
idxLambda1SE = FitInfo.Index1SE;
sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0)
