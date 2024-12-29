%delta deltath V gamma teta q alpha alphadot h hdot t
data(:,end)=[];
data(:,4)=[];

y = detrend(data(:,8));
data(:,8)=y;
dt=[data(2:end,:) data(1:end-1,:)];

numTimeStepsTrain = floor(0.8*size(dt,1));
dataTrain = dt(1:numTimeStepsTrain+1,:);
dataTest = dt(numTimeStepsTrain+1:end,:);

mu = mean(dt);
sig = std(dt);
datastd=(dt-mu) ./ sig;

XTrain = datastd(1:end-1,:);
YTrain = datastd(2:end,4:7);


numFeatures = 18;
numResponses = 4;
numHiddenUnits = 44;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.01)
    fullyConnectedLayer(numResponses)
    regressionLayer];
%
maxEpochs = 150;
miniBatchSize = 1024;
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


dataTestStandardized = (dataTest - mu) ./ sig;
XTest = dataTestStandardized(1:end-1,:);
YTest = dataTestStandardized(2:end,[4:7]);

net = predictAndUpdateState(net,XTest(1:2,:)');
[net,YPred] = predictAndUpdateState(net,XTest(2,:)');

numTimeStepsTest = size(XTest,1);
for i = 2:numTimeStepsTest
   if mod(i,20)==0
       resetState(net);
       [net,YPred(:,i)] = predictAndUpdateState(net,XTest(i-1,:)');
   else     
       xt=[XTest(i,(1:3)), YPred(1:end,i-1)',XTest(i,8:end)];
       [net,YPred(:,i)] = predictAndUpdateState(net, xt','ExecutionEnvironment','cpu');
       %resetState(net);
   end
end
YPred=YPred';
%anomal=(abs(YPred-YTest)>abs(3*MAD));
rmse = sqrt(mean((YPred-YTest).^2));
stderror = std( YTest ) / sqrt( size(YTest,1) );
ForecastFI1 = YTest - 2*stderror;
ForecastFI2 = YTest + 2*stderror;
anomal=(abs(YPred-YTest)>abs(3*stderror));
text=["teta","q","alpha","alphaDot"];

for j=1:4
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

