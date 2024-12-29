%data=[th.data(:,1) delta.data(:,1) t(:,1) hdot.data(:,1) h.data(:,1) teta.data(:,1) q.data(:,1) alpha.data(:,1) v.data(:,1)];
data=[delta.data(:,1) P.data(:,1) dth.data(:,1) V.data(:,1) gamma.data(:,1) teta.data(:,1) q.data(:,1) alpha.data(:,1) alphadot.data(:,1) h.data(:,1) hdot.data(:,1) t.data(:,1)];

%08,09: t,v,alpha,teta,q,h,hdot,delta,delta_th
%data=data(:,[3,4,5,7,8]);

data=struct2cell(data);
data=data{1}.data;
data(:,12)=[];
mu = mean(data);
sig = std(data);
data=(data-mu) ./ sig;
inTrain=data(1:3000,:);
outTrain=data(1:3000,:); %hdot, h,teta,q,v

numFeatures = 11;
numResponses = 11;
numHiddenUnits = 50;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(100)
    dropoutLayer(0.01)
    fullyConnectedLayer(numResponses)
    regressionLayer];
%
maxEpochs = 200;
miniBatchSize = 120;
%solver adam, sgdm
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',0.1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateDropFactor',0.1, ...
    'Shuffle','never', ...
    'Plots','training-progress');

net = trainNetwork(inTrain',outTrain',layers,options);

net = resetState(net);
t=data(3000:end,:);

outTest=t;
mux = mu;
sigx = sig;
%mut=mean(t);
%sigt=std(t);
dataTestStandardized = (t - mux) ./ sigx;
inTest=t;
%outTest=dataTestStandardized(:,1);

[net,YPred] = predictAndUpdateState(net,inTest');

%YPred = predict(net,inTest','MiniBatchSize',150);

err=(YPred'-outTest)./outTest;
rmse = sqrt(mean((YPred'-outTest).^2));
anomal=(rmse> 0.1);
%alpha, tetq,q,hdot
p=1;
pn="alph";
figure
subplot(2,1,1)
plot(outTest(:,p))
hold on
plot(YPred(p,:)','.-')
hold off
legend(["Observed" "Predicted"])
ylabel(pn)
title("Forecast pure normal sine")
subplot(2,1,2)
stem(YPred(p,:)' - outTest(:,p))
xlabel("ms")
ylabel("Error")
title("RMSE = " + rmse(p))


%===========slow=========
%data=[delta1.data(:,1) P1.data(:,1) dth1.data(:,1) V1.data(:,1) gamma1.data(:,1) teta1.data(:,1) q1.data(:,1) alpha1.data(:,1) alphadot1.data(:,1) h1.data(:,1) hdot1.data(:,1) t1.data(:,1)];

inTrain=data(1:300,[2,4,5,10]);
outTrain=data(1:300,[4,5,10]); %hdot, h,teta,q,v

mu = mean(inTrain);
sig = std(inTrain);
inTrain=(inTrain-mu) ./ sig;
 
numFeatures = 4;
numResponses = 3;
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(100)
    dropoutLayer(0.05)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = 300;
miniBatchSize = 100;
%solver adam, sgdm
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateDropPeriod',100, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.01, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

net = trainNetwork(inTrain',outTrain',layers,options);
