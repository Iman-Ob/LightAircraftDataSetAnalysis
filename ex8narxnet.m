data=[delta1.data(:,1) P1.data(:,1) dth1.data(:,1) V1.data(:,1) gamma1.data(:,1) teta1.data(:,1) q1.data(:,1) alpha1.data(:,1) alphadot1.data(:,1) h1.data(:,1) hdot1.data(:,1) t1.data(:,1)];
%1,2,,3,,,,4,,,,5,,,6,,,7,,,,,8,9
%t,v,alpha,teta,q,h,hdot,delta,delta_th
path='D:\Study\Code Matlab\ex9JoystickAutopilot';
files = dir (strcat(path,'\*.mat'))

L = length (files);

for i=1:L
   Xtrain{i}=load(strcat(path,'\',files(i).name));
   XTrain(i)=struct2cell(Xtrain{i});
   sequenceLengths(i) = size(XTrain{1,i},1);
end


data=[XTrain{1}.data;XTrain{2}.data];

inTrain=data(1:8500,[3,4,5,7,8]);
outTrain=data(1:8500,[3,4,5,7]);

inTrain = con2seq(inTrain');
outTrain = con2seq(outTrain');

net = narxnet(1:3,1:3,100);
[Xs,Xi,Ai,Ts] = preparets(net,inTrain,{},outTrain);
net = train(net,Xs,Ts,Xi,Ai);
[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y); 

[netc,Xic,Aic] = closeloop(net,Xf,Af);
%view(netc)

test=data(8501:8681,:);
Xnew=con2seq(test(1:3,[3,4,5,7,8])');
Ynew=con2seq(test(1:3,[3,4,5,7])');
[Xs,Xi,Ai,Ts,EWs,shift] = preparets(netc,Xnew,{},Ynew);
 
%delta P V gamma teta q alpha alphadot h hdot t

xTest=test(1:20,[3,4,5,7,8]); %th, delta
yTest=test(1:20,[3,4,5,7]); %t, hdot,h,teta,alpha,v

Xtest=con2seq(xTest');
y2 = netc(Xtest);
y2=cell2mat(y2);
rmse = sqrt(mean(yTest-y2').^2);
error=(yTest-y2')./yTest;
p=2;
pn="teta";
figure
subplot(2,1,1)
plot(yTest(:,p))
hold on
plot(y2(p,:)','.-')
hold off
legend(["Observed" "Predicted"])
ylabel(pn)
%title("Forecast pure normal sine")
subplot(2,1,2)
stem(y2(p,:)' - yTest(:,p))
xlabel("ms")
ylabel("Error")
title("RMSE = " + rmse(p))

%===========slow
%delta1 P1 dth1 V1 gamma1 teta1 q1 alpha1 alphadot1 h1 hdot1 t1
inTrain=data(1:300,2);
outTrain=data(1:300,[4,5,10]);

inTrain = con2seq(inTrain');
outTrain = con2seq(outTrain');

net = narxnet(1:2,1:2,60);
[Xs,Xi,Ai,Ts] = preparets(net,inTrain,{},outTrain);
net = train(net,Xs,Ts,Xi,Ai);
[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y); 

[netc,Xic,Aic] = closeloop(net,Xf,Af);


test=data(300:303,:);
Xnew=con2seq(test(1:2,2)');
Ynew=con2seq(test(1:2,[4,5,10])');
[Xs,Xi,Ai,Ts,EWs,shift] = preparets(netc,Xnew,{},Ynew);
 
%delta P V gamma teta q alpha alphadot h hdot t

xTest=test(2:3,2); %th, delta
yTest=test(2:3,[4,5,10]); %t, hdot,h,teta,alpha,v

Xtest=con2seq(xTest');
y2 = netc(Xtest);
y2=cell2mat(y2);
rmse = sqrt(mean(yTest-y2').^2);

%delta1 P1 dth1 V1 gamma1 teta1 q1 alpha1 alphadot1 h1 hdot1 t1
p=3;
pn="h";
figure
subplot(2,1,1)
plot(yTest(:,p))
hold on
plot(y2(p,:)','.-')
hold off
legend(["Observed" "Predicted"])
ylabel(pn)
title("Forecast pure normal sine")
subplot(2,1,2)
stem(y2(p,:)' - yTest(:,p))
xlabel("ms")
ylabel("Error")
title("RMSE = " + rmse(p))

