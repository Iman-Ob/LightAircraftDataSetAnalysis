function y= shpt(u)

    persistent net
    y=zeros(5,1);
    %stanadardize data
%     mu=zeros(1,4);
%     mat=zeros(40,4);
      mat=u;
%     mu = mean(mat);
%     if mu>0
%         sig = std(mat);
%         ustd = (mat - mu) ./ sig;
%     else
      ustd=mat;
%     end
    if isempty (net)
        if exist('D:\Study\Code Matlab\ex3\data training\net.mat', 'file')
            % Recall a from mat file
            load('D:\Study\Code Matlab\ex3\data training\net.mat');
            net = predictAndUpdateState(net,ustd');
        end
    else
   
    %ustd=u;
%   predict with access to actual data
%    net = predictAndUpdateState(net,ustd');
    YPred = [];
    numTimeStepsTest = size(ustd,1);
    ustd=ustd';
    for i = 1:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,ustd(:,i),'ExecutionEnvironment','GPU');
    end
    y=double(YPred(4,:)');
    end
    
        %test without access to the test data
%     [net,YPred] = predictAndUpdateState(net,u(end,:)');
%     a=size(u,1);
%     for i = 2:a
%         [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','GPU');
%     end
end
