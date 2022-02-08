function [YPred, YPredPre]= cyclePredictWithRNN(dataStd, startPoint, numPredSteps, net)
%   cyclePredictionWithRNN 循环使用模型进行预测
%   Input:
%   --dataStd:正则化后的训练+验证数据
%   --StartPoint：预测起点
%   --numPredSteps：预测步数
%   --net：训练好的网络
%   Output:
%   --YPed：所有预测值
%   --YPredPre：当前时刻预测值


    inputDim = size(dataStd,1);
    outputDim = net.Layers(end-1).OutputSize;
    XTestPre = dataStd(:,1:startPoint-1);
    net = resetState(net);
    [net,YPredPre] = predictAndUpdateState(net,XTestPre); % update net to startPoint-1
    YPredPre = YPredPre(1:3,:);
    xTest = dataStd(:,startPoint);
    if inputDim==3 % prediction with temporal correlation LSTM
        outputSteps = outputDim/inputDim;
        YPred = zeros(inputDim,ceil(numPredSteps/outputSteps)*outputSteps);
        for cyc = 1:ceil(numPredSteps/outputSteps)
            [net, yPred] = predictAndUpdateState(net,xTest); % output the prediction results at xTest
            xTest = reshape(yPred(:,end),[],outputSteps); % setting prediction results at the last point as the input of the next predicition cycle 
            YPred(:,(cyc-1)*outputSteps+(1:outputSteps)) = xTest; % saving the prediction results at current cycle
        end
        YPred = YPred(:,1:numPredSteps);
    else % prediction with spatial-temporal correlation LSTM
        YPred = zeros(outputDim,numPredSteps);
        for cyc = 1:numPredSteps
            [net, YPred(:,cyc)] = predictAndUpdateState(net,xTest); % output the prediction results at xTest
            xTest = [dataStd(1:end-outputDim,startPoint+cyc); YPred(:,cyc)];
        end
    end  
end