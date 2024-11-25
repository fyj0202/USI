%%%% Image Size %%%% ↓
inputsize=[192 192 1];

%%%% Model File %%%% ↓
unet=model(inputsize,0);

%%%% Hyperparameter Settings %%%% % ↓
options = trainingOptions('adam', ...
'MiniBatchSize',64, ...
'Shuffle','every-epoch', ...
'ValidationData',varSet, ...
'MaxEpochs',150, ...
'Plots','training-progress', ... 
'InitialLearnRate',0.0005, ...
'LearnRateDropFactor',0.90, ...
'LearnRateDropPeriod', 15, ...
'L2Regularization',0.02, ...
'Verbose',true, ...
'VerboseFrequency', 546, ... 
'ValidationFrequency',546, ...
'ExecutionEnvironment','gpu');
%%%% Details of the training process %%%% ↓
[net,netinfo]=trainNetwork(trainSet,'labelimg',unet,options);