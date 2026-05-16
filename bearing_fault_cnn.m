%% PROJECT 27: Deep Learning for Bearing Fault Classification
% 1D CNN trained on raw vibration signals
% Classifies: healthy, inner race fault, outer race fault

%--- Signal Generation ---
fs      = 12000;
t_end   = 0.1;
t       = 0:1/fs:t_end-1/fs;
N       = length(t);
n_each  = 300;
n_class = 3;

fprintf('--- Bearing Fault CNN Classifier ---\n')
fprintf('Sampling frequency : %d Hz\n', fs)
fprintf('Segment length     : %.1f seconds\n', t_end)
fprintf('Samples per class  : %d\n', n_each)
fprintf('Total samples      : %d\n', n_each*n_class)

%--- Generate Synthetic Bearing Signals ---
rng(42)
X_data = zeros(N, 1, n_each*n_class);
Y_data = zeros(n_each*n_class, 1);

for s = 1:n_each*n_class
    class_id = ceil(s/n_each);
    sig = 0.3*randn(1,N);

    if class_id == 1
        sig = sig + sin(2*pi*200*t) + 0.5*sin(2*pi*400*t);
    elseif class_id == 2
        f_inner = 162;
        phase   = 2*pi*rand;
        sig = sig + sin(2*pi*200*t) + ...
                    2*sin(2*pi*f_inner*t+phase).* ...
                    (1+0.5*sin(2*pi*20*t));
    else
        f_outer = 108;
        sig = sig + sin(2*pi*200*t) + ...
                    2.5*abs(sin(2*pi*f_outer*t));
    end

    sig = sig/max(abs(sig));
    X_data(:,1,s) = sig';
    Y_data(s)     = class_id;
end

%--- Train/Test Split ---
idx_all   = randperm(n_each*n_class);
n_train   = round(0.8*n_each*n_class);
idx_train = idx_all(1:n_train);
idx_test  = idx_all(n_train+1:end);

X_train = X_data(:,:,idx_train);
Y_train = categorical(Y_data(idx_train));
X_test  = X_data(:,:,idx_test);
Y_test  = categorical(Y_data(idx_test));

fprintf('\nTraining samples : %d\n', n_train)
fprintf('Test samples     : %d\n', length(idx_test))
fprintf('Signal length    : %d points\n', N)

%--- Reshape to cell array of sequences ---
X_train_cell = cell(1, n_train);
for i = 1:n_train
    X_train_cell{i} = squeeze(X_train(:,1,i))';
end

X_test_cell = cell(1, length(idx_test));
for i = 1:length(idx_test)
    X_test_cell{i} = squeeze(X_test(:,1,i))';
end

%--- Define 1D CNN Architecture ---
% Use smaller pooling to avoid size issues
layers = [
    sequenceInputLayer(1, 'Name', 'input', 'MinLength', N)
    convolution1dLayer(16, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    convolution1dLayer(8, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    convolution1dLayer(4, 128, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    globalAveragePooling1dLayer('Name', 'gap')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.3, 'Name', 'drop')
    fullyConnectedLayer(n_class, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
    ];

%--- Training Options ---
options = trainingOptions('adam', ...
    'MaxEpochs',           30, ...
    'MiniBatchSize',       32, ...
    'InitialLearnRate',    0.001, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      {X_test_cell, Y_test}, ...
    'ValidationFrequency', 10, ...
    'Verbose',             true, ...
    'Plots',               'none');

fprintf('\nTraining 1D CNN...\n')
net = trainNetwork(X_train_cell, Y_train, layers, options);

%--- Evaluate ---
Y_pred = classify(net, X_test_cell);
acc    = sum(Y_pred == Y_test)/numel(Y_test)*100;

fprintf('\n--- Test Results ---\n')
fprintf('Test accuracy: %.1f%%\n', acc)

%--- Plot ---
figure;

subplot(1,2,1)
cm = confusionchart(Y_test, Y_pred);
cm.Title = sprintf('Confusion Matrix - Accuracy: %.1f%%', acc);
cm.RowSummary = 'row-normalized';

subplot(1,2,2)
hold on
class_names = {'Healthy','Inner Race','Outer Race'};
colors      = {'b','r','g'};
t_plot      = t(1:500);
for c = 1:3
    idx_c = find(Y_data==c,1);
    plot(t_plot*1000, X_data(1:500,1,idx_c)+(c-1)*3, ...
         colors{c}, 'LineWidth', 1.5, 'DisplayName', class_names{c})
end
legend('Location','best')
title('Sample Signals per Class')
xlabel('Time (ms)')
ylabel('Amplitude (offset)')
grid on
hold off

sgtitle('1D CNN Bearing Fault Classification')
saveas(gcf, 'bearing_fault_cnn.png')

fprintf('\n--- Class Characteristics ---\n')
fprintf('Healthy    : mesh harmonics only\n')
fprintf('Inner race : modulated at 162 Hz\n')
fprintf('Outer race : amplitude modulated at 108 Hz\n')
fprintf('CNN learned features automatically\n')