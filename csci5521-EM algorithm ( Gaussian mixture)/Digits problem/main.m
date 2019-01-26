%%
percent = 0.9;

TrainingData=csvread(['.\Digits089.csv']);
flags = TrainingData(:, 1);
labels = TrainingData(:, 2); % indices of observations in class 1.
pixels = TrainingData(:, 3:end); % indices of observations in class 2.

train_idx = flags <= 4;
test_idx = flags == 5;

% preprocessing
pixels_pca = preproc(pixels(:, :), percent);
% train_pca = preproc( pixels(train_idx, :));
% test_pca = preproc( pixels(test_idx, :));


% train the clustering_machine
idx_init = [1, 1000, 1001, 2000, 2001, 3000];
ktot = length(idx_init);
centroid_init = pixels_pca(idx_init, :);

train = pixels_pca(train_idx, :);
test = pixels_pca(test_idx, :);
[cluster_idx, centroid_out, StepCount] = DoKmeans(train, centroid_init);

% validation
% label_dict = [0, 8, 9];
% vs = [labels(train_idx), cluster_idx];
% confusionM = zeros(ktot,3);
% for c = 1:ktot
%     idx_temp = vs(vs(:, 2) == c, 1);
%     for labl_i = 1:3
%         confusionM(c, labl_i) = sum(idx_temp == label_dict(labl_i));
%     end
% end
confusionM = GetConfusionMatrix(labels(train_idx), cluster_idx);
disp('confusion matrix:');
disp(confusionM);
[max_count, ind] = max(confusionM, [], 2);

err = 1 - sum(max_count)/sum(sum(confusionM));

disp('error rate: ');
disp(err);
disp(['StepCount = ',num2str( StepCount)]);











