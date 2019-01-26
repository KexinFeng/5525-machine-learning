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
idx_init = [1, 1000, 1001, 2000];
ktot = length(idx_init);
centroid_init = pixels_pca(idx_init, :);

train = pixels_pca(train_idx, :);
test = pixels_pca(test_idx, :);
[N, D ] = size(train);

cluster_idx = kmeans(train, centroid_init);
% cluster_idx is the trained result.

% validation
label_dict = [0, 8, 9];
vs = [labels(train_idx), cluster_idx];
confusionM = zeros(ktot,3);
for c = 1:ktot
    idx_temp = vs(vs(:, 2) == c, 1);
    for labl_i = 1:3
        confusionM(c, labl_i) = sum(idx_temp == label_dict(labl_i));
    end
end
% disp('confusion matrix:');
% disp(confusionM);
% [max_count, ind] = max(confusionM, [], 2);
% 
% err = 1 - sum(max_count)/sum(sum(confusionM));
% 
% disp('error rate: ');
% disp(err);

%%%%%%%%%%
% data: N * D
% pi: K * 1
% mu: K * D
% sigma: D * D
% gamma: N * K

data = train;
target = labels(train_idx);

pi0 = zeros(ktot, 1);
mu0 = zeros(ktot, D);
sigma0 = zeros(D, D, ktot);
for k = 1:ktot
    idxk = (cluster_idx == k);
    
    Nk = sum(idxk);
    pi0(k) = Nk / N;
    
    muk = mean(data(idxk, :), 1);
    mu0(k, :) = muk;
    
    sigma0(:, :, k) = (data(idxk, :) - muk)'* (data(idxk, :) - muk) ./ Nk;
end

sigma_init = zeros(D);
for k = 1:ktot
    sigma_init = sigma_init + pi0(k) .* sigma0(:, :, k);
end

[NewMus,NewSigmas,NewPriors,JointProbs,Posteriors] = DoEM(data, mu0, sigma_init, pi0);


% size(JointProbs)

% size(Posteriors)

%%%%%%%%%%%%%%%%%%%%%
[~, cluster_id] = max(Posteriors);
confusionM = GetConfusionMatrix(labels(train_idx), cluster_idx);

disp('confusion matrix:');
disp(confusionM);
[max_count, ind] = max(confusionM, [], 2);

err = 1 - sum(max_count)/sum(sum(confusionM));

disp('error rate: ');
disp(err);











