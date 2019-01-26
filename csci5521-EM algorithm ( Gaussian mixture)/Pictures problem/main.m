data = imread('goldy.ppm');
% data = imread('stadium.ppm');
[d1, d2, d3] = size(data);
% d1, d2, d3 are the digit unit, and d3 has the highest order.
% to test:

%  A = [    3     2;
%      5     6;
%     20    50;
%     30    89;
%     33    55;
%      7     8    ];

% preprocess:
data_2d = reshape(data, d1 * d2, []);
[N,num_f] = size(data_2d);

data_3d = reshape(data_2d, d1, d2, []);

figure;
imagesc(data_3d);

% initialize
ks = [3, 4, 7];
ktot = 3;
init_cent_idx = randperm(N, ktot);
centroid_init = data_2d(init_cent_idx, :);

% train 
[cluster_idx, cent, StepCount] = DoKmeans(data_2d, centroid_init);
cent = uint8(cent);
% validation
data_2d_clus = zeros(N, num_f, 'uint8');
for n = 1:N
    data_2d_clus(n, :) = cent(cluster_idx(n), :);
end
cent
data_3d_2 = reshape(data_2d_clus, d1, d2, 3);

figure
imagesc(data_3d_2);
all(all(all(logical(data_3d_2<= 255) .* logical(data_3d_2>=0))));

disp(['StepCount = ',num2str( StepCount)]);
















