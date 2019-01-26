function [gauss, log_norm_recip] = Gauss_dist(data, prior, mu, sigma)
% data: N * D
% prior: K * 1
% mu: K * D
% sigma: D * D
% gamma: N * K
% x: N * D

[N, D] = size(data);
K = length(prior);

x = data;
gauss = zeros(N, K);
maxes = zeros(1, K );
for k = 1:K
    arg =  0.5 * reshape(diag((x - mu(k, :)) * pinv(sigma) * (x - mu(k, :))'), [], 1);
    log_norm_recip = log(prior(k)) + D/2 * log( 1/(2 * pi))  + -0.5 * sum(log(eig(sigma)));
    arg_max = max(arg);
    maxes(k) = arg_max;
    gauss(:, k) = exp(arg - maxes(k));
end
global_max = max(maxes);
for k = 1:K
    gauss(:, k) = gauss(:, k) * exp( maxes(k) - global_max ) ; 
end