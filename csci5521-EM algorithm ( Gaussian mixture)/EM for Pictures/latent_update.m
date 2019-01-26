function gamma_nk = latent_update(data, pi, mu, sigma)
% data: N * D
% pi: K * 1
% mu: K * D
% sigma: D * D
% gamma: N * K
[N, D ] = size(data);
K = length(pi);

arg = zeros(N, K);
for k = 1:K
arg(:, k) = log(pi(k)) - reshape(0.5 * diag((data - mu(k, :)) *  pinv(sigma) * (data - mu(k, :))' ), [], 1); 
end

max_arg = max(arg, 2);
arg = arg - reshape(max_arg, N ,[]);

gamma_nk = exp(arg)./ sum(exp(arg), 2);
% The same as the Pr(cluster_j | x_i) 

if ~allclose(sum(sum(gamma_nk, 1)),N)
    error('gamma sum not equal to N')
end

end