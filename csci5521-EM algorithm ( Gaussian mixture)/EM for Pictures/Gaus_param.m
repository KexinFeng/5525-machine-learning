function [pi, mu, sigma] = Gaus_param(data, gamma)
% data: N * D
% pi: K * 1
% mu: K * D
% sigma: D * D
% gamma: N * K

[N, D ] = size(data);
[N, K ] = size(gamma);

if ~allclose(sum(gamma, 2), ones(N, 1), 1e-10)
    error('gamma not normalized');
end

pi = reshape(sum(gamma, 1), [], 1) / N;
if ~allclose(sum(sum(gamma, 1)),N)
    error('gamma sum not equal to N')
end


sigma0 = zeros(D, D, K);
for k = 1:K
    
    Nk = sum(gamma(:, k), 1);
   	
    mu(k, :) = sum(data .* gamma(:, k), 1)/ Nk;
       
    sigma0(:, :, k) = (data - mu(k, :))'* diag(gamma(:, k)) * (data - mu(k, :)) ;
end

sigma = zeros(D);
for k = 1:K
    sigma = sigma + sigma0(:, :, k) / N;
end



end