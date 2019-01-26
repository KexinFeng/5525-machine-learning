function [NewMus,NewSigmas,NewPriors,JointProbs,Posteriors] ...
= DoEM(Xvec,InitMus,InitSigmas,InitPriors)
% data: N * D
% pi: K * 1
% mu: K * D
% sigma: D * D
% gamma: N * K

% function [pi_new, mu_new, sigma_new] = DoEM_origin(data, gamma_init)
data = Xvec;
pi0 = InitPriors;
mu0 = InitMus;
sigma_init = InitSigmas;

[N, D ] = size(data);


gamma_init = latent_update(data, pi0, mu0, sigma_init);
if ~allclose(sum(gamma_init, 2), ones(N, 1))
    error('gamma not normalized');
end

[~, K] = size(gamma_init);
epsilon = 1e-2;
count = 0; 

[pi, mu, sigma] = Gaus_param(data, gamma_init);

while 1
    count = count + 1;
    disp(['count = ', num2str(count)]);
   
    % E step: updata latent
    gamma_new = latent_update(data, pi, mu, sigma);
    
    if ~allclose(sum(gamma_new, 2), ones(N, 1))
        error('gamma not normalized');
    end
    
    % M step: Gaus_parameters
    [pi_new, mu_new, sigma_new] = Gaus_param(data, gamma_new);
    
    % termination condition
%     disp(norm(mu));
    if allclose(mu, mu_new, epsilon * norm(mu))
        break;
    end
    
    % update
    pi = pi_new; 
    mu = mu_new; 
    sigma = sigma_new; 
end

% pi_new, mu_new, sigma_new
NewMus = mu_new;
NewSigmas = sigma_new;
NewPriors = pi_new;



% JointProbs,Posteriors

pi_out = pi_new;
mu_out = mu_new;
sigma_out = sigma_new;

Posteriors = latent_update(data, pi_out, mu_out, sigma_out);




[gauss, log_inv_norm] = Gauss_dist(data, pi_out, mu_out, sigma_out);
JointProbs = gauss;


end