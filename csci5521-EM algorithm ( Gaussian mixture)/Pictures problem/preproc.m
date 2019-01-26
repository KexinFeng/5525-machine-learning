%%
function pixels_trunk = preproc(pixels_org, percent)
% when percent is int, it means the absolute truncating dimension.
[N, D] = size(pixels_org);

mu_pixel = mean(pixels_org, 1);
pixels = pixels_org - mu_pixel;

if percent ~= fix(percent)
    % svd method:
    [~,sv,v] = svd(pixels / sqrt(N) );
    s = diag(sv);
    s = sort(s.^2, 'descend');
    rate = cumsum(s/sum(s));
    dim = find(rate >= percent, 1);
    pixels_rec1 = pixels * (v(:, 1:dim) * v(:, 1:dim)') ;
    pixels_trunk = pixels * v(:, 1:dim);
    
    % eig method:
    % disp(['N = ',num2str(N)]);
    % S = 1/N * (pixels' * pixels);
    % [v, s] = eig(S);
    % s = sort(diag(s), 'descend');
    % rate = cumsum(s/sum(s));
    % dim = find(rate >= 0.9 , 1);
    % pixels_rec1 = pixels * (v(:, end - dim : end) * v(:, end - dim : end)') + mu_pixel ;
    % pixels_trunk = pixels * v(:, end - dim: end);
    
else
    [~,~,v] = svd(pixels / sqrt(N) );
    dim = percent;
    pixels_rec1 = pixels * (v(:, 1:dim) * v(:, 1:dim)') ;
    pixels_trunk = pixels * v(:, 1:dim);
end



% Verify by plotting and recovering to the original picture:
% plot_digit(pixels(20, :));
% plot_digit(pixels_rec1(20, :));


% pca not figured out yet...
% [coeff,score,latent] = pca(pixels_org(5,:));
% Xcentered = score*coeff';
% plot_digit(Xcentered);

end
































