%%
function plot_digit(pixels)
    d = sqrt(length(pixels));
    picture = reshape(pixels, d, d);
    
    figure;
    heatmap(picture);
end























