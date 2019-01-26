function output = kmeans(data, centroid_init)

[N, D ] = size(data);
[ktot, ~] = size(centroid_init);

epsilon = 1e-12;

cent = centroid_init;
count = 0; 
while 1
    count = count + 1;
%     disp(['count = ', num2str(count)]);
    
    [num_cl, D_query] = size(cent);
    [N_query, ~] = size(data);
    
    if any([num_cl ~=  ktot, D_query ~= D, N_query ~= N])
        error('bug');
    end
    
    % E step: asign cluster
    dist = distance(data, cent);
    [~, cluster_idx] = min(dist, [], 2);
    
    % M step: find centroid
    cent_new = [];
    for k = 1:ktot
        cent_new = [cent_new; mean(data(cluster_idx == k, :), 1)];
    end
    
    % stopping condition
    if allclose(cent_new, cent, epsilon)
        break;
    end
    
    cent = cent_new;    

end

output = cluster_idx;

end

function d = distance(data, c)
[num_cl, ~] = size(c);
[N, ~] = size(data);
data_kron = kron(data, ones(num_cl, 1));
% c = reshape(c, 1, []);
c_kron = kron(ones(N, 1), c);
dd = sqrt(sum((data_kron - c_kron).^2, 2));
d = reshape(dd, num_cl, [])';
end