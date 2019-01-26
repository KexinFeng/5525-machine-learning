function [assignments, centers, StepCount] = DoKmeans(data, InitialCenters)
centroid_init = double(InitialCenters);
data = double(data);

[N, D ] = size(data);
[ktot, ~] = size(centroid_init);

epsilon = 1e-3;

cent = centroid_init;
count = 0; 
while 1
    count = count + 1;
    disp(['count = ', num2str(count)]);
    
    [num_cl, D_query] = size(cent);
    [N_query, ~] = size(data);
    
    if any([num_cl ~=  ktot, D_query ~= D, N_query ~= N])
        error('bug');
    end
    
    % E step: asign cluster
    dist = distance(data, cent);
    [mins, cluster_id] = min(dist, [], 2);

    
    % M step: find centroid
    cent_new = [];
    for k = 1:ktot
        cluster = data(cluster_id == k, :);

        if isempty(cluster)
            % 选所有的点里最远的点
            temp = mins;
            temp(cluster_id > k) = 0;
            [~, ind] = max(temp);
            cluster_id(ind) = k;
            cluster = data(ind, :);
            
            
            
        end
        

        cent_new = [cent_new; mean(cluster, 1)];

%         cent_new = [cent_new; mean(data(cluster_idx == k, :), 1)];
    end
    

    if allclose(cent_new, cent, epsilon)
        break;
    end
    
    cent = cent_new;    

end

output = cluster_id;

StepCount = count;
assignments = cluster_id;
centers = cent_new;

% assignments: N * 1, with number indicating the cluster assigned to 
% centers: K * D

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