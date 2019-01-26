function [assignments, centers, StepCount]=DoKmeans(data,InitialCenters)
centroid_init = InitialCenters;

[N, D ] = size(data);
[ktot, ~] = size(centroid_init);

epsilon = 1e-12;

cent = centroid_init;
count = 0; 
cluster_id = [];
% last_cluster_id = zeros(N,1);
% last_cluster_id(init_cent_idx) = 1:ktot;

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
            
%             ind = randperm(N, 1);
%             cluster_id(ind) = k;
%             cluster = data(ind, :);

%             if ~isempty(last_cluster_id)  
%                 %  在last_cluster_id里随机选
% %                 temp = find(last_cluster_id == k);
% %                 ind = temp(randperm(length(temp), 1));
% 
%                 % 在last_cluster_id里选最远的点                
%                 indk = find(last_cluster_id == k);
%                 [~, ord] = max(mins(indk));
%                 ind = indk(ord);
%                                 
%                 cluster_id(ind) = k;
%                 cluster = data(ind, :);
%             else
%                 [~, ind] = max(mins);
%                 cluster_id(ind) = k;
%                 cluster = data(ind, :);
%             end
            
        end
        
%         for kid = 1:k
%             if isempty(find(cluster_id == kid))
%                 error('how come?')
%             end
%         end
%         
%         if isempty(cluster)
%             error('WTF');
%         end
%         
        cent_new = [cent_new; uint8(mean(cluster, 1))];
    end
    
%     for k = 1:ktot
%         if isempty(find(cluster_id == k))
%             error('how come?')
%         end
%     end
    
    if allclose(cent_new, cent, epsilon)
        break;
    end
    
    cent = cent_new;  
%     last_cluster_id = cluster_id;
    
%     for k = 1:ktot
%         if isempty(find(last_cluster_id == k))
%             error('how come?')
%         end
%     end

end


StepCount = count;
centers = cent_new;
assignments = cluster_id;

end

function d = distance(data, c)
dtype = class(c);
[num_cl, ~] = size(c);
[N, ~] = size(data);
data_kron = kron(data, ones(num_cl, 1, dtype));
% c = reshape(c, 1, []);
c_kron = kron(ones(N, 1, dtype), c);
dd = sqrt(sum((data_kron - c_kron).^2, 2));
d = reshape(dd, num_cl, [])';

% cos distance
% data: n * 3
% c: k * 3
% d: n * k

% dd = zeros(N, num_cl, dtype);
% for k = 1:num_cl
%     prod = sum(data .* c(k, :), 2);
%     norm_data = sqrt(sum(data.^2, 2));
%     temp = prod ./ norm_data / sqrt(sum(c(k, :).^2));
%     dd(:, k) = reshape(temp, [], 1);
% end
% d = dd;
end