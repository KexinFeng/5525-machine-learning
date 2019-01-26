function [ConfusionMatrix]=GetConfusionMatrix(TrueLabels,Assignments)
vs = [TrueLabels, Assignments];
label_dict = [0, 8, 9];
% vs = [labels(train_idx), cluster_idx];
ktot = max(Assignments);
if ktot ~= 4
    warning('ktot ~= t');
end

confusionM = zeros(ktot,3);
for c = 1:ktot
    idx_temp = vs(vs(:, 2) == c, 1);
    for labl_i = 1:3
        confusionM(c, labl_i) = sum(idx_temp == label_dict(labl_i));
    end
end
ConfusionMatrix = confusionM;
end