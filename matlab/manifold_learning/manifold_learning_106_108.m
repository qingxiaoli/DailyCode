%% this script is test of Dmaps and MVU to dimensionality reduction
% coder: Jie An
% version: 20170411
% bug_submission: pkuanjie@gmail.com

%% data load and add path
load('umist_cropped.mat');
addpath(genpath('./drtoolbox/'));

%% dimensional reduction
[x, y, z] = size(facedat{1});
data = zeros(z, x * y);
for i = 1 : z
    data(i, :) = im2double(reshape(facedat{1}(:, :, i), [1, x * y]));
end;
data_mapped = diffusion_maps(data, 2, 1, 1);
figure,
plot(data_mapped(:, 2), data_mapped(:, 1), 'ro','MarkerFaceColor','r');
title('diffusion map result');
[mappedX, mapping] = fastmvu(data, 2);
figure,
plot(mappedX(2, :), mappedX(1, :), 'ro','MarkerFaceColor','r');
title('MVU result');
