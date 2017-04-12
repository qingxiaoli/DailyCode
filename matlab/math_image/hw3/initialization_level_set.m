%% this script is to constructe inilization level set function for image processing
function [level_set] = initialization_level_set(img)
% coder: Jie An
% version: 20170401
% bug_submission: pkuanjie@gmail.com

%% set up
% BORDER_OFFSET = 10;
[row, col] = size(img);

%% construction
% [x, y] = meshgrid(1: col, 1: row);
% level_set = sqrt((x - col / 2).^2 + (y - row / 2).^2) - (min(row, col) / 2 - 10);
level_set = zeros(row, col);
level_set(10: row - 10, [10, col - 10]) = 1;
level_set([10, row - 10], 10: col - 10) = 1;
level_set = bwdist(level_set);
level_set(10: row - 10, 10: col - 10) = -level_set(10: row - 10, 10: col - 10);

% figure,
% mesh(level_set), axis image;
