%% this script is to constructe inilization level set function for image processing
function [level_set] = initialization_level_set(img)
% coder: Jie An
% version: 20170401
% bug_submission: pkuanjie@gmail.com

%% set up
% BORDER_OFFSET = 10;
[row, col] = size(img);

%% construction
[x, y] = meshgrid(1: row, 1: col);
level_set = sqrt((x-256).^2 + (y-256).^2) - 240;

% figure,
% mesh(level_set), axis image;
