%% this script is to constructe inilization level set function for image processing
function [level_set] = initialization_level_set(img)
% coder: Jie An
% version: 20170401
% bug_submission: pkuanjie@gmail.com

%% set up
BORDER_OFFSET = 10;
[row, col] = size(img);

%% construction
level_set = zeros(row, col);
level_set(BORDER_OFFSET: row - BORDER_OFFSET, BORDER_OFFSET) = 1;
level_set(BORDER_OFFSET: row - BORDER_OFFSET, col - BORDER_OFFSET) = 1;
level_set(BORDER_OFFSET, BORDER_OFFSET: col - BORDER_OFFSET) = 1;
level_set(row - BORDER_OFFSET, BORDER_OFFSET: col - BORDER_OFFSET) = 1;
level_set = bwdist(level_set);
level_set(BORDER_OFFSET: row - BORDER_OFFSET, BORDER_OFFSET: col - BORDER_OFFSET) = -1 * level_set(BORDER_OFFSET: row - BORDER_OFFSET, BORDER_OFFSET: col - BORDER_OFFSET);
level_set = 10 * level_set;

% figure,
% mesh(level_set), axis image;
