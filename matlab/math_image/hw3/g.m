%% this script is to create g function of image
function [result] = g(img)
% coder: Jie An
% version: 20170403
% bug_submission: pkuanjie@gmail.com

result = 1 ./ (1 + img .^ 2); 