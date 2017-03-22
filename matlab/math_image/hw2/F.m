%% this function is F function in shock filter method
function fl = F(L)
% input:
%     L: a matrix of image size;
% output:
%     fl: a matrix of image size which have been processed;
% coder: Jie An
% version: 20170321
% bug_submission: pkuanjie@gmail.com

fl = L;
fl(find(fl > 0)) = 1;
fl(find(fl < 0)) = -1;
