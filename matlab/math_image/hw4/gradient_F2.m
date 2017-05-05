%% this function is to compute gradient of F_2
function result = gradient_F2(alpha, f, W, WT, A)
% coder: Jie An
% version: 20170504
% bug_submission: pkuanjie@gmail.com

tmp1 = WT(alpha);
tmp2 = imfilter(tmp1, A, 'circular');
tmp3 = tmp2 - f;
tmp4 = imfilter(tmp3, A, 'circular');
result = W(tmp4);