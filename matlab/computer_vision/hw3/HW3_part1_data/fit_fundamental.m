%% this function is to compute fundamental matrix with 8 points algorithm
function F = fit_fundamental(matches)
% matches: matches points pairs
% F: fundamental matirx
% coder: Jie An
% version: 20170502
% bug_submission: pkuanjie@gmail.com

%% with linear MLS
% F = fminunc(@obj_func, ones(8, 1));
% F = [F; 1];
% F = reshape(F, [3, 3]);
% [U, S, V] = svd(F);
% S(3, 3) = 0;
% F = U * S * V';

%% with original form
% Aeq = zeros(9, 9);
% Aeq(9, 9) = 1;
% beq = zeros(9, 1);
% beq(9) = 1;
% F = fmincon(@obj_func2, ones(9, 1), [], [], Aeq, beq);
% F = reshape(F, [3, 3]);
% [U, S, V] = svd(F);
% S(3, 3) = 0;
% F = U * S * V';

%% with data nomalization
