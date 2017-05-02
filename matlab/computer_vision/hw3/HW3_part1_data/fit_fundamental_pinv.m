%% this function is to compute fundamental matrix with 8 points algorithm
function F = fit_fundamental_pinv(matches)
% matches: matches points pairs
% F: fundamental matirx
% coder: Jie An
% version: 20170502
% bug_submission: pkuanjie@gmail.com

% while 1
%     choose = unique(randperm(size(matches, 1), 8));
%     if (size(choose, 2) == 8)
%         break;
%     end;
% end;
% A_piece = matches(choose, 1: 2);
% B_piece = matches(choose, 3: 4);
A_piece = matches(:, 1: 2);
B_piece = matches(:, 3: 4);

mat_data = zeros(size(matches, 1), 8);
mat_right = -1 * ones(size(matches, 1), 1);
% construct mat_data
mat_data(:, 1) = A_piece(:, 1) .* B_piece(:, 1);
mat_data(:, 2) = A_piece(:, 1) .* B_piece(:, 2);
mat_data(:, 3) = A_piece(:, 1);
mat_data(:, 4) = A_piece(:, 2) .* B_piece(:, 1);
mat_data(:, 5) = A_piece(:, 2) .* B_piece(:, 2);
mat_data(:, 6) = A_piece(:, 2);
mat_data(:, 7) = B_piece(:, 1);
mat_data(:, 8) = B_piece(:, 2);
% cunstruct end
maybe_model = [mat_data \ mat_right; 1];
F = reshape(maybe_model, [3, 3]);
[U, S, V] = svd(F);
S(3, 3) = 0;
F = U * S * V';
