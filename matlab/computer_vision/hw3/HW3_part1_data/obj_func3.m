%% this function is object function for optimization of fundamental matrix computing
function result = obj_func3(F)
% coder: Jie An
% version: 20170502
% bug_submission: pkuanjie@gmail.com

matches = load('house_matches.txt'); 
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
result = sum((mat_data * F - mat_right) .^ 2);