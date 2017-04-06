%% this script is to use SIFT and RANSAC to mosaic imagesc
% coder: Jie An
% version: 20170405
% bug_submission: pkuanjie@gmail.com

%% set up
PATH1 = 'uttower_left.jpg';
PATH2 = 'uttower_right.jpg';
N = 4;
K = 1000;
T = 5;
D = 70;
PEAK_THRESH = 5;
EDGE_THRESH = 5;

%% image import
img1 = single(rgb2gray(imread(PATH1)));
img2 = single(rgb2gray(imread(PATH2)));

%% using sift to detect feature points
[frames1, descriptors1] = vl_sift(img1, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
figure(1),
imagesc(img1), axis off,
title('img1 with sift feature');
vl_plotframe(frames1);
feature_ori1 = frames1;
descriptors_ori1 = descriptors1;

[frames2, descriptors2] = vl_sift(img2, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
figure(2),
imagesc(img2), axis off,
title('img2 with sift feature');
vl_plotframe(frames2);
feature_ori2 = frames2;
descriptors_ori2 = descriptors2;

%% match with sift descriptor
[matches, scores] = vl_ubcmatch(descriptors1, descriptors2);
if size(matches, 2) > 100
    matches = matches(:, 1: 100);
    scores = scores(:, 1: 100);
end;
[drop, perm] = sort(scores, 'descend');
matches = matches(:, perm);
scores  = scores(perm);
figure(3) ; clf;  
imagesc(cat(2, img1, img2));  
title('two image match result');
axis image off;  

xa = frames1(1, matches(1, :));
xb = frames2(1, matches(2, :)) + size(img1, 2);  
ya = frames1(2, matches(1, :));  
yb = frames2(2, matches(2, :));  
  
hold on ;  
h = line([xa ; xb], [ya ; yb]);  
set(h,'linewidth', 1, 'color', 'b');  
  
vl_plotframe(frames1(:, matches(1, :)));  
frames2(1, :) = frames2(1, :) + size(img1, 2);  
vl_plotframe(frames2(:, matches(2, :)));
axis image off;  

%% compute transform matrix with RANSAC
A = [xa', ya'];
B = [xb' - size(img1, 2), yb'];
[best_model, best_consensus_set_A, best_consensus_set_B, best_error] = ransac(A, B, N, K, T, D);

figure(4), 
imagesc(cat(2, img1, img2)), axis image off, hold on,
title('RANSAC match points');
plot(best_consensus_set_A(:, 1), best_consensus_set_A(:, 2), 'go', 'LineWidth',2), hold on,
plot(best_consensus_set_B(:, 1) + size(img1, 2), best_consensus_set_B(:, 2), 'go', 'LineWidth',2),
h_m = line([best_consensus_set_A(:, 1)'; best_consensus_set_B(:, 1)' + size(img1, 2)], [best_consensus_set_A(:, 2)'; best_consensus_set_B(:, 2)']);
set(h_m,'linewidth', 1, 'color', 'b');
hold off;

%% transform image and show
figure(5), 
map = imref2d(size(img2));
T = projective2d(best_model);
[img1_t, RB] = imwarp(img1, T);
imshowpair(uint8(img1_t), RB, uint8(img2), map, 'method', 'blend'), axis image off, hold on,


