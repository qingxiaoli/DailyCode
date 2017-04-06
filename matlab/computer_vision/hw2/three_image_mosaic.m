%% this script is to complement 3 image mosaic with SIFT and RANSAC
% coder: Jie An
% version: 20170406
% bug_submission: pkuanjie@gmail.com

%% set up
PATH1 = '';
PATH2 = '';
PATH3 = '';
N = 4;
K = 1000;
T = 5;
D = 70;
PEAK_THRESH = 5;
EDGE_THRESH = 5;

%% image import
img1 = single(rgb2gray(imread(PATH1)));
img2 = single(rgb2gray(imread(PATH2)));
img3 = single(rgb2gray(imread(PATH3)));

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

[frames3, descriptors3] = vl_sift(img3, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
figure(3),
imagesc(img3), axis off,
title('img3 with sift feature');
vl_plotframe(frames3);
feature_ori3 = frames3;
descriptors_ori3 = descriptors3;

%% match with sift descriptor
[matches12, scores12] = vl_ubcmatch(descriptors1, descriptors2);
if size(matches12, 2) > 100
    matches12 = matches12(:, 1: 100);
    scores12 = scores12(:, 1: 100);
end;
[drop12, perm12] = sort(scores12, 'descend');
matches12 = matches12(:, perm12);
scores12  = scores12(perm12);

[matches23, scores23] = vl_ubcmatch(descriptors2, descriptors3);
if size(matches23, 2) > 100
    matches23 = matches23(:, 1: 100);
    scores23 = scores23(:, 1: 100);
end;
[drop23, perm23] = sort(scores23, 'descend');
matches23 = matches23(:, perm23);
scores23  = scores23(perm23);

figure(4) ; clf;  
imagesc(cat(3, img1, img2, img3));  
title('three image match result');
axis image off;  

xa = frames1(1, matches12(1, :));
xb = frames2(1, matches12(2, :)) + size(img1, 2);  
ya = frames1(2, matches12(1, :));  
yb = frames2(2, matches12(2, :));  

xa2 = frames1(1, matches23(1, :)) + size(img1, 2);
xb2 = frames2(1, matches23(2, :)) + size(img1, 2) + size(img2, 2);  
ya2 = frames1(2, matches23(1, :));  
yb2 = frames2(2, matches23(2, :));  
  
hold on ;  
h = line([xa ; xb], [ya ; yb]);  
h23 = line([xa2 ; xb2], [ya2 ; yb2]);  
set(h,'linewidth', 1, 'color', 'b');  
  
vl_plotframe(frames1(:, matches12(1, :)));  
frames2(1, :) = frames2(1, :) + size(img1, 2);  
vl_plotframe(frames2(:, matches12(2, :)));
frames3(1, :) = frames3(1, :) + size(img1, 2) + size(img2, 2);  
vl_plotframe(frames2(:, matches23(1, :)));
vl_plotframe(frames3(:, matches23(2, :)));
axis image off;  

%% compute transform matrix with RANSAC
A = [xa', ya'];
B = [xb' - size(img1, 2), yb'];
[best_model12, best_consensus_set_A12, best_consensus_set_B12, best_error12] = ransac(A, B, N, K, T, D);

A2 = [xa2', ya2'];
B2 = [xb2' - size(img1, 2) - size(img2, 2), yb2'];
[best_model23, best_consensus_set_A23, best_consensus_set_B23, best_error23] = ransac(A2, B2, N, K, T, D);

figure(4), 
imagesc(cat(3, img1, img2, img3)), axis image off, hold on,
title('RANSAC match points');
plot(best_consensus_set_A12(:, 1), best_consensus_set_A12(:, 2), 'go', 'LineWidth',2), hold on,
plot(best_consensus_set_B12(:, 1) + size(img1, 2), best_consensus_set_B12(:, 2), 'go', 'LineWidth',2),
plot(best_consensus_set_A23(:, 1) + size(img1, 2), best_consensus_set_A23(:, 2), 'go', 'LineWidth',2), hold on,
plot(best_consensus_set_B23(:, 1) + size(img1, 2) + size(img2, 2), best_consensus_set_B23(:, 2), 'go', 'LineWidth',2),
h_m = line([best_consensus_set_A12(:, 1)'; best_consensus_set_B12(:, 1)' + size(img1, 2)], [best_consensus_set_A12(:, 2)'; best_consensus_set_B12(:, 2)']);
h_m2 = line([best_consensus_set_A23(:, 1)' + size(img1, 2); best_consensus_set_B23(:, 1)' + size(img1, 2) + size(img2, 2)], [best_consensus_set_A23(:, 2)'; best_consensus_set_B23(:, 2)']);
set(h_m,'linewidth', 1, 'color', 'b');
hold off;