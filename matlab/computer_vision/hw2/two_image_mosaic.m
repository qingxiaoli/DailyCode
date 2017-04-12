%% this function is to use SIFT and RANSAC to mosaic imagesc
function mosaic = two_image_mosaic(img1_rgb, img2_rgb)
% coder: Jie An
% version: 20170405
% bug_submission: pkuanjie@gmail.com

%% set up
N = 4;
K = 300;
T = 2;
D = 60;
PEAK_THRESH = 0.5;
EDGE_THRESH = 10;

%% using sift to detect feature points
img1 = single(rgb2gray(img1_rgb));
img2 = single(rgb2gray(img2_rgb));
[frames1, descriptors1] = vl_sift(img1, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
figure(1),
imagesc(img1), axis off,
title('img1 with sift feature');
vl_plotframe(frames1);

[frames2, descriptors2] = vl_sift(img2, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
figure(2),
imagesc(img2), axis off,
title('img2 with sift feature');
vl_plotframe(frames2);

%% match with sift descriptor
[matches, scores] = vl_ubcmatch(descriptors1, descriptors2);
if size(matches, 2) > 300
    matches = matches(:, 1: 300);
    scores = scores(:, 1: 300);
end;
[~, perm] = sort(scores, 'descend');
matches = matches(:, perm);
% figure(3) ; clf;  
% imagesc(cat(2, img1, img2));  
% title('two image match result');
% axis image off;  

xa = frames1(1, matches(1, :));
xb = frames2(1, matches(2, :)) + size(img1, 2);  
ya = frames1(2, matches(1, :));  
yb = frames2(2, matches(2, :));  
  
% hold on ;  
% h = line([xa ; xb], [ya ; yb]);  
% set(h,'linewidth', 1, 'color', 'b');  
%   
% vl_plotframe(frames1(:, matches(1, :)));  
frames2(1, :) = frames2(1, :) + size(img1, 2);  
% vl_plotframe(frames2(:, matches(2, :)));
% axis image off;  

%% compute transform matrix with RANSAC
A = [xa', ya'];
B = [xb' - size(img1, 2), yb'];
[best_model, best_consensus_set_A, best_consensus_set_B, best_error] = ransac(A, B, N, K, T, D);

% figure(4), 
% imagesc(cat(2, img1, img2_extend)), axis image off, hold on,
% title('RANSAC match points');
% plot(best_consensus_set_A(:, 1), best_consensus_set_A(:, 2), 'go', 'LineWidth',2), hold on,
% plot(best_consensus_set_B(:, 1) + size(img1, 2), best_consensus_set_B(:, 2), 'go', 'LineWidth',2),
% h_m = line([best_consensus_set_A(:, 1)'; best_consensus_set_B(:, 1)' + size(img1, 2)], [best_consensus_set_A(:, 2)'; best_consensus_set_B(:, 2)']);
% set(h_m,'linewidth', 1, 'color', 'b');
% hold off;

%% transform image and show 
T = projective2d(best_model);
[img1_t, RB] = imwarp(img1_rgb, T);
x_l = round(min(RB.XWorldLimits(1), 1));
x_r = round(max(RB.XWorldLimits(2), size(img2, 2)));
y_l = round(min(RB.YWorldLimits(1), 1));
y_r = round(max(RB.YWorldLimits(2), size(img2, 1)));
position = zeros(2, 2);
position(2, 2) = 1;
position(2, 1) = 1;
position(1, 2) = round(RB.XWorldLimits(1));
position(1, 1) = round(RB.YWorldLimits(1));
if (x_l <= 0)
    x_r = x_r - x_l + 1;
    position(1, 2) = 1;
    position(2, 2) = 1 - x_l + 1;
    x_l = 1;
end;
if (y_l <= 0)
    y_r = y_r - y_l + 1;
    position(1, 1) = 1;
    position(2, 1) = 1 - y_l + 1;
    y_l = 1;
end;
img1_m = zeros(y_r - y_l + 1, x_r - x_l + 1, 3);
img2_m = zeros(y_r - y_l + 1, x_r - x_l + 1, 3);
img1_m(position(1, 1): position(1, 1) + RB.ImageSize(1) - 1, position(1, 2): position(1, 2) + RB.ImageSize(2) - 1, :) = img1_t;
img2_m(position(2, 1): position(2, 1) + size(img2, 1) - 1, position(2, 2): position(2, 2) + size(img2, 2) - 1, :) = img2_rgb;
mosaic = imsubtract(uint8(img1_m), uint8(img2_m));
mosaic = imadd(mosaic, uint8(img2_m));
% mosaic = medfilt2(mosaic);
figure,
imshow(mosaic), axis image off;
title('mosaic image');



