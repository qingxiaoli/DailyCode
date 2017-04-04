%% this script is to create naive test image to test correctness of geodesic active contour model
% coder: Jie An
% version: 20170403
% bug_submission: pkuanjie@gmail.com

%% set up
IMG_SIZE = 512;
CIRCLE_POSITION = [150, 255];
GRID_POSITION = [350, 255];
CIRCLE_RADIUS = 40;
GRID_LENGTH = 40;
x = 1 : IMG_SIZE;
y = 1 : IMG_SIZE;
[X, Y] = meshgrid(x, y);

%% construction
img = zeros(IMG_SIZE, IMG_SIZE);
img(find(((X - GRID_POSITION(1)) .^ 2 + (Y - GRID_POSITION(2)) .^ 2) < 1.5 * GRID_LENGTH .^ 2)) = 1;
img(find(X - GRID_POSITION(1) > GRID_LENGTH)) = 0;
img(find(X - GRID_POSITION(1) < -GRID_LENGTH)) = 0;
img(find(Y - GRID_POSITION(2) > GRID_LENGTH)) = 0;
img(find(Y - GRID_POSITION(2) < -GRID_LENGTH)) = 0;
img(find(((X - CIRCLE_POSITION(1)) .^ 2 + (Y - CIRCLE_POSITION(2)) .^ 2) < CIRCLE_RADIUS ^ 2)) = 1;
figure,
imagesc(img), axis image;
imwrite(img, 'test_img.tif');