%% this script is to perform image denoising with shock filter method
% coder: Jie An
% version: 20170321
% bug_submission: pkuanjie@gmail.com


%% set up
IMG_PATH = 'lena.bmp';
MAX_ITERATION = 200;
NOISE_SCALE = 100;
DISCRETE_TIME = 0.01;
SIGMA = 2;



%% image preprocessing
img = im2double(rgb2gray(imread(IMG_PATH)));
% figure,
% imshow(img), axis image;
kernel = fspecial('gaussian', [15, 15], SIGMA);
img_blur = imfilter(img, kernel, 'circular');
img_noise = img_blur + max(max(img_blur)) / NOISE_SCALE * randn(size(img));
% figure,
% imshow(img_noise), axis image;


%% perona malik equation
img_pro = img_noise;
Lap = fspecial('laplacian', 1);
result = cell(1, 2);
% figure,
for i = 1 : MAX_ITERATION
    [dx, dy] = gradient(img_pro);
    L = imfilter(img_pro, Lap, 'circular');
    tmp = -sqrt(dx .^ 2 + dy .^ 2) .* F(L);
    img_pro = img_pro + DISCRETE_TIME * tmp;
%     imshow(img_pro);
    disp(['num = ', num2str(i), ' image processing finished']);
end;
result{1} = img_pro;

img_pro = img_noise;
% figure,
for i = 1 : MAX_ITERATION
    [dx, dy] = gradient(img_pro);
    [dxx, dxy] = gradient(dx);
    [~, dyy] = gradient(dy);
    L = dx .^ 2 .* dxx + 2 * dx .* dy .* dxy + dy .^ 2 .* dyy;
    L = L ./ (dx .^ 2 + dy .^ 2);
    tmp = -sqrt(dx .^ 2 + dy .^ 2) .* F(L);
    img_pro = img_pro + DISCRETE_TIME * tmp;
%     imshow(img_pro);
    disp(['num = ', num2str(i), ' image processing finished']);
end;
result{2} = img_pro;

%% result show
figure,
subplot(131), imshow(img), axis image, title('original image');
subplot(132), imshow(img_noise), axis image, title('blur image with noise');
subplot(133), imshow(result{1}), axis image, title('denoised image');

figure,
subplot(121), imshow(result{1}), axis image, title('first type of L');
subplot(122), imshow(result{2}), axis image, title('second type of L');