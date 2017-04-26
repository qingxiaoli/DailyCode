%% this script is to use admm algorithm and discrete wavelet transform to implement image deblur
% coder: Jie An
% version: 20170426
% bug_submission: pkuanjie@gmail.com

clear
clc
addpath(genpath('./2DTWFT'));

%% set up
KERNEL_SIZE = 15;
SHIFT_LEN = (KERNEL_SIZE - 1) / 2;
SIGMA = 2.0;
SCALE = 200;
MU = 0.3;
LAMBDA = 0.06;
TAU = LAMBDA / MU;
TOL = 1e-3;
DELTA = 0.12;
MAX_ITERATION = 500;
PATH_OF_IMAGE = 'aircraft.jpg';
FRAME = 1;
LEVEL = 2;
[D,R]=GenerateFrameletFilter(FRAME);
W  = @(x) FraDecMultiLevel2D(x,D,LEVEL); % Frame decomposition
WT = @(x) FraRecMultiLevel2D(x,R,LEVEL); % Frame reconstruction

%% image pre processing
disp('start image preprocessing');
img = imread(PATH_OF_IMAGE);
if size(size(img), 2) == 3
    img = im2double(rgb2gray(img));
else
    img = im2double(img);
end;
img = imresize(img, [512, 512]);
% figure,
% imshow(img);
kernel = fspecial('gaussian', [15, 15], SIGMA);
img_blur = imfilter(img, kernel, 'circular');
% figure,
% imshow(img_blur);
img_pro = img_blur + max(max(img)) / SCALE * randn(size(img, 1));
% figure,
% imshow(img_pro);

%% admm
A = kernel;
f = img_pro;
d = W(img);
b = d;
Af = imfilter(f, A, 'circular');
tempA = zeros(size(img));
tempI = zeros(size(img));
tempA(1: KERNEL_SIZE, 1: KERNEL_SIZE) = A;
tempA = circshift(tempA, [-SHIFT_LEN, -SHIFT_LEN]);
tempI(1: 3, 1: 3) = [0, 0, 0; 0, 1, 0; 0, 0, 0];
tempI = MU * circshift(tempI, [-1, -1]);
ftA = fft2(tempA);
ftI = fft2(tempI);
tempFT = ftA .* ftA + ftI;% this mat is kernel to solve equation

error = zeros(MAX_ITERATION, 1);
figure,
for i = 1 : MAX_ITERATION

    % step 1
    [dx, dy] = gradient(d - b);
    dx = dx(1: size(img, 1), :);
    dy = dy(size(img, 1) + 1: end, :);
    u = real(ifft2(fft2(Af + MU * WT(d - b)) ./ (tempFT)));
    imshow(u);

    % step 2
    tmp = W(u) + b;
    tmp3 = sqrt([tmp(1: size(img, 1), :) .^ 2 + tmp(size(img, 1) + 1: end, :) .^ 2; tmp(1: size(img, 1), :) .^ 2 + tmp(size(img, 1) + 1: end, :) .^ 2]);
    tmp(find(tmp3 > TAU)) = tmp(find(tmp3 > TAU)) ./ tmp3(find(tmp3 > TAU)) .* (tmp3(find(tmp3 > TAU)) - TAU);
    tmp(find(tmp3 <= TAU)) = 0 * tmp(find(tmp3 <= TAU));
    d = tmp;

    % step 3
    b = b + DELTA * ([dxu; dyu] - d);

    % error saving and break checking
    check2 = norm([dxu; dyu] - d, 'fro') / norm(f, 'fro');
    error(i) = norm(u - img, 'fro') ^ 2;
    if check2 < TOL
        break;
    end;
    disp(['iteration num = ', num2str(i), ' finished. ', 'check error = ', num2str(check2)]);
end;

%% result show
figure,
subplot(221), imshow(img), title('original image');
subplot(222), imshow(f), title('filtered image');
subplot(223), imshow(u), title('reconstructed image');
subplot(224), plot(error), title('error');

figure,
subplot(131), imagesc(img), axis image, title('original image'), colorbar;
subplot(132), imagesc(f), title('filtered image'), axis image, colorbar;
subplot(133), imagesc(u), title('reconstructed image'), axis image, colorbar;
