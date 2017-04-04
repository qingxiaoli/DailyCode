%% this script is to implement geodesic active contour model with level set formulation
% coder: Jie An
% version: 20170403
% bug_submission: pkuanjie@gmail.com

%% set up
IMG_PATH = 'test_img.tif';
SIGMA = 0.000001;
NOISE_SCALE = 10000000;
STEP_SIZE = 0.1;
ALPHA = 10;
MAX_ITERATION = 500;

%% image pre pocessing
img = im2double(imread(IMG_PATH));
% figure,
% imshow(img), axis image;
kernel = fspecial('gaussian', [15, 15], SIGMA);
img_blur = imfilter(img, kernel, 'circular');
img_noise = img_blur + max(max(img_blur)) / NOISE_SCALE * randn(size(img));
% figure,
% imshow(img_noise), axis image;

%% geodesic active contour model
v = initialization_level_set(img_noise);
% figure,
% imagesc(u), axis image;
figure,
for i = 1 : MAX_ITERATION
    img_show = img_noise;
    img_show(find(abs(v) < 0.1)) = 5;
%     imagesc(img_show), axis image, drawnow
    imagesc(v), axis image, drawnow
    [dx_I, dy_I] = gradient(img_noise);
%     tmp1 = g(sqrt(dx_I .^ 2 + dy_I .^ 2));
    tmp1 = ones(size(v));
    [dx_v, dy_v] = gradient(v);
    dv_norm = sqrt(dx_v .^ 2 + dy_v .^ 2 + eps);
    tmp1 = tmp1 .* dv_norm;
    dx_v_norm = dx_v ./ dv_norm;
    dy_v_norm = dy_v ./ dv_norm;
    tmp1 = tmp1 .* divergence(dx_v_norm, dy_v_norm);

    tmp2 = ALPHA * g(sqrt(dx_I .^ 2 + dy_I .^ 2)) .* dv_norm;

    [dx_g, dy_g] = gradient(g(sqrt(dx_I .^ 2 + dy_I .^ 2)));
    tmp3 = dx_g .* dx_v + dy_g .* dy_v;
    v = v + STEP_SIZE * (tmp1);
%     imagesc(tmp1), axis image, drawnow
%     if mod(i, 10) == 9
%         v = Reinitial2D(v, 10);
%     end;
    disp(['itetarion number ', num2str(i), ' finished!']);
end;


