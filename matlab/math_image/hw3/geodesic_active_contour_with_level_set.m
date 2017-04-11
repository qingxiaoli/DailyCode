%% this script is to implement geodesic active contour model with level set formulation
% coder: Jie An
% version: 20170403
% bug_submission: pkuanjie@gmail.com

%% set up
IMG_PATH = 'test_img.tif';
SIGMA = 0.000001;
NOISE_SCALE = 10000000;
STEP_SIZE = 0.01;
ALPHA = 50;
MAX_ITERATION = 1000;

%% image pre pocessing
img = double(imread(IMG_PATH));
% figure,
% imshow(img), axis image;
kernel = fspecial('gaussian', [15, 15], SIGMA);
img_blur = imfilter(img, kernel, 'circular');
img_noise = img_blur + max(max(img_blur)) / NOISE_SCALE * randn(size(img));
img_noise = im2double(img_noise);
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
%     imagesc(v), colormap gray, axis image; drawnow
    if mod(i, 2) == 1
        imagesc(v), axis image, hold on,
    contour(v, [0, 0], 'k'), drawnow, hold off;
    end;
    [dx_I, dy_I] = gradient(img_noise);
%     tmp1 = g(sqrt(dx_I .^ 2 + dy_I .^ 2));
    [dx_v, dy_v] = gradient(v);
    dv_norm = sqrt(dx_v .^ 2 + dy_v .^ 2 + eps);
    dI_norm = sqrt(dx_I .^ 2 + dy_I .^ 2 + eps);
    tmp1 = dv_norm;
    dx_v_norm = dx_v ./ dv_norm;
    dy_v_norm = dy_v ./ dv_norm;
    tmp1 = tmp1 .* divergence(dx_v_norm, dy_v_norm) .* g(dI_norm);
    v_x_minus = circshift(v, [0, 1]);
    v_x_add = circshift(v, [0, -1]);
    v_y_minus = circshift(v, [0, 1]);
    v_y_add = circshift(v, [0, -1]);
    delta_x_add = v_x_add - v;
    delta_x_minus = v - v_x_minus;
    delta_y_add = v_y_add - v;
    delta_y_minus = v - v_y_minus;
    delta_x_minus(delta_x_minus < 0) = 0;
    delta_y_minus(delta_y_minus < 0) = 0;
    delta_x_add(delta_x_add > 0) = 0;
    delta_y_add(delta_y_add > 0) = 0;
    tmp2 = ALPHA * g(dI_norm) .* sqrt(delta_x_minus .^ 2 + delta_x_add .^ 2 + delta_y_minus .^ 2 + delta_y_add .^ 2);
    [dx_g, dy_g] = gradient(g(dI_norm));
    a1_max = dx_g;
    a1_min = dx_g;
    a2_max = dy_g;
    a2_min = dy_g;
    a1_max(a1_max < 0) = 0;
    a1_min(a1_min > 0) = 0;
    a2_max(a2_max < 0) = 0;
    a2_min(a2_min > 0) = 0;
    delta_x_add = v_x_add - v;
    delta_x_minus = v - v_x_minus;
    delta_y_add = v_y_add - v;
    delta_y_minus = v - v_y_minus;
%     tmp3 = a1_max .* delta_x_minus + a1_min .* delta_x_add + a2_max .* delta_y_minus + a2_min .* delta_y_add;
    tmp3 = -img_noise;
    v = v + STEP_SIZE * (tmp1 + tmp2 + tmp3);
%     imagesc(tmp1), axis image; drawnow
    if mod(i, 10) == 9
        v = Reinitial2D(v, 10);
    end;
    disp(['itetarion number ', num2str(i), ' finished!']);
end;


