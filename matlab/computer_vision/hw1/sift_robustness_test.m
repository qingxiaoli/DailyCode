%% this script is write to test the robustness of SIFT detector and descriptor
% coder: Jie An
% version: 20170319
% bug_submission: pkuanjie@gmail.com

%% set up
IMG_PATH = 'cover_1.jpg';
PEAK_THRESH = 1;
EDGE_THRESH = 5;
ROTATE_ANGLE = 15;
SCALE_FACTOR = 1.2;

%% img import
img = single(rgb2gray(imread(IMG_PATH)));
figure,
imagesc(img), axis image;

%% using sift to detect feature points
[frames, descriptors] = vl_sift(img, 'PeakThresh', PEAK_THRESH, 'EdgeThresh', EDGE_THRESH);
vl_plotframe(frames);
feature_ori = frames;


%% rotate image and detect same corner point
feature_matches_rotation = zeros(size(ROTATE_ANGLE : ROTATE_ANGLE : 359, 2), 1);
for i = ROTATE_ANGLE : ROTATE_ANGLE : 359
    img_rotation = imrotate(img, i);
%     figure,
%     imagesc(img_rotation);
    rot_mat = [cos(i / 180 * pi), -sin(i / 180 * pi); sin(i / 180 * pi), cos(i / 180 * pi)];
    [frames_rot, descriptors_rot] = vl_sift(img_rotation, 'PeakThresh', 1, 'EdgeThresh', 5);
    feature_rot = ((feature_ori(1: 2, :)' -  size(img) / 2) * rot_mat)';
    for j = 1 : size(feature_ori, 2)
        if size(find(abs(frames_rot(1, :) - size(img_rotation, 1) / 2 - feature_rot(1, j)) < 2), 1) > 0 && size(find(abs(frames_rot(2, :) - size(img_rotation, 2) / 2 - feature_rot(2, j)) < 2), 1)
            feature_matches_rotation(i / ROTATE_ANGLE) = feature_matches_rotation(i / ROTATE_ANGLE) + 1;
        end;
    end;
end;
figure,
plot(ROTATE_ANGLE : ROTATE_ANGLE : 359, log(feature_matches_rotation), 'ro'), title('repeatability of rotation');
xlabel('rotation angle'),
ylabel('sift matches rate'),
set(gca, 'XTick', ROTATE_ANGLE : ROTATE_ANGLE : 359);


%% scale image and detect same corner point
feature_matches_scale = zeros(9, 1);
for i = 0 : 8
    img_scale = imresize(img, SCALE_FACTOR ^ i);
%     figure,
%     imagesc(img_rotation);
    scale_mat = [SCALE_FACTOR ^ i, 0; 0, SCALE_FACTOR ^ i];
   [frames_scale, descriptors_scale] = vl_sift(img_scale, 'PeakThresh', 1, 'EdgeThresh', 5);
    feature_scale = ((feature_ori(1: 2, :)' -  size(img) / 2) * scale_mat)';
    for j = 1 : size(feature_ori, 2)
        if size(find(abs(frames_scale(1, :) - size(img_scale, 1) / 2 - feature_scale(1, j)) < 2), 1) > 0 && size(find(abs(frames_scale(2, :) - size(img_scale, 2) / 2 - feature_scale(2, j)) < 2), 1)
            feature_matches_scale(i +1) = feature_matches_scale(i +1) + 1;
        end;
    end;
end;
figure,
plot(0 : 8, log(feature_matches_scale), 'ro'), title('repeatability of scaling');
xlabel('scale'),
ylabel('sift matches rate'),
set(gca, 'XTick', 0 : 8);