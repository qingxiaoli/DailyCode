#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "iostream"
using namespace std;
using namespace cv;

int main(){
    const Size KERNEL_SIZE(15, 15);
    const double SIGMA = 1.5;
    const int SCALE = 100;

    // image resize, Gaussian blur and noise adding of original image

    Mat img = imread("lena.bmp");
    Mat img_gray;
    cvtColor(img, img_gray, CV_RGB2GRAY);
    img.release();
    resize(img_gray, img_gray, Size(), 1.0 / 4, 1.0 / 4, INTER_LINEAR);
    img_gray.convertTo(img_gray, CV_32FC1, 1.0/255.0);
    namedWindow("img1", WINDOW_NORMAL);
    imshow("img1", img_gray);
    waitKey(3000);
    GaussianBlur(img_gray, img_gray, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    namedWindow("img2", WINDOW_NORMAL);
    imshow("img2", img_gray);
    waitKey(3000);
    double* value_max_img = new double[1];
    cuda::minMax(img_gray, NULL, value_max_img);
    Mat img_noise(img_gray.rows, img_gray.cols, CV_32FC1);
    randn(img_noise, 0, value_max_img[0]);
    cuda::multiply(img_noise, value_max_img[0] / SCALE, img_noise);
    cuda::add(img_gray, img_noise, img_gray);
    namedWindow("img3", WINDOW_NORMAL);
    imshow("img3", img_gray);
    waitKey(3000);
    img_gray.release();
    img_noise.release();
    free(value_max_img);
}
