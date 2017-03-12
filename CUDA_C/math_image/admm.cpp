#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "iostream"
using namespace std;
using namespace cv;
using namespace cv::gpu;

int main(){
    const Size KERNEL_SIZE(15, 15);
    const double SIGMA = 1.5;



    Mat img = imread("lena.bmp");
    Mat img_gray;
    cvtColor(img, img_gray, CV_RGB2GRAY);
    img_gray.convertTo(img, CV_32FC1, 1.0/255.0);
    imshow("test", img);
    waitKey(0);
    img_gray.release();
    GaussianBlur(img, img, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    double value_max_img;
    minMax(img, img_noise, img)
    Mat img_noise;
    randn(img_noise, 0, value_max_img);
    gpu::add(img, img_noise, img);
}
