// WARNNING!!! this code have been give up!
// due to BORDER_WRAP can't access in my opencv setup, but in this case I have
// to use it. To avoiding making wheels by hand, I just use Matlab to complete
// my hw1, code can be see in MATLAB dir;
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudev.hpp>
#include <stdio.h>
#include "iostream"
#include "cuda_runtime.h"
#include "admm_tools.cuh"
using namespace std;
using namespace cv;

int main(){
    const Size KERNEL_SIZE(15, 15);
    const float SIGMA = 1.5;
    const int SCALE = 100;
    const float LAMBDA = sqrt(2) - 1;

    // image resize, Gaussian blur and noise adding of original image

    cout << "pre processing of image" << endl;
    Mat img = imread("lena.bmp");
    Mat img_gray_tmp;
    Mat_<float> img_gray;
    cvtColor(img, img_gray_tmp, CV_RGB2GRAY);
    img_gray_tmp.convertTo(img_gray, CV_32FC1, 1.0/255.0);
    img.release();
    img_gray_tmp.release();
    resize(img_gray, img_gray, Size(), 1.0 / 16, 1.0 / 16, INTER_LINEAR);
    Mat_<float> img_ori;
    img_gray.copyTo(img_ori);
    //namedWindow("img1", WINDOW_NORMAL);
    //imshow("img1", img_gray);
    //waitKey(0);
    GaussianBlur(img_gray, img_gray, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    //namedWindow("img2", WINDOW_NORMAL);
    //imshow("img2", img_gray);
    //waitKey(3000);
    double* value_max_img = new double[1];
    cuda::minMax(img_gray, NULL, value_max_img);
    Mat_<float> img_noise(img_gray.rows, img_gray.cols, CV_32FC1);
    randn(img_noise, 0, value_max_img[0]);
    cuda::multiply(img_noise, value_max_img[0] / SCALE, img_noise);
    cuda::add(img_gray, img_noise, img_gray);
    //namedWindow("img3", WINDOW_NORMAL);
    //imshow("img3", img_gray);
    //waitKey(3000);
    img_noise.release();
    free(value_max_img);

    // generate matrix A and W

    cout << "start compute A and W" << endl;
    Mat_<float> A = Mat::zeros(img_gray.rows * img_gray.cols, img_gray.rows * img_gray.cols, CV_32FC1);
    Mat_<float> W = Mat::zeros(img_gray.rows * img_gray.cols, img_gray.rows * img_gray.cols, CV_32FC1);
    cudev::GpuMat_<float> A_device(A);
    cudev::GpuMat_<float> W_device(W);
    if (A_device.isContinuous() == 0){
        throw "memory of gray image is not continuous, cannot use cuda!";
    }
    if (W_device.isContinuous() == 0){
        throw "memory of gray image is not continuous, cannot use cuda!";
    }
    Mat_<float> Gauss = Mat::zeros(15, 15, CV_32FC1);
    Gauss.at<float>(7, 7) = 1;
    GaussianBlur(Gauss, Gauss, KERNEL_SIZE, SIGMA, BORDER_WRAP);
    cudev::GpuMat_<float> Gauss_device(Gauss);
    dim3 thread_perblock(Gauss.rows, Gauss.cols);
    compute_A<<<A.rows, thread_perblock>>>(A_device.ptr<float>(0), Gauss_device.ptr<float>(0), A.rows, A.cols, img_gray.rows, img_gray.cols, Gauss.rows, Gauss.cols);
    compute_W<<<W.rows, 1>>>(W_device.ptr<float>(0), W.rows, W.cols, img_gray.rows, img_gray.cols, LAMBDA);
    Gauss.release();
    Gauss_device.release();

    // Split Bregman Algorithm iteration

    cout << "prepare iteration" << endl;
    const float mu = 0.5;
    const float tol = 1e-6;
    const float delta = 0.5;
    const int MAX_ITERATION = 100;
    const float L = 1;

    Mat_<float> img_gray_t;
    transpose(img_gray, img_gray_t);
    Mat_<float> img_gray_t2 = img_gray_t.reshape(0, img_gray.rows * img_gray.cols);
    cudev::GpuMat_<float> f(img_gray_t2);
    cudev::GpuMat_<float> f_modified(img_gray_t2);
    img_gray_t.release();
    img_gray_t2.release();
    Mat_<float> source = Mat::zeros(f.rows, f.cols, CV_32FC1);
    Mat_<float> source1 = Mat::ones(f.rows, f.cols, CV_32FC1);
    Mat_<float> source2 = Mat::zeros(A.rows, A.cols, CV_32FC1);
    cudev::GpuMat_<float> d(source1);
    cudev::GpuMat_<float> b(source1);
    cudev::GpuMat_<float> u(source1);
    cudev::GpuMat_<float> tmp1(source);
    cudev::GpuMat_<float> tmp2(source);
    cudev::GpuMat_<float> tmp3(source1);
    cudev::GpuMat_<float> tmp4(source);
    cudev::GpuMat_<float> nomeaning(source);
    cudev::GpuMat_<float> nomeaning2(source2);

    source.release();
    source2.release();
    cuda::gemm(A_device, f, 1.0, nomeaning, 0, f, GEMM_1_T);
    cuda::gemm(A_device, A_device, 1.0, nomeaning2, 0, A_device, GEMM_1_T);
    cuda::gemm(W_device, W_device, mu, A_device, 1.0, A_device, GEMM_1_T);
    A_device.download(A);
    invert(A, A, DECOMP_LU);
    A_device.upload(A);
    cuda::gemm(A_device, f, 1.0, nomeaning, 0, f, 0);
    cuda::gemm(A_device, W_device, mu, nomeaning2, 0, A_device, GEMM_2_T);
    nomeaning.release();
    nomeaning2.release();
    cout << "start iteration" << endl;

    for (int i = 0; i < MAX_ITERATION; i++)
    {
        cuda::addWeighted(d, 1.0, b, -1.0, 1.0, tmp1);
        cuda::gemm(A_device, tmp1, 1.0, f, 1.0, u);
        cuda::gemm(W_device, u, 1.0, b, 1.0, tmp2);
        float tmp_norm = cuda::norm(tmp2, NORM_L2);
        if (tmp_norm - L / mu > 0)
        {
            cuda::addWeighted(tmp2, (tmp_norm - L / mu) / tmp_norm, tmp4, 1.0, 1.0, d);
        }
        else
        {
            tmp4.copyTo(d);
        }
        cuda::gemm(W_device, u, 1.0, d, -1.0, tmp3);
        cuda::addWeighted(tmp3, delta, b, 1.0, 1.0, b);
        float r1 = cuda::norm(tmp3, NORM_L2);
        float r2 = cuda::norm(f_modified, NORM_L2);
        cout << r2 << " " << r1 << endl;
        if (r1 / r2 < tol)
        {
            break;
        }
        cout << "iteration num = " << i << endl;
    }

    // result show

    Mat_<float> result;
    Mat_<float> result_tmp;
    u.download(result_tmp);
    result = result_tmp.reshape(0, img_gray.rows);
    result_tmp.release();
    transpose(result, result);
    namedWindow("result show", WINDOW_NORMAL);
    imshow("result show", result);
    waitKey(0);
    float error = norm(result, img_ori, NORM_L2);
    cout << "error of processing: " << error << endl;

    // release memory

    A_device.release();
    W_device.release();
    d.release();
    b.release();
    u.release();
    tmp1.release();
    tmp2.release();
    tmp3.release();
    tmp4.release();
    img_gray.release();
    img_ori.release();
    result.release();
}
