#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "tic_toc.h"
#include "ffd.h"
using namespace std;

int main(int argc, char const *argv[])
{
    /* code */
    if(argc < 2) return -1;
    string image_path = argv[1];
    auto input_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    FastFeatureDetector ffd(3, 0.05);

    std::vector<cv::KeyPoint> keypoints;
    TicToc tic;
    int size = ffd.detect(input_img, keypoints);
    cout << "cost " << tic.toc() << " ms" << endl;
    cout << "detect " << size << " keypoints" << endl;
    cv::Mat keypoint_img;
    cv::drawKeypoints(input_img, keypoints, keypoint_img);
    cv::imshow("keypoints", keypoint_img);
    cv::waitKey(0);
    return 0;
}
