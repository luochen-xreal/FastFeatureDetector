#ifndef FFD_H
#define FFD_H

#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class FastFeatureDetector{
public:
    FastFeatureDetector(size_t scale_depth = 3,
    float extrema_threshold = 0.1f,
    float cm_lower_threshold = 0.7f,
    float cm_upper_threshold = 1.5f);


    int detect(cv::Mat image,
                std::vector<cv::KeyPoint>& keypoints);
                
    inline float rescale_sigma(float linear_sigma) {
        return 0.82357472f * std::exp(0.68797398f * linear_sigma);
    }
private:
    size_t scale_depth_;
    
    std::vector<cv::Mat> kernels_;
    std::vector<cv::Mat> coarse_imgs_;
    cv::Mat dx_kernel = (cv::Mat_<float>(1, 3) << -0.5f, 0, 0.5f);
    cv::Mat dy_kernel = (cv::Mat_<float>(3, 1) << -0.5f, 0, 0.5f);
    cv::Mat hxx_kernel = (cv::Mat_<float>(1, 3) << 1.f, -2.f, 1.f);
    cv::Mat hyy_kernel = (cv::Mat_<float>(3, 1) << 1.f, -2.f, 1.f);

    
    float extrema_threshold_ = 0.1f;
    float cm_lower_threshold_ = 0.7f;
    float cm_upper_threshold_ = 1.5f;
};



#endif