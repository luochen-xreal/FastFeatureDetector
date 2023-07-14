#include "ffd.h"


FastFeatureDetector::FastFeatureDetector(size_t scale_depth,
    float extrema_threshold,
    float cm_lower_threshold,
    float cm_upper_threshold ) : scale_depth_(scale_depth), extrema_threshold_(extrema_threshold), cm_lower_threshold_(cm_lower_threshold), cm_upper_threshold_(cm_upper_threshold){
    kernels_.resize(scale_depth_ + 3);
    coarse_imgs_.resize(scale_depth_ + 3);
    kernels_[0] = cv::Mat::zeros(1, 5, CV_32F);
    const cv::Mat gauss_kernel = cv::getGaussianKernel(5, 0.6, CV_32F);

    for (int y = -2; y <= 2; y++) {
        // kernels_[0].at<float>(2 + y) =  gauss_kernel.at<float>(2 + y, 0) / 16.0;
        kernels_[0].at<float>(2 + y) =  gauss_kernel.at<float>(2 + y, 0);
    }
    
    const float base_spline[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};

    for (size_t i = 1; i <= scale_depth_ + 2; i++) {
        const int pitch = (int)std::pow(2, i - 1) - 1;
        const int half_size = (5 + 4 * pitch) / 2;
        kernels_[i] = cv::Mat::zeros(1, (pitch * 4) + 5, CV_32F);
        for (int yi = -2; yi <= 2; yi++) {
            kernels_[i].at<float>((pitch + 1) * (yi + 2)) =  base_spline[2 + yi];
        }
    }


    
}

int FastFeatureDetector::detect(cv::Mat input_img,std::vector<cv::KeyPoint>& keypoints){

    cv::Mat image;
    input_img.convertTo(image, CV_32F);

    image = image / 255.f;
    cv::Mat prev_img = image;

    const int rows = image.rows;
    const int cols = image.cols;

    for (size_t d = 0; d < scale_depth_ + 3; d++) {
        coarse_imgs_[d] = cv::Mat::zeros(rows, cols, CV_32F);
        cv::sepFilter2D(prev_img, coarse_imgs_[d], -1, kernels_[d], kernels_[d].t());
        cv::Mat img;
        prev_img = coarse_imgs_[d];

    }
    std::vector<cv::Mat> fine_imgs(scale_depth_ + 2);
    {
        
        for (size_t i = 0; i < scale_depth_ + 2; i++) {
            fine_imgs[i] = coarse_imgs_[i] - coarse_imgs_[i + 1];
        }
    }
    keypoints.clear();
    for (size_t d = 0; d < scale_depth_; d++) {
        cv::Mat &pre = fine_imgs[d];
        cv::Mat &cur = fine_imgs[d + 1];
        cv::Mat &nxt = fine_imgs[d + 2];

        for (size_t y = 1; y < rows - 1; y++) {
            for (size_t x = 1; x < cols - 1; x++) {
                if (cur.at<float>(y, x) < extrema_threshold_ - 0.01) continue;

                cv::Mat dD(3, 1, CV_32F);
                dD.at<float>(0, 0) = (cur.at<float>(y, x + 1) - cur.at<float>(y, x - 1)) / 2.f;
                dD.at<float>(1, 0) = (cur.at<float>(y + 1, x) - cur.at<float>(y - 1, x)) / 2.f;
                dD.at<float>(2, 0) = (nxt.at<float>(y, x) - pre.at<float>(y, x)) / 2.f;

                cv::Mat H(3, 3, CV_32F);
                H.at<float>(0, 0) =
                    cur.at<float>(y, x + 1) + cur.at<float>(y, x - 1) - 2.f * cur.at<float>(y, x);
                H.at<float>(1, 1) =
                    cur.at<float>(y + 1, x) + cur.at<float>(y - 1, x) - 2.f * cur.at<float>(y, x);
                H.at<float>(2, 2) = nxt.at<float>(y, x) + pre.at<float>(y, x) - 2.f * cur.at<float>(y, x);
                H.at<float>(0, 1) = H.at<float>(1, 0) =
                    (cur.at<float>(y + 1, x + 1) - cur.at<float>(y + 1, x - 1) -
                    cur.at<float>(y - 1, x + 1) + cur.at<float>(y - 1, x - 1)) /
                    4.f;
                H.at<float>(0, 2) = H.at<float>(2, 0) =
                    (nxt.at<float>(y, x + 1) - pre.at<float>(y, x + 1) - nxt.at<float>(y, x - 1) +
                    pre.at<float>(y, x - 1)) /
                    4.f;
                H.at<float>(1, 2) = H.at<float>(2, 1) =
                    (nxt.at<float>(y + 1, x) - pre.at<float>(y + 1, x) - nxt.at<float>(y - 1, x) +
                    pre.at<float>(y - 1, x)) /
                    4.f;

                cv::Mat dpos;
                cv::solve(H, -dD, dpos);
                if (std::abs(dpos.at<float>(0, 0)) < 0.5f && std::abs(dpos.at<float>(1, 0)) < 0.5f &&
                    std::abs(dpos.at<float>(2, 0)) < 0.5f) {
                    double cm = 1 - 4*(H.at<float>(0, 0) * H.at<float>(1, 1) - H.at<float>(1, 0)*H.at<float>(1, 0)) / 
                    ((H.at<float>(0, 0) + H.at<float>(1, 1))* (H.at<float>(0, 0) + H.at<float>(1, 1)));
                    float response = cur.at<float>(y, x) + dD.dot(dpos) / 2.f;
                    if(response > extrema_threshold_ && 
                        (cm <= cm_lower_threshold_ || cm >= cm_upper_threshold_)){
                        cv::KeyPoint kpt(cv::Point2f(x + dpos.at<float>(0, 0), y + dpos.at<float>(1, 0)),
                                    rescale_sigma(d + 1 + dpos.at<float>(2, 0)),  response);
                        keypoints.push_back(kpt);
                    }
                }
            }
        }
    }
    return keypoints.size();
}
