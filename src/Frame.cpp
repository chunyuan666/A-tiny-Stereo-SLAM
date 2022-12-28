#include "Frame.h"
#include "Common.h"

namespace myslam{
    unsigned long Frame::nLastId = 0;

    Frame::Frame(const cv::Mat &imgleft, const cv::Mat &imgright, const double &timestamp,
          const Mat33d &k){

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        cv::Mat leftimg_clahe, rightimg_clahe;
        clahe->apply(imgleft, leftimg_clahe);
        clahe->apply(imgright, rightimg_clahe);
        mImgLeft = imgleft.clone();
        mImgRight = imgright.clone();
        mImgOrigin = imgleft.clone();
        mTimeStamp = timestamp;
        mPose = SE3_Identity;
        mK = k;
        mFrameId = nLastId++;
    }


}