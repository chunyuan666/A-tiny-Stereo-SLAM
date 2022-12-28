#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "Common.h"
#include "Feature.h"

namespace myslam{

class KeyFrame;

class Frame{
public:
    Frame()=default;
    Frame(const cv::Mat &imgleft, const cv::Mat &imgright, const double &timestamp,
          const Mat33d &k);
    void SetPose(const SE3 &pose){
        mPose = pose;
    }

    SE3 GetPose(){
        return mPose;
    }

    std::vector<cv::KeyPoint> GetKeyPoints(){
        std::vector<cv::KeyPoint> vkps(mvpFeatureLeft.size());
        for(int i = 0; i < vkps.size(); i++){
            vkps[i] = mvpFeatureLeft[i]->mKeyPoint;
        }
        return vkps;
    }

public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long mFrameId;
    static unsigned long nLastId;
    cv::Mat mImgLeft, mImgRight, mImgOrigin;
    SE3 mPose;
    double mTimeStamp;
    std::vector<Feature::Ptr> mvpFeatureLeft, mvpFeatureRight;
    Mat33d mK;
    std::shared_ptr<KeyFrame> mReferenceKF;
};

}
#endif