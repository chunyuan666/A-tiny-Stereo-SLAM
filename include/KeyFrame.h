#ifndef MYSLAM_KEYFRAME_H
#define MYSLAM_KEYFRAME_H

#include "Common.h"
#include "Frame.h"
#include "DeepLCD.h"

namespace myslam{

class KeyFrame:public Frame{
public:
    KeyFrame(const Frame::Ptr &frame);
    std::vector<cv::KeyPoint> mvPyramidKeyPoints;
    cv::Mat mDescriptors;
    DeepLCD::DescrVector mDescrVector;

public:
    typedef std::shared_ptr<KeyFrame> Ptr;
    unsigned long mKeyFrameId;
    static unsigned long nLastId;
    KeyFrame::Ptr mLastKF, mLoopKF;
    SE3 mRelativePoseToLastKF, mRelativePoseToLoopKF;
};

}

#endif