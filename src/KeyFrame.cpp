#include "KeyFrame.h"

namespace myslam{
    unsigned long KeyFrame::nLastId = 0;

    KeyFrame::KeyFrame(const Frame::Ptr &frame){
        mFrameId = frame->mFrameId;
        mImgLeft = frame->mImgLeft;
        mImgRight = frame->mImgRight;
        mTimeStamp = frame->mTimeStamp;
        mvpFeatureLeft = frame->mvpFeatureLeft;
        mvpFeatureRight = frame->mvpFeatureRight;
        mPose = frame->mPose;
        mK = frame->mK;

        mKeyFrameId = nLastId++;

    }

}