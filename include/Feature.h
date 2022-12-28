#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include "Common.h"
#include "MapPoints.h"

namespace myslam
{
    class MapPoints;
class Feature{
public:
    Feature()= default;
    Feature(const cv::KeyPoint &keypoint){
        mKeyPoint = keypoint;
    }
public:
    typedef std::shared_ptr<Feature> Ptr;
    cv::KeyPoint mKeyPoint;
    std::weak_ptr<MapPoints> mpMapPoint;
    bool IsOutLier = false; //用来标记其对应的地图点为外点，位姿估计时使用

private:


};



} // namespace myslam


#endif