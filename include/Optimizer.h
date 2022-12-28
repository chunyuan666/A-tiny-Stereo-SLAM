#ifndef MYSLAM_OPTIMIZER_H
#define MYSLAM_OPTIMIZER_H
#include "Common.h"
#include "Frame.h"
#include "Map.h"
#include "Camera.h"
#include "g2o_types.h"

namespace myslam{
    class Optimizer{
    public:
        static unsigned long PoseOptimization(Frame::Ptr &Frame);
        static void OptimizeActivateMap(Map::Ptr &Map, const Camera::Ptr &camera);
        static void GlobalBundleAdjustment(Map::Ptr &Map, KeyFrame::Ptr &mLoopKF, KeyFrame::Ptr &CurKF);
    };
}

#endif //MYSLAM_OPTIMIZER_H
