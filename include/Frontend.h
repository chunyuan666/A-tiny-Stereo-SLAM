#ifndef _FRONTEND_H
#define _FRONTEND_H

#include <utility>

#include "Common.h"
#include "Backend.h"
#include "Frame.h"
#include "ORBExtractor.h"
#include "Camera.h"
#include "MapPoints.h"
#include "Map.h"
#include "KeyFrame.h"
#include "Optimizer.h"
#include "Viewer.h"

namespace myslam{

class Frontend{

public:

    /**
     * @ 构造函数
    */
    Frontend()= default;

    /**
     * @brief 前端主线程
    */
   bool Running(const cv::Mat &leftImg, const cv::Mat &rightImg, const double timestamp);

   /**
    * @brief 设置跟踪参数状态
   */
    void SetTrackingPara(const int &initGood, const int &trackingGood, const int &trackingBad,
                        const float &thDepth){
        mnFeaturesTrackingGood = trackingGood;
        mnFeaturesTrackingBad=trackingBad;
        mnFeaturesInitGood=initGood;
        ThDepth = thDepth;
    }

  
    bool StereoInit();

    /**
     * @brief 提取右图特征点
     */
    unsigned long MatchFeaturesInRight();

    /**
     * @brief 提取特征点
    */
    unsigned long DetectFeature();

    /**
     * @brief 设置特征提取器
    */
    void SetORBExtractor(const ORBExtractor::Ptr& orb_extor_init,const ORBExtractor::Ptr& orb_extor){
        mORBExtractor = orb_extor;
        mORBExtractorInit = orb_extor_init;
    }

    /**
     * @brief 设置相机
     * */
    void SetCamera(const Camera::Ptr &cam_left, const Camera::Ptr &cam_right){
        mCameraLeft = cam_left;
        mCameraRight = cam_right;
    }

    /**
     * @beirf 设置地图
    */
    void SetMap(const Map::Ptr &map){
        mMap = map;
    }

    /**
     * ＠brief 初始化地图
     * */
     unsigned long  MapInit();

    /**
     * ＠brief 三角化函数
     * ＠param[in] 相机1的投影矩阵
     * ＠param[in] 相机2的投影矩阵
     * ＠param[in] 图像1的特征点
     * ＠param[in] 图像2的特征点
     * ＠param[out] 三角化后地图点的坐标
     * */
    static void Triangulation(const Mat34d &P1, const Mat34d &P2,
                       const cv::KeyPoint &Kp1, const cv::KeyPoint &Kp2, Vector3d &X3D);

    /**
     * @brief 检查是否需要插入关键帧
    */
    bool CheckIsInserKF();

    /**
     * @brief 插入关键帧
     * 
    */
    bool InsertKeyFrame();

    /**
     * @brief　追踪函数
    */
    bool Tracking();

    void Track();

    /**
     * @brief 光流匹配
     * 
    */
    bool MatchLastFrameByLKFlow();

    /**
     * @brief 估计当前帧位姿
    */
    unsigned long EstimatePose();

    /**
     * @brief 创建新的地图点
    */
    unsigned long CrateNewMapPoints();

    /**
     * @brief 设置后端
    */
    void SetBackend(const Backend::Ptr &backend){
        mBackend = backend;
    }

    bool isInBorder(const cv::Point2f &point){
        float x = point.x;
        float y = point.y;
        if((x >= 0 && x <= mCurrentFrame->mImgRight.cols)
            && (y >= 0 && y <= mCurrentFrame->mImgRight.rows))
            return true;
        else
            return false;
    }

    void SetObsForKF(const KeyFrame::Ptr &kf){
        for(auto &feat : kf->mvpFeatureLeft){
            auto mp = feat->mpMapPoint.lock();
            if(mp){
                mp->AddObservation(kf->mKeyFrameId, feat);
            }
        }
    }

    void SetViewer(const Viewer::Ptr &v){
        mViewer = v;
    }

    bool TrackWithMotionModel();

    bool TrackWithReferenceKF();

    unsigned long RefinePose(SE3 &Tcw_rough);

    unsigned long SearchByLKFlow(KeyFrame::Ptr &kf, Frame::Ptr &frame);

    template<typename T>
    void reduceVector(T &v, std::vector<uchar> status)
    {
        if(status.empty())  return;
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }

public:
    typedef std::shared_ptr<Frontend> Ptr;
    Backend::Ptr mBackend;
    Frame::Ptr mCurrentFrame;
    Frame::Ptr mLastFrame;
    ORBExtractor::Ptr mORBExtractor, mORBExtractorInit;
    Camera::Ptr mCameraLeft, mCameraRight;
    Map::Ptr mMap;
    Viewer::Ptr mViewer;
    KeyFrame::Ptr mReferenceKF;
    bool IsInsernewkf = false;

    std::list<SE3> mRelativeToRefPose;
    std::list<KeyFrame::Ptr> mlReferKFs;

    enum TrackingStatus{INIT, OK, LOST};
    

private:
    TrackingStatus mStatus = INIT; 
     // params for deciding the tracking status
    int mnFeaturesTrackingGood;
    int mnFeaturesTrackingBad;
    int mnFeaturesInitGood;

    float ThDepth;

    SE3 mVelocity = SE3_Identity; 


};

}

#endif