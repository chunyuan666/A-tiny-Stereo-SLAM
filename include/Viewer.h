#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include "Common.h"
#include "Frame.h"
#include "Map.h"

namespace myslam{

class Viewer{
public:
    Viewer(float pointX, float pointY, float pointZ, float pointF,
            float GraphLineWidth, float PointSize, float CameraSize, float CameraLineWidth, 
            float KeyFrameSize, float KeyFrameLineWidth):
            mViewpointX(pointX), mViewpointY(pointY), mViewpointZ(pointZ),mViewpointF(pointF),
            mGraphLineWidth(GraphLineWidth), mPointSize(PointSize), mCameraSize(CameraSize),
            mCameraLineWidth(CameraLineWidth),
             mKeyFrameSize(KeyFrameSize),mKeyFrameLineWidth(KeyFrameLineWidth)
            {
                mViewerIsRunning.store(true);
                mthreadViewer = std::thread(std::bind(&Viewer::LoopRunning, this));
            }

    void SetMap(const Map::Ptr &map){
        mMap = map;
    }

    void LoopRunning();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    cv::Mat PlotFrameImage(Frame::Ptr Frame);

    void DrawFrame(Frame::Ptr frame, const float* color);

    void DrawKFsAndMPs(const bool menuShowKeyFrames, const bool menuShowPoints);

    void Close(){
        mViewerIsRunning.store(false);
        mthreadViewer.join();
    }

    void AddCurrentFrame(const Frame::Ptr &frame){
        std::unique_lock<std::mutex> lock(mViewerDataMutex);
        mCurrentFrame = frame;
    }

    void DrawLine();

public:
    typedef std::shared_ptr<Viewer> Ptr;

private:
    std::thread mthreadViewer;
    std::atomic<bool> mViewerIsRunning;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
    float mGraphLineWidth, mPointSize, mCameraSize, mCameraLineWidth, mKeyFrameSize,mKeyFrameLineWidth;

    std::mutex mViewerDataMutex;

    Frame::Ptr mCurrentFrame;
    Frame::Ptr mLastFrame;

    Map::Ptr mMap;

    const float red[3] = {1, 0, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
};

}

#endif //MYSLAM_VIEWER_H
