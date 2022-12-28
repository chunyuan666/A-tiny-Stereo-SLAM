#ifndef _BACKEND_H
#define _BACKEND_H

#include "Common.h"
#include "KeyFrame.h"
#include "Map.h"
#include "Optimizer.h"
#include "LoopClosing.h"

namespace myslam{
 

class Backend{
public:
    Backend();
    void Running();

    void Stop(){
        mbNeedOptimize = false;
        mbBackendIsRunning.store(false);
        mBackendHandle.join();
    }

    void InsertKeyFrame(const KeyFrame::Ptr &kf){
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        mlpNewkeyFrames.push_back(kf);
    }

    bool CheckNewKeyFrame(){
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return (!mlpNewkeyFrames.empty());
    }

    void ProcessKeyFrame();

    void SetMap(const Map::Ptr &map){
        mMap = map;
    }

    void SetCamera(const Camera::Ptr &cam){
        mCameraLeft = cam;
    }

    void SetLoopClosing(const LoopClosing::Ptr &lp){
        mLoopClosing = lp;
    }

    void RequestPause(){
        mbRequestPause.store(true);
    }

    bool HasPaused(){
        return mbRequestPause.load() && mbHasPause.load();
    }
    
    void Resume(){
        mbRequestPause.store(false);
    }

public:
    typedef std::shared_ptr<Backend> Ptr;
    bool mbNeedOptimize = false;
    
private:
    std::list<KeyFrame::Ptr> mlpNewkeyFrames;
    KeyFrame::Ptr mCurrentKeyFrame;
    Camera::Ptr mCameraLeft;
    Map::Ptr mMap;

    std::atomic<bool> mbBackendIsRunning, mbRequestPause, mbHasPause;

    std::mutex mMutexNewKFs;

    std::thread mBackendHandle;

    LoopClosing::Ptr mLoopClosing;
};


}
#endif