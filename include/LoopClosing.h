#ifndef MYSLAM_LOOPCLOSING_H
#define MYSLAM_LOOPCLOSING_H

#include "Common.h"
#include "KeyFrame.h"
#include "DeepLCD.h"
#include "ORBExtractor.h"
#include "g2o_types.h"
#include "Map.h"

namespace myslam{

    class Backend;

class LoopClosing{
public:
    LoopClosing(int nLevels, float simThe1, float simThe2);

    void InserKeyFrame(KeyFrame::Ptr &kf){
        std::unique_lock<std::mutex> lck(mmutexNewKF);
        if(mLastCloseKF == nullptr || kf->mKeyFrameId-mLastCloseKF->mKeyFrameId > 5){
            mlpNewKF.push_back(kf);
        }
        
    }
    
    void setORBExtractor(ORBExtractor::Ptr &orbextractor){
        mpORBextractor = orbextractor;
    }

    void setMap(Map::Ptr &map){
        mMap = map;
    }

    void setBackend(std::shared_ptr<Backend> backend){
        mBackend = backend;
    }

    void Stop(){
        mbLoopRunning.store(false);
        mthreadLoopClosing.join();
    }

    bool checkNewKF();

    bool ProcessKF();

    void Running();

    void ProcessByPyramid();

    bool CaluCorrectedPose();

    bool FindLoopcloseKF();

    int OptmizeCurKFPose();

    void CorrectPose();

    
public:
    typedef std::shared_ptr<LoopClosing> Ptr;
     

private:
    std::shared_ptr<DeepLCD> mDeepLCD;
    std::mutex mmutexNewKF;
    KeyFrame::Ptr mLastKF, mCurKF, mLastCloseKF;
    std::list<KeyFrame::Ptr> mlpNewKF;
    KeyFrame::Ptr mLoopcloseKF;
    std::map<unsigned long, KeyFrame::Ptr> KFDB;
    std::map<int, int> mValidFeaMatches;
    SE3 mCorrectedCurPose;
    bool mbNeedCorrect;


    int nLevels;
    float similarThre1, similarThre2;

    std::thread mthreadLoopClosing;

    ORBExtractor::Ptr mpORBextractor;

    Map::Ptr mMap;

    std::weak_ptr<Backend> mBackend;

    std::atomic<bool> mbLoopRunning;

};


}

#endif