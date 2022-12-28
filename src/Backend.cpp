#include "Backend.h"
#include "KeyFrame.h"
#include "Map.h"

namespace myslam{

    Backend::Backend(){
        mbNeedOptimize = false;
        mbBackendIsRunning.store(true);
        mBackendHandle = std::thread(std::bind(&Backend::Running,  this));
    }


    void Backend::Running(){
        while(mbBackendIsRunning.load()){
            while(CheckNewKeyFrame()){
                // LOG(INFO) << "+++++++++++++++++++begin process KF+++++++++";
                ProcessKeyFrame();
                // LOG(INFO) << "+++++++++++++++++++process KF finished!+++++++++";
            }

            if(mbRequestPause.load()){
                LOG(INFO) << "暂停后端优化";
                while(mbRequestPause.load()){
                    mbHasPause.store(true);
                    usleep(1000);
                }
            }
            mbHasPause.store(false);

            //　局部优化位姿和地图
            if(!CheckNewKeyFrame() && mbNeedOptimize){
                // LOG(INFO) << "**********开始LocalMap优化***********";
                Optimizer::OptimizeActivateMap(mMap, mCameraLeft);
                // LOG(INFO) << "**********结束LocalMap优化***********";
                mbNeedOptimize = false;
            }
            usleep(1000);
        }
        std::cout << "BackendOptimization Stop!" << std::endl;
    }

    void Backend::ProcessKeyFrame(){
        {
            std::unique_lock<std::mutex> lock(mMutexNewKFs);
            mCurrentKeyFrame = mlpNewkeyFrames.front();
            mlpNewkeyFrames.pop_front();
            //LOG(INFO) << "pop keyframe id: " << mCurrentKeyFrame->mKeyFrameId;
        }
        mMap->InserKeyFrame(mCurrentKeyFrame);
        // if(mMap->GetActivateKeyFrames().size() >= mMap->mNumActivateMap)
            mbNeedOptimize = true;
        /***TODO**/
        //插入到回环线程中
        mLoopClosing->InserKeyFrame(mCurrentKeyFrame);

    }

}