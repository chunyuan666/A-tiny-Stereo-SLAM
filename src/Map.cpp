#include "Common.h"
#include "Map.h"

namespace myslam{
    
    void Map::InserMapPoint(const MapPoints::Ptr &MapPoint){
        std::unique_lock<std::mutex> lock(mMutexData);
        //还没有存在地图中
        if(mmAllMapPoints.find(MapPoint->mid) == mmAllMapPoints.end()){
            mmAllMapPoints.insert({MapPoint->mid, MapPoint});
        }else{  // 在地图里了
            mmAllMapPoints[MapPoint->mid] = MapPoint;
        }
    }

    void Map::InserActivateMapPoints(const MapPoints::Ptr &MapPoint){
        std::unique_lock<std::mutex> lock(mMutexData);
        //还没有存在地图中
        if(mmActivateMapPoints.find(MapPoint->mid) == mmActivateMapPoints.end()){
            mmActivateMapPoints.insert({MapPoint->mid, MapPoint});
        }else{  // 在地图里了
            mmActivateMapPoints[MapPoint->mid] = MapPoint;
        }
    }

    void Map::InserKeyFrame(const KeyFrame::Ptr &kf){
        mpCurrentKeyFrame = kf;
        {
            std::unique_lock<std::mutex> lock(mMutexData);
            //还没有存在地图中
            if(mmAllKeyFrames.find(kf->mKeyFrameId) == mmAllKeyFrames.end()){
                mmAllKeyFrames.insert(std::make_pair(kf->mKeyFrameId, kf));
                mmActivateKeyFrames.insert(std::make_pair(kf->mKeyFrameId, kf));
            }else{ // 在地图里了
                mmAllKeyFrames[kf->mKeyFrameId] = kf;
                mmActivateKeyFrames[kf->mKeyFrameId] = kf;
            }
        }
        //　插入地图点
        for(auto &fea : kf->mvpFeatureLeft){
            auto map = fea->mpMapPoint.lock();
            if(map){
                map->AddActivateObservation(kf->mKeyFrameId, fea);
                InserActivateMapPoints(map);
            }
        }

        // 如果活跃帧大于阈值10．则删除一些活跃帧
        if(mmActivateKeyFrames.size() > mNumActivateMap){
            CullOldActivateKF();
            CullOldActivateMapPoint();
        }
    }

    void Map::CullOldActivateKF(){
        std::unique_lock<std::mutex> lock(mMutexData);
        if(!mpCurrentKeyFrame)
            return;
        auto T_wc = mpCurrentKeyFrame->GetPose().inverse();
        double maxDistance = 0; unsigned long maxID = 0;
        double minDistance = 999.9; unsigned long minID = 0;
        for(auto &kf : mmActivateKeyFrames){
            if(kf.first == mpCurrentKeyFrame->mKeyFrameId)
                continue;
            auto distance = (kf.second->GetPose() * T_wc).log().norm();
            if(distance > maxDistance){
                maxDistance = distance;
                maxID = kf.first;
            }
            if(distance < minDistance){
                minDistance = distance;
                minID = kf.first;
            }
        }
        unsigned long toDelId;
        double min_th = 0.8;
        if(minDistance < min_th){
            toDelId = minID;
        }else{
            toDelId = maxID;
        }
        // 取消地图点对该帧的观测
        for(auto &feat : mmActivateKeyFrames[toDelId]->mvpFeatureLeft){
            auto map = feat->mpMapPoint.lock();
            if(map){
                map->RemoveActiveObservation(feat);
            }
        }
        // 删除关键帧
        mmActivateKeyFrames.erase(toDelId);
        // LOG(INFO) << "----------remove keyframe ID: " << toDelId << "--------------";
    }
    
    //　删除观测为0的局部地图点
    void Map::CullOldActivateMapPoint() {
        int cnt_landmark_removed = 0;
        std::unique_lock<std::mutex> lock(mMutexData);
        for(auto iter = mmActivateMapPoints.begin(); iter!=mmActivateMapPoints.end();){
            if(iter->second->GetActivateObsCnt() == 0){
                mmActivateMapPoints.erase(iter++);
                cnt_landmark_removed++;
            }else{
                iter++;
            }
        }
        LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks  "
                  << "remaining  " << mmActivateMapPoints.size();
    }

    void Map::RemoveOutlierMapPoints() {
        std::unique_lock<std::mutex> lock(mMutexData);
        for(auto iter = mmActivateMapPoints.begin(); iter != mmActivateMapPoints.end();){
            auto mp = iter->second;
            if(mp->mbIsOutlier){
                mmActivateMapPoints.erase(iter++);
            }else{
                iter++;
            }
        }
        int cnt = 0;
        for(auto iter = mmAllMapPoints.begin(); iter != mmAllMapPoints.end();){
            auto mp = iter->second;
            if(mp->mbIsOutlier){
                mmAllMapPoints.erase(iter++);
                cnt++;
            }else{
                iter++;
            }
        }
        LOG(INFO) << "一共移除 " << cnt << "个外点";
    }

    void Map::RemoveMapPoints(const MapPoints::Ptr &MapPoint){
        std::unique_lock<std::mutex> lock(mMutexData);
        unsigned long mapId = MapPoint->mid;
        mmAllMapPoints.erase(mapId);
        mmActivateMapPoints.erase(mapId);
    }

}