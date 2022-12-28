//
// Created by yuan on 2022/7/8.
//

#ifndef MYSLAM_MAPPOINTS_H
#define MYSLAM_MAPPOINTS_H

#include "Common.h"
#include "Feature.h"


namespace myslam{
    class Feature;
    class MapPoints {
    public:

        MapPoints(){
            mid = nLastId++;
        }

        MapPoints(const Vector3d& pose){
            mCoordinate = pose;
            mid = nLastId++;
        }

        void SetPose(const Vector3d &pose){
            mCoordinate = pose;
        }

        Vector3d GetPose(){
            return mCoordinate;
        }

        std::map<unsigned long, std::weak_ptr<Feature> > GetActivateObservation(){
            std::unique_lock<std::mutex> lock(mObservationMutex);
            return mActiveObservation;
        }

        std::map<unsigned long, std::weak_ptr<Feature> > GetObservation(){
            std::unique_lock<std::mutex> lock(mObservationMutex);
            return mObservation;
        }

        void AddObservation(unsigned long id, const std::weak_ptr<Feature>& feat){
            std::unique_lock<std::mutex> lock(mObservationMutex);
            mObservation.insert({id, feat});
            mObservedCnt++;
        }
       
        void AddActivateObservation(unsigned long id, const std::shared_ptr<Feature>& feat){
            std::unique_lock<std::mutex> lock(mObservationMutex);
            mActiveObservation.insert({id, feat});
            mActiveObesedrCnt++;
        }


        void RemoveActiveObservation(const std::shared_ptr<Feature>& feat){
            std::unique_lock<std::mutex> lock(mObservationMutex);
            for(auto iter = mActiveObservation.begin();iter!=mActiveObservation.end();iter++){
                if(iter->second.lock() == feat){
                    mActiveObservation.erase(iter);
                    mActiveObesedrCnt--;
                    break;
                }
            }
        }

        void RemoveObservation(std::shared_ptr<Feature> &feature){
            std::unique_lock<std::mutex> lock(mObservationMutex);
            for(auto iter = mObservation.begin(); iter != mObservation.end(); iter++){
                if(iter->second.lock() == feature){
                    mObservation.erase(iter);
                    mObservedCnt--;
                    break;
                }
            }
        }

        unsigned long GetActivateObsCnt() {
            std::unique_lock<std::mutex> lck(mObservationMutex);
            return mActiveObesedrCnt;
        }

        unsigned long GetObsCnt() {
            std::unique_lock<std::mutex> lock(mObservationMutex);
            return mObservedCnt;
        }


    public:
        typedef std::shared_ptr<MapPoints> Ptr;
        unsigned long mid;
        static unsigned long nLastId;

        //　估计位姿时生成判断该点是否为异常点
        bool mbIsOutlier = false;
    private:
        Vector3d mCoordinate = Vector3d::Zero();
        // 第一个为关键帧ID, 第二个为特征点
        std::map<unsigned long, std::weak_ptr<Feature> > mObservation;
        std::map<unsigned long, std::weak_ptr<Feature> > mActiveObservation;

        unsigned long mObservedCnt = 0;
        unsigned long mActiveObesedrCnt = 0;

        std::mutex mObservationMutex;
    };
}



#endif //MYSLAM_MAPPOINTS_H
