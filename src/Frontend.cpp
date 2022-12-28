#include "Frontend.h"
#include "Common.h"


namespace myslam {
    bool Frontend::Running(const cv::Mat &leftImg, const cv::Mat &rightImg, const double timestamp) {
        mCurrentFrame = std::make_shared<Frame>(leftImg, rightImg, timestamp, mCameraLeft->GetK());
        // LOG(INFO) << "第" << mCurrentFrame->mFrameId << "帧";
/*
         switch (mStatus) {
            case TrackStatus::INIT:
                if(StereoInit()){
                    mStatus = TrackStatus::GOOD;
                }
                break;
            case TrackStatus::GOOD:
            case TrackStatus::BAD:
                Tracking();
                break;
            case TrackStatus::LOST:
                LOG(INFO) << "LOST!";
                // while(1){}
                return false;
                break;
        }
*/     
        Track();
        // mLastFrame = mCurrentFrame;
        return true;
    }

    /**
     * @brief 双目初始化
     * @details 
     * 1. 提取左目图像的特征点
    */
    bool Frontend::StereoInit() {
        // 提取左目图像的特征点
        DetectFeature();
        // 匹配右图特征点
        unsigned long cnt = MatchFeaturesInRight();
        if(cnt < mnFeaturesInitGood )
            return false;
        //　初始化地图
        unsigned long num_maps = MapInit();
        if( num_maps < mnFeaturesTrackingBad )
            return false;
        // 将初始化帧作为关键帧插入
        InsertKeyFrame();
        LOG(INFO) << "map init finished! sum " << num_maps << " map points.";
        return true;
    }

    /**
     * @brief 提取特征点
    */
    unsigned long Frontend::DetectFeature() {
        // 制作掩膜
        cv::Mat mask(mCurrentFrame->mImgLeft.size(), CV_8UC1, 255);
        for (auto feature: mCurrentFrame->mvpFeatureLeft) {
            cv::rectangle(mask, feature->mKeyPoint.pt - cv::Point2f(20, 20),
                          feature->mKeyPoint.pt + cv::Point2f(20, 20), 0, cv::FILLED);
        }
        std::vector<cv::KeyPoint> KeyPoints;
        if (mStatus == INIT) {
            mORBExtractorInit->Detect(mCurrentFrame->mImgLeft, mask, KeyPoints);
        } else {
            mORBExtractor->Detect(mCurrentFrame->mImgLeft, mask, KeyPoints);
        }
        // cv::Mat outimg;
        // cv::drawKeypoints(img_clahe, KeyPoints, outimg,
        //                   cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        // std::string id = IntToString(mCurrentFrame->mFrameId);
        // putText(outimg, id, cv::Point(0, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 0), 2, 8, 0);
        // cv::imshow("detect", outimg);
        // cv::waitKey(1);
        unsigned long cnt = 0;
        for (auto &KeyPoint: KeyPoints) {
            Feature::Ptr feature = std::make_shared<Feature>(KeyPoint);
            mCurrentFrame->mvpFeatureLeft.push_back(feature);
            cnt++;
        }
        // LOG(INFO) << "一共提取新的特征点个数为" << cnt;
        return cnt;
    }

    unsigned long Frontend::MatchFeaturesInRight() {
        std::vector<cv::Point2f> vPointFeaLeft, vPointFeaRight;
        // 准备左右目的特征点
        for (auto &feature: mCurrentFrame->mvpFeatureLeft) {
            vPointFeaLeft.push_back(feature->mKeyPoint.pt);
            auto map = feature->mpMapPoint.lock();
            if (map) {
                auto PointFeaRight = mCameraRight->World2Pixel(map->GetPose(),mCurrentFrame->GetPose());
                if(isInBorder(PointFeaRight)) vPointFeaRight.push_back(PointFeaRight);
                else vPointFeaRight.push_back(feature->mKeyPoint.pt);
            } else {
                vPointFeaRight.push_back(feature->mKeyPoint.pt);
            }
        }
        //　利用光流匹配右目
        std::vector<uchar> status;
        cv::Mat error;
        // LOG(INFO) << "左右目光流追踪。";
        cv::calcOpticalFlowPyrLK(mCurrentFrame->mImgLeft, mCurrentFrame->mImgRight,
                                 vPointFeaLeft, vPointFeaRight, status, error, cv::Size(11, 11),
                                 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        
/*        
        mCurrentFrame->mvpFeatureRight.resize(vPointFeaLeft.size(), nullptr);
        int cnt = 0;
        for(int i = 0; i < status.size(); i++){
            if(status[i]){
                cnt++;
                mCurrentFrame->mvpFeatureRight[i] = (std::make_shared<Feature>(cv::KeyPoint(vPointFeaRight[i],7)));
            }
        }
        assert(!mCurrentFrame->mvpFeatureRight.empty());
*/

        // 剔除左目的无法匹配的特征点
        reduceVector(mCurrentFrame->mvpFeatureLeft, status);
        // 利用RANSAC剔除错误匹配
        reduceVector(vPointFeaLeft, status);
        reduceVector(vPointFeaRight, status);
        std::vector<uchar> RansacStatus;
        cv::Mat Fundamental = cv::findFundamentalMat(vPointFeaLeft, vPointFeaRight, RansacStatus, cv::FM_RANSAC);
        reduceVector(vPointFeaLeft, RansacStatus);
        reduceVector(vPointFeaRight, RansacStatus);
        reduceVector(mCurrentFrame->mvpFeatureLeft, RansacStatus);
        mCurrentFrame->mvpFeatureRight.clear();
        int cnt = 0;
        for(int i = 0; i < vPointFeaLeft.size(); i++){
            mCurrentFrame->mvpFeatureRight.push_back(std::make_shared<Feature>(cv::KeyPoint(vPointFeaRight[i],7)));
            cnt++;
        }

        int cnt_map = 0;
        for(auto fea : mCurrentFrame->mvpFeatureLeft){
            if(!fea->mpMapPoint.expired()){
                cnt_map++;
            }
        }

// 可视化
#if 0
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat img_match; std::vector<cv::DMatch> matches;
        for(int i = 0; i < vPointFeaLeft.size(); i++){
            keypoints_1.push_back(cv::KeyPoint(vPointFeaLeft[i],7));
            keypoints_2.push_back(cv::KeyPoint(vPointFeaRight[i],7));
            matches.push_back(cv::DMatch(i, i, 10));
        }
        assert(!keypoints_1.empty());
        assert(!keypoints_2.empty());
        cv::drawMatches(mCurrentFrame->mImgLeft, keypoints_1, mCurrentFrame->mImgRight, keypoints_2, matches, img_match);
        std::string idCur = IntToString(mCurrentFrame->mFrameId);
        std::string text = "第" + idCur + "帧左->右"; 
        cv::imshow("text", img_match);
        cv::waitKey(0);
        // cv::destroyWindow(text);
#endif
        // LOG(INFO) << "一共匹配右目" << cnt << "个特征点,其中包含地图点的个数为：" << cnt_map;
        return cnt;
    }

    //　初始化地图
    unsigned long  Frontend::MapInit() {
        // 创造新的地图点
        auto nGoodPoints = CrateNewMapPoints();
        // 设第一帧的位姿为[R|t] = [I | 0]
        mCurrentFrame->SetPose(SE3_Identity);
        return nGoodPoints;
    }      

    unsigned long Frontend::RefinePose(SE3 &Tcw_rough){
        // 匹配特征点,用光流法匹配，追踪上一帧和当前帧,当前帧得到特征点和地图点
        MatchLastFrameByLKFlow();
#if 0
        cv::Mat r, t;
        std::vector<cv::Point3f> pts_3d;
        std::vector<cv::Point2f> pts_2d;
        for(auto fea : mCurrentFrame->mvpFeatureLeft){
            if(fea->mpMapPoint.expired()) continue;
            auto kp = fea->mKeyPoint;
            auto mp = fea->mpMapPoint.lock()->GetPose();
            // Vec4d mp_4d(mp.matrix()[0], mp.matrix()[1], mp.matrix()[2], 1);
            // Vec3d point3d = Tcw_rough.matrix3x4() * mp_4d;
            // pts_3d.push_back(cv::Point3d(point3d[0], point3d[1], point3d[2]));
            pts_3d.push_back(cv::Point3f(mp[0], mp[1], mp[2]));
            pts_2d.push_back(kp.pt);
        }
        if(pts_3d.empty()) return 0;
        cv::Mat K;
        cv::eigen2cv(mCurrentFrame->mK, K);
        // cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
        cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
        cv::Mat R;
        cv::Rodrigues(r, R);
        // LOG(INFO) << "R=\n" << R << "\nt=" << t;
        Mat33d R_eigen;
        Vec3d t_eigen;
        cv::cv2eigen(R, R_eigen);
        cv::cv2eigen(t, t_eigen);
        SE3 Tcw_pnp(R_eigen, t_eigen);
        // LOG(INFO) << "Tcw_pnp: \n" << Tcw_pnp.matrix3x4();
        // LOG(INFO) << "Tcw_rough: \n" << Tcw_rough.matrix3x4();
        // Sophus::SE3d pose_gn;
#endif
        mCurrentFrame->SetPose(Tcw_rough);
        auto InLiers = Optimizer::PoseOptimization(mCurrentFrame);
        //　剔除外点
        for(auto &fea : mCurrentFrame->mvpFeatureLeft){
            if(!fea->IsOutLier)
                continue;
            auto mapPoint = fea->mpMapPoint.lock();
            if(mapPoint){
                mapPoint->mbIsOutlier = true; //先标记，在后端优化时再删除
                fea->mpMapPoint.reset();
            }
            fea->IsOutLier = false;
        }
        return InLiers;
    }  

    bool Frontend::TrackWithMotionModel(){
        // LOG(INFO) << "Trackoing中的上一帧ID:" << mLastFrame->mFrameId << "位姿:\n" << mLastFrame->GetPose().matrix3x4();
        // LOG(INFO) << "初始化第" << mCurrentFrame->mFrameId<< "帧的位姿：";
        // mCurrentFrame->SetPose( mVelocity * mLastFrame->GetPose());

        // 1、利用参考关键帧更新上一帧在世界坐标系下的位姿
        auto LastRefKF = mLastFrame->mReferenceKF;
        SE3 Tlr = mRelativeToRefPose.back();
        mLastFrame->SetPose(Tlr * LastRefKF->GetPose());
        SE3 Tcw_rough = mVelocity * mLastFrame->GetPose();
        mCurrentFrame->SetPose(Tcw_rough);
        // LOG(INFO) << "\n" <<  Tcw_rough.matrix3x4();
        if(!MatchLastFrameByLKFlow())
            return false;
        auto InLiers = Optimizer::PoseOptimization(mCurrentFrame);
        // auto trackingInliers = RefinePose(Tcw_rough);
        // LOG(INFO) << "估计第 " << mCurrentFrame->mFrameId << " 帧位姿所得内点" << "trackingInliers : " << InLiers;
        // LOG(INFO) << "估计第:" << mCurrentFrame->mFrameId << " 帧位姿:\n" << mCurrentFrame->GetPose().matrix3x4();
        return InLiers>=mnFeaturesTrackingBad;
    }
    
    bool Frontend::TrackWithReferenceKF(){
        LOG(INFO) << "参考关键帧追踪";
        // 1、将上一帧的位姿作为初始位态
        mCurrentFrame->SetPose(mLastFrame->GetPose());
        // 2、通过光流跟踪特征点
        SearchByLKFlow(mReferenceKF, mCurrentFrame);
        // 3、通过优化3D-2D的重投影误差来获得精确的位姿
        auto trackingInliers = Optimizer::PoseOptimization(mCurrentFrame);
        LOG(INFO) << "估计第 " << mCurrentFrame->mFrameId << " 帧位姿所得内点" << "trackingInliers : " << trackingInliers;
        // LOG(INFO) << "估计第:" << mCurrentFrame->mFrameId << " 帧位姿:\n" << mCurrentFrame->GetPose().matrix3x4();
        return trackingInliers>=mnFeaturesTrackingBad;
    }

    unsigned long Frontend::SearchByLKFlow(KeyFrame::Ptr &kf, Frame::Ptr &frame){
        std::vector<cv::Point2f> vFeaPointsKF, vFeaPointsFrame;
        std::vector<std::weak_ptr<MapPoints>> vpKFMaps; //记录上一帧的地图点
        for(auto &fea : kf->mvpFeatureLeft){
            if(fea->mpMapPoint.expired()==false && fea->mpMapPoint.lock()->mbIsOutlier==false){
                vFeaPointsKF.push_back(fea->mKeyPoint.pt);
                vpKFMaps.push_back(fea->mpMapPoint);
                auto mapPoint = fea->mpMapPoint.lock();
                cv::Point2f pointCurrent = mCameraLeft->World2Pixel(mapPoint->GetPose(),frame->GetPose());
                vFeaPointsFrame.push_back(pointCurrent);
            }else{
                vFeaPointsKF.push_back(fea->mKeyPoint.pt);
                vFeaPointsFrame.push_back(fea->mKeyPoint.pt);
                vpKFMaps.push_back(std::make_shared<MapPoints>());
            }
        }
        if(vFeaPointsKF.empty()){
            return 0;
        } 
        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(kf->mImgLeft, frame->mImgLeft,
                                 vFeaPointsKF, vFeaPointsFrame, status, error, cv::Size(11, 11),
                                 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        // cv::calcOpticalFlowPyrLK(mLastFrame->mImgLeft, mCurrentFrame->mImgLeft,
        //                          vFeaPointsLast, vFeaPointsCurrent, status, error, cv::Size(21, 21),
        //                          3);
        for(int i = 0; i < vFeaPointsFrame.size(); i++){
            if(status[i] && !isInBorder(vFeaPointsFrame[i]))
                status[i] = 0;
        }
        
        reduceVector(vpKFMaps, status);
        // 利用RANSAC剔除错误匹配
        reduceVector(vFeaPointsKF, status);
        reduceVector(vFeaPointsFrame, status);
        std::vector<uchar> RansacStatus;
        cv::Mat Fundamental = cv::findFundamentalMat(vFeaPointsKF, vFeaPointsFrame, RansacStatus, cv::FM_RANSAC);
        
        reduceVector(vFeaPointsKF, RansacStatus);
        reduceVector(vFeaPointsFrame, RansacStatus);
        reduceVector(vpKFMaps, RansacStatus);
        frame->mvpFeatureLeft.clear();
        for(int i = 0; i < vFeaPointsFrame.size(); i++){
            if(vpKFMaps[i].expired())
                continue;
            auto feature = std::make_shared<Feature>(cv::KeyPoint(vFeaPointsFrame[i],7));
            feature->mpMapPoint = vpKFMaps[i];
            frame->mvpFeatureLeft.push_back(feature);
        }
// 可视化
#if 0
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat img_match; std::vector<cv::DMatch> matches;
        for(int i = 0; i < vFeaPointsLast.size(); i++){
            keypoints_1.push_back(cv::KeyPoint(vFeaPointsLast[i], 7));
            keypoints_2.push_back(cv::KeyPoint(vFeaPointsCurrent[i], 7));
            matches.push_back(cv::DMatch(i, i, 0));
        }
        cv::drawMatches(mCurrentFrame->mImgLeft, keypoints_1, mCurrentFrame->mImgRight, keypoints_2, matches, img_match);
        std::string idLast = IntToString(mLastFrame->mFrameId);
        std::string idCur = IntToString(mCurrentFrame->mFrameId);
        std::string text = idLast + "->" + idCur; 
        cv::imshow(text, img_match);
        cv::waitKey(0);
        cv::destroyWindow(text);
#endif
        LOG(INFO) << "追踪到一共 "<<frame->mvpFeatureLeft.size()<<" 个地图点";
        return frame->mvpFeatureLeft.size();
    }

    void Frontend:: Track(){
        if(mStatus == INIT){
            if(StereoInit()){
                mStatus = OK;
            }else{
                mStatus = INIT;
                return;
            };
        }
        else{
            bool bOK;
            if(mStatus == OK){
                if(mVelocity.log().norm()==0.0){
                    bOK = TrackWithReferenceKF();
                }else{
                    bOK = TrackWithMotionModel();
                    if(!bOK){
                        bOK = TrackWithReferenceKF();
                    }
                }
            }else{
                // 重定位
                LOG(INFO) << "LOST!!!";
                exit(0);
            }

            if(bOK)
                mStatus = OK;
            else
                mStatus = LOST;

            if(mStatus == OK){
                // 更新速度
                mVelocity = mCurrentFrame->GetPose() * mLastFrame->GetPose().inverse();
                //　剔除外点
                for(auto &fea : mCurrentFrame->mvpFeatureLeft){
                    if(!fea->IsOutLier)
                        continue;
                    auto mapPoint = fea->mpMapPoint.lock();
                    if(mapPoint){
                        mapPoint->mbIsOutlier = true; 
                        fea->mpMapPoint.reset();
                    }
                    fea->IsOutLier = false;
                }
                // 检查是否需要插入关键帧
                if(CheckIsInserKF()){
                    // 查找新的特征点
                    DetectFeature();
                    // 匹配右目特征点
                    MatchFeaturesInRight();
                    //　三角化新的地图点
                    CrateNewMapPoints();
                    //　插入关键帧
                    InsertKeyFrame();
                }
                
            }
        }
        
        mCurrentFrame->mReferenceKF = mReferenceKF;
        SE3 Tcr = mCurrentFrame->GetPose()*mCurrentFrame->mReferenceKF->GetPose().inverse();
        mRelativeToRefPose.push_back(Tcr);
        mLastFrame = mCurrentFrame;
        if(mViewer)   
            mViewer->AddCurrentFrame(mCurrentFrame);
    }

    // 恒速运动模式追踪
/*
    bool Frontend::Tracking(){
        auto trackingInliers = MotionTracking();
        if(trackingInliers < mnFeaturesTrackingBad){
           trackingInliers = ReferenceKFTracking();
        }
        if(trackingInliers >= mnFeaturesTrackingGood ){
            mStatus = TrackStatus::GOOD;
        }else if(trackingInliers >= mnFeaturesTrackingBad && trackingInliers < mnFeaturesTrackingGood){
            mStatus = TrackStatus::BAD;
        }else{
            mStatus = TrackStatus::LOST;
        } 

        if(mStatus == TrackStatus::BAD){
            // 查找新的特征点
            DetectFeature();
            // 匹配右目特征点
            MatchFeaturesInRight();
            //　三角化新的地图点
            CrateNewMapPoints();
            //　插入关键帧
            InsertKeyFrame();
        }

        mVelocity = mCurrentFrame->GetPose() * mLastFrame->GetPose().inverse();
        mCurrentFrame->mReferenceKF = mReferenceKF;
        SE3 Tcr = mCurrentFrame->GetPose()*mCurrentFrame->mReferenceKF->GetPose().inverse();
        mRelativeToRefPose.push_back(Tcr);
        // LOG(INFO) << "---------MOtion 更新----------\n" ;
        if(mViewer)   
            mViewer->AddCurrentFrame(mCurrentFrame);
        return false;
    }
*/
    // 创造新的地图点
    unsigned long Frontend::CrateNewMapPoints(){
        // 1. 三角化形成地图点
        cv::KeyPoint KpL, KpR;
        const float b = mCameraLeft->mbaseline;
        const float f = mCameraLeft->mfx;
        const float invfx = 1 / mCameraLeft->mfx;
        const float invfy = 1 / mCameraLeft->mfy;
        const float cx = mCameraLeft->mcx;
        const float cy = mCameraLeft->mcy;
        const float minD = 0;
        const float maxD = f;
        unsigned long nGoodPoints = 0;
        int nBadPoints = 0;
        for(unsigned long i = 0; i < mCurrentFrame->mvpFeatureLeft.size(); i++){
            // if(!mCurrentFrame->mvpFeatureRight[i])
            //     continue;
            auto map = mCurrentFrame->mvpFeatureLeft[i]->mpMapPoint.lock();
            if(map)  // 该特征点已有地图点，不再添加
                continue;
            KpL = mCurrentFrame->mvpFeatureLeft[i]->mKeyPoint;
            KpR = mCurrentFrame->mvpFeatureRight[i]->mKeyPoint;
/*
            SE3 Tlw = mCurrentFrame->GetPose();
            SE3 Tlr = mCameraRight->GetCameraPose();
            SE3 Trw = Tlr.inverse() * Tlw;            // 相机2的pose为Tlr
            Mat33d K = mCameraLeft->GetK();
            Mat34d P1 = K * Tlw.matrix3x4();
            Mat34d P2 = K * Trw.matrix3x4();
            Vector3d X3D; 
            Triangulation(P1, P2, KpL, KpR, X3D);

            if(X3D[2] < 0){
                nBadPoints++;
                continue;
            }
*/          
            float uL = KpL.pt.x;
            float uR = KpR.pt.x;
            float depth = 0;
            float disparity = 0.0;
            disparity = uL-uR;
            if(disparity >= minD && disparity < maxD){
                depth = b * f / disparity;
            }
            if(depth <=0){
                nBadPoints++;
                continue;
            }

            const float u = KpL.pt.x;
            const float v = KpL.pt.y;
            const float x = (u-cx) * depth * invfx;
            const float y = (v-cy) * depth * invfy;
            Vector3d X3Dc;
            X3Dc << x, y, depth;
            Vector3d X3Dw = mCurrentFrame->GetPose().inverse() * X3Dc;
            nGoodPoints++;
            // 2. 创建新的地图点
            MapPoints::Ptr NewPoint =  std::make_shared<MapPoints>(X3Dw);
            mCurrentFrame->mvpFeatureLeft[i]->mpMapPoint = NewPoint;
            mMap->InserMapPoint(NewPoint);
        }
        unsigned long sum = 0;
        for(auto fea : mCurrentFrame->mvpFeatureLeft){
            if(fea->mpMapPoint.expired()==false)
                sum++;
        }
        // LOG(INFO) << "新增" << nGoodPoints << "个" << "地图点,一共有" << sum <<"个,坏点有 " << nBadPoints;
        return nGoodPoints;
    }

    //　估计当前帧位姿
    // unsigned long Frontend::EstimatePose(){
    //     // 计算位姿
    //     if(mCurrentFrame->mvpFeatureLeft.size()==0) return 0;
    //     auto InLiers = Optimizer::PoseOptimization(mCurrentFrame);
    //     //　剔除外点
    //     for(auto &fea : mCurrentFrame->mvpFeatureLeft){
    //         if(!fea->IsOutLier)
    //             continue;
    //         auto mapPoint = fea->mpMapPoint.lock();
    //         if(mapPoint){
    //             mapPoint->mbIsOutlier = true;
    //             fea->mpMapPoint.reset();
    //         }
    //         fea->IsOutLier = false;
    //     }
    //     return InLiers;
    // }

    // 光流匹配
    bool Frontend::MatchLastFrameByLKFlow(){
        std::vector<cv::Point2f> vFeaPointsLast, vFeaPointsCurrent;
        std::vector<std::weak_ptr<MapPoints>> vpLastMaps; //记录上一帧的地图点
        for(auto &fea : mLastFrame->mvpFeatureLeft){
            if(fea->mpMapPoint.expired()==false && fea->mpMapPoint.lock()->mbIsOutlier==false){
                vFeaPointsLast.push_back(fea->mKeyPoint.pt);
                vpLastMaps.push_back(fea->mpMapPoint);
                auto mapPoint = fea->mpMapPoint.lock();
                cv::Point2f pointCurrent = mCameraLeft->World2Pixel(mapPoint->GetPose(),mCurrentFrame->GetPose());
                vFeaPointsCurrent.push_back(pointCurrent);
            }else{
                vFeaPointsLast.push_back(fea->mKeyPoint.pt);
                vFeaPointsCurrent.push_back(fea->mKeyPoint.pt);
                vpLastMaps.push_back(std::make_shared<MapPoints>());
            }
        }
        if(vFeaPointsLast.empty()){
            // LOG(INFO) << "vFeaPointsLast为空";
            return false;
        } 
        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(mLastFrame->mImgLeft, mCurrentFrame->mImgLeft,
                                 vFeaPointsLast, vFeaPointsCurrent, status, error, cv::Size(11, 11),
                                 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        // cv::calcOpticalFlowPyrLK(mLastFrame->mImgLeft, mCurrentFrame->mImgLeft,
        //                          vFeaPointsLast, vFeaPointsCurrent, status, error, cv::Size(21, 21),
        //                          3);
        for(int i = 0; i < vFeaPointsCurrent.size(); i++){
            if(status[i] && !isInBorder(vFeaPointsCurrent[i]))
                status[i] = 0;
        }
        
        reduceVector(vpLastMaps, status);
        // 利用RANSAC剔除错误匹配
        reduceVector(vFeaPointsLast, status);
        reduceVector(vFeaPointsCurrent, status);
        std::vector<uchar> RansacStatus;
        cv::Mat Fundamental = cv::findFundamentalMat(vFeaPointsLast, vFeaPointsCurrent, RansacStatus, cv::FM_RANSAC);
        
        reduceVector(vFeaPointsLast, RansacStatus);
        reduceVector(vFeaPointsCurrent, RansacStatus);
        reduceVector(vpLastMaps, RansacStatus);

        mCurrentFrame->mvpFeatureLeft.clear();
        for(int i = 0; i < vFeaPointsCurrent.size(); i++){
            if(vpLastMaps[i].expired())
                continue;
            auto feature = std::make_shared<Feature>(cv::KeyPoint(vFeaPointsCurrent[i],7));
            feature->mpMapPoint = vpLastMaps[i];
            mCurrentFrame->mvpFeatureLeft.push_back(feature);
        }

// 可视化
#if 0
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat img_match; std::vector<cv::DMatch> matches;
        for(int i = 0; i < vFeaPointsLast.size(); i++){
            keypoints_1.push_back(cv::KeyPoint(vFeaPointsLast[i], 7));
            keypoints_2.push_back(cv::KeyPoint(vFeaPointsCurrent[i], 7));
            matches.push_back(cv::DMatch(i, i, 0));
        }
        cv::drawMatches(mCurrentFrame->mImgLeft, keypoints_1, mCurrentFrame->mImgRight, keypoints_2, matches, img_match);
        std::string idLast = IntToString(mLastFrame->mFrameId);
        std::string idCur = IntToString(mCurrentFrame->mFrameId);
        std::string text = idLast + "->" + idCur; 
        cv::imshow(text, img_match);
        cv::waitKey(0);
        cv::destroyWindow(text);
#endif
        // LOG(INFO) << "追踪到上一帧一共 "<<mCurrentFrame->mvpFeatureLeft.size()<<" 个地图点";
        return mCurrentFrame->mvpFeatureLeft.size()>=20;
    }

    
    /**
     * ＠brief 三角化函数
     * ＠param[in] 相机1的投影矩阵
     * ＠param[in] 相机2的投影矩阵
     * ＠param[in] 图像1的特征点
     * ＠param[in] 图像2的特征点
     * ＠param[out] 三角化后地图点的坐标
     * */
    void Frontend::Triangulation(const Mat34d &P1, const Mat34d &P2,
                                 const cv::KeyPoint &Kp1, const cv::KeyPoint &Kp2, 
                                 Vector3d &X3D) {
        cv::Mat A(4, 4, CV_32F);
        cv::Mat p1, p2;
        cv::eigen2cv(P1, p1);
        cv::eigen2cv(P2, p2);
        static_cast<cv::Mat>(Kp1.pt.x * p1.row(2) - p1.row(0)).copyTo(A.row(0));
        static_cast<cv::Mat>(Kp1.pt.y * p1.row(2) - p1.row(1)).copyTo(A.row(1));
        static_cast<cv::Mat>(Kp2.pt.x * p2.row(2) - p2.row(0)).copyTo(A.row(2));
        static_cast<cv::Mat>(Kp2.pt.y * p2.row(2) - p2.row(1)).copyTo(A.row(3));
        //奇异值分解的结果
        cv::Mat u,w,vt;
        //对系数矩阵A进行奇异值分解
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        //根据前面的结论，奇异值分解右矩阵的最后一行其实就是解，原理类似于前面的求最小二乘解，四个未知数四个方程正好正定
        //别忘了我们更习惯用列向量来表示一个点的空间坐标
        cv::Mat x3D = vt.row(3).t();
        //为了符合其次坐标的形式，使最后一维为1
        x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
        cv::cv2eigen(x3D, X3D);
    }

    // 检查是否插入关键帧
    bool Frontend::CheckIsInserKF(){
        int nTrackedMaps = 0;
        for(auto fea : mCurrentFrame->mvpFeatureLeft){
            if(!fea->mpMapPoint.expired()){
                nTrackedMaps++;
            }
        }
        if(nTrackedMaps < mnFeaturesTrackingGood)
            return true;
        return false;
    }

    //　插入关键帧
    bool Frontend::InsertKeyFrame(){
        // Vec6d se3_zero;
        // se3_zero.setZero();
        KeyFrame::Ptr newKeyFrame = std::make_shared<KeyFrame>(mCurrentFrame);
        // LOG(INFO) << "设第" << newKeyFrame->mFrameId << "帧" << "为关键帧id: " << newKeyFrame->mKeyFrameId;
        // 为关键帧的地图点添加观测
        for(auto fea : newKeyFrame->mvpFeatureLeft){
            auto mp = fea->mpMapPoint.lock();
            if(mp){
                mp->AddObservation(newKeyFrame->mKeyFrameId, fea);
            }
        }
        if(mStatus == INIT){
            newKeyFrame->mLastKF = nullptr;
            newKeyFrame->mRelativePoseToLastKF = SE3_Identity;
        }else{
            newKeyFrame->mLastKF = mReferenceKF;
            newKeyFrame->mRelativePoseToLastKF = newKeyFrame->GetPose() * (newKeyFrame->mLastKF)->GetPose().inverse();
        }

        mReferenceKF = newKeyFrame;
        if(mBackend){
            mBackend->InsertKeyFrame(newKeyFrame);// 插入后端
        }
        // mCurrentFrame->SetPose(SE3::exp(se3_zero));
        return true;
    }
}