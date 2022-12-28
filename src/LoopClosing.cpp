#include <LoopClosing.h>
#include "Backend.h"
#include "Optimizer.h"

namespace myslam{
    LoopClosing::LoopClosing(int nLevels, float simThe1, float simThe2):
    nLevels(nLevels), similarThre1(simThe1), similarThre2(simThe2){
        mbLoopRunning.store(true);
        mDeepLCD = std::make_shared<DeepLCD>();
        mthreadLoopClosing = std::thread(std::bind(&LoopClosing::Running, this));
    }

    bool LoopClosing::checkNewKF(){
        std::unique_lock<std::mutex> lck(mmutexNewKF);
        return !mlpNewKF.empty();
    }

    bool LoopClosing::ProcessKF(){
        // 弹出最新的关键帧
        std::unique_lock<std::mutex> lck(mmutexNewKF);
        {
            if(!mlpNewKF.empty()){
                mCurKF = mlpNewKF.front();
                mlpNewKF.pop_front();
            }
        }
        // 计算当前帧的特征向量
        mCurKF->mDescrVector = mDeepLCD->calcDescrOriginalImg(mCurKF->mImgLeft);
        // 计算特征点金字塔和描述子
        ProcessByPyramid();
        return true;
    }

    void LoopClosing::ProcessByPyramid(){
        // 1、将特征点按照金字塔扩充
        std::vector<cv::KeyPoint> vPyramidKps;
        vPyramidKps.reserve(mCurKF->mvpFeatureLeft.size() * nLevels );
        for(int i = 0; i < mCurKF->mvpFeatureLeft.size(); i++){
            auto kp = mCurKF->mvpFeatureLeft[i]->mKeyPoint;
            kp.class_id = i;
            for(int level = 0; level < nLevels; level++){
                cv::KeyPoint newkp(kp);
                newkp.octave = level;
                newkp.class_id = i;
                newkp.response = -1;
                vPyramidKps.push_back(newkp);
            }
        }

        // 2、剔除超边缘的特征点和非FAST角点，计算特征点的方向
        mpORBextractor->CalcuKeyPoints(mCurKF->mImgLeft, vPyramidKps, mCurKF->mvPyramidKeyPoints);

        // 3、计算描述子
        mpORBextractor->CalcuDescriptors(mCurKF->mImgLeft, mCurKF->mvPyramidKeyPoints, mCurKF->mDescriptors);
        
    }

    bool LoopClosing::FindLoopcloseKF(){
        // 1、从数据库中寻找相似关键帧
        if(KFDB.empty())    return false;
        
        float max_score = 0; unsigned long max_id;
        int cntsimilar = 0;
        for(auto KF_data : KFDB){
            auto kf = KF_data.second;
            if(mCurKF->mKeyFrameId - kf->mKeyFrameId < 20)
                break;
            float score = mDeepLCD->score(mCurKF->mDescrVector, kf->mDescrVector);
            // LOG(INFO) << "score: " << score;
            if(score > max_score){
                max_score = score;
                max_id = KF_data.first;
            }
            if(score >similarThre2){
                cntsimilar++;
            }
        }
        // LOG(INFO) << "max_score: " << max_score << " cntsimilar: " << cntsimilar << " DB_size: " << KFDB.size();
        if(max_score < similarThre1 || cntsimilar > 3){
            return false;
        }
        KeyFrame::Ptr candicateKF = KFDB[max_id];

        // 2、将闭环候选帧帧投影到当前帧，匹配描述子
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        std::vector<cv::DMatch> matches;
        matcher->match(candicateKF->mDescriptors, mCurKF->mDescriptors, matches);
        // 筛选匹配良好的点对
        auto min_max = std::minmax_element(matches.begin(), matches.end(), 
                            [](const cv::DMatch &m1, const cv::DMatch &m2){
                                return m1.distance < m2.distance;
                            });
        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;
        //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < matches.size(); i++) {
            if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
                good_matches.push_back(matches[i]);
            }
        }
        mValidFeaMatches.clear();
        for(auto match : good_matches){
            int candiKFFeaID = candicateKF->mvPyramidKeyPoints[match.queryIdx].class_id;
            int curKFFeaID = mCurKF->mvPyramidKeyPoints[match.trainIdx].class_id;
            if(mValidFeaMatches.find(candiKFFeaID) == mValidFeaMatches.end())
                mValidFeaMatches.insert({candiKFFeaID, curKFFeaID});
        }
        if(mValidFeaMatches.size() < 10) return false;
        mLoopcloseKF = candicateKF;
        // LOG(INFO) << "找到了回环帧。Id: " << mLoopcloseKF->mKeyFrameId << " Id: "
        // << mCurKF->mKeyFrameId << "->" << mLoopcloseKF->mKeyFrameId;
        return true;
    }

    bool LoopClosing::CaluCorrectedPose(){
        if(mLoopcloseKF == nullptr) return false;
        // 1、使用p3p计算当前帧的正确位姿
        std::vector<cv::Point3f> pts_3d;
        std::vector<cv::Point2f> curkf_pts_2d, loopkf_pts_2d;
        std::vector<cv::DMatch> vMatches;
       
        for(auto iter = mValidFeaMatches.begin(); iter != mValidFeaMatches.end();){
            int loopFeaId = iter->first;
            int curFeaId = iter->second;
            auto loopKF_Fea = mLoopcloseKF->mvpFeatureLeft[loopFeaId];
            auto curKF_Fea = mCurKF->mvpFeatureLeft[curFeaId];
            auto mp = loopKF_Fea->mpMapPoint.lock();
            if(mp){
                auto pos = mp->GetPose();
                pts_3d.push_back(cv::Point3f(pos[0], pos[1], pos[2]));
                curkf_pts_2d.push_back(curKF_Fea->mKeyPoint.pt);
                loopkf_pts_2d.push_back(loopKF_Fea->mKeyPoint.pt);
                cv::DMatch match(loopFeaId, curFeaId, 10);
                vMatches.push_back(match);
                iter++;
            }else{
                iter = mValidFeaMatches.erase(iter);
            }
        }
       

        if(pts_3d.size() < 10){
            // LOG(INFO) << "3D点个数为: " << pts_3d.size() << " 个数不足，不计算位姿";
            return false;
        } 

        LOG(INFO) << "匹配特征点大小：" << mValidFeaMatches.size();
        cv::Mat r, t, R, K;
        Mat33d R_eigen;
        Vec3d t_eigen;
        cv::eigen2cv(mCurKF->mK, K);
        if(!cv::solvePnPRansac(pts_3d, curkf_pts_2d, K, cv::Mat(), r, t, false, 100)){
            LOG(INFO) << "solverPnPRansac失败!";
            return false;
        }
        cv::Rodrigues(r, R);
        cv::cv2eigen(R, R_eigen);
        cv::cv2eigen(t, t_eigen);
        // LOG(INFO) << "R: \n" << R_eigen << "\n t: " << t_eigen.transpose();
        mCorrectedCurPose = SE3(R_eigen, t_eigen);
        // LOG(INFO) << "Tracking的位姿: \n" << mCurKF->GetPose().matrix3x4(); 
        // LOG(INFO) << "优化前PNP求解的mCorrectedCurPose:\n" << mCorrectedCurPose.matrix3x4();
        // 2、优化位姿
        int cntInliers = OptmizeCurKFPose();
            LOG(INFO) << "LoopClosing: number of match inliers (after optimization): " <<  cntInliers;

        if( cntInliers < 5){
            LOG(INFO) << "内点数小于5,抛弃。";
            return false;
        }
        mCurKF->mLoopKF = mLoopcloseKF;
        mCurKF->mRelativePoseToLoopKF = mCorrectedCurPose * mLoopcloseKF->GetPose().inverse(); 
        mLastCloseKF = mCurKF;
#if 1
    cv::Mat draw_img;
    cv::drawMatches(mLoopcloseKF->mImgLeft, mLoopcloseKF->GetKeyPoints(), mCurKF->mImgLeft, mCurKF->GetKeyPoints(),
                vMatches, draw_img);
    cv::resize(draw_img, draw_img, cv::Size(), 0.5, 0.5);
    cv::imshow("vaild match", draw_img);
    cv::waitKey(1);
#endif
        return true;
    }

    int LoopClosing::OptmizeCurKFPose(){
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        //vertex
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(0);
        vertex_pose->setEstimate(mCorrectedCurPose);
        optimizer.addVertex(vertex_pose);
        // edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        edges.reserve(mValidFeaMatches.size());
        std::vector<bool> vEdgeIsOutlier;
        vEdgeIsOutlier.reserve(mValidFeaMatches.size());
        std::vector<std::map<int, int>::iterator > vMatches;
        vMatches.reserve(mValidFeaMatches.size());
        
        for(auto iter = mValidFeaMatches.begin(); iter != mValidFeaMatches.end(); iter++){
            int loopFeatureId = (*iter).first;
            int currentFeatureId = (*iter).second;
            auto mp = mLoopcloseKF->mvpFeatureLeft[loopFeatureId]->mpMapPoint.lock();
            auto point2d = mCurKF->mvpFeatureLeft[currentFeatureId]->mKeyPoint.pt;

            
            if(mp == nullptr){
                LOG(INFO) << "mp为空!!!!\n";
                continue;
                assert(mp != nullptr);
            }
            EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->GetPose(), mCurKF->mK);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(cvPoint2Vec2(point2d));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            vMatches.push_back(iter);
            optimizer.addEdge(edge);

            index++;
        }

        // estimate the Pose and determine the outliers
        // start optimization
        const double chi2_th = 5.991;
        int cntOutliers = 0;
        int numIterations = 4;

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        // use the same strategy as in frontend
        for(int iteration = 0; iteration < numIterations; iteration++){
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cntOutliers = 0;

            // count the outliers
            for(size_t i = 0, N = edges.size(); i < N; i++){
                auto e = edges[i];
                if(vEdgeIsOutlier[i]){
                    e->computeError();
                }
                if(e->chi2() > chi2_th){
                    vEdgeIsOutlier[i] = true;
                    e->setLevel(1);
                    cntOutliers++;
                } else{
                    vEdgeIsOutlier[i] = false;
                    e->setLevel(0);
                }

                if(iteration == numIterations - 2){
                    e->setRobustKernel(nullptr);
                }
            }
        }

        // remove the outlier match
        for(size_t i = 0, N = vEdgeIsOutlier.size(); i < N; i++){
            if(vEdgeIsOutlier[i]){
                mValidFeaMatches.erase(vMatches[i]);
            }
        }
    
        mCorrectedCurPose = vertex_pose->estimate();
        // LOG(INFO) << "优化后mCorrectedCurPose: \n" << mCorrectedCurPose.matrix3x4();

        return mValidFeaMatches.size();
    }

    void LoopClosing::CorrectPose(){
        // 1、正确的位姿与原始位姿相差不大，不用重新融合位姿
        double error = (mCurKF->GetPose() * mCorrectedCurPose.inverse()).log().norm();
        LOG(INFO) << "error:" << error;
        if (error <= 1)
            return;

        auto backend = mBackend.lock();
        backend->RequestPause();
        while(!backend->HasPaused()){
            LOG(INFO) << "等待后端优化暂停...";
            usleep(1000);
        }

        // 2、计算activateKF的正确位姿
        std::map<unsigned long, SE3> correctedActiKFPoses;
        std::map<unsigned long, KeyFrame::Ptr> actiKFs = mMap->GetActivateKeyFrames();
        for(auto m : actiKFs){
            unsigned long actKFId = m.first;
            KeyFrame::Ptr actiKF = m.second;
            if(actiKF->mKeyFrameId == mCurKF->mKeyFrameId)
                continue;
            SE3 T_aw = actiKF->GetPose();
            SE3 T_cw = mCurKF->GetPose();
            SE3 T_ac = T_aw * T_cw.inverse();
            SE3 T_acorrected = T_ac * mCorrectedCurPose;
            correctedActiKFPoses.insert({actKFId, T_acorrected});
        }
        correctedActiKFPoses.insert({mCurKF->mKeyFrameId, mCorrectedCurPose});

        // 3、修正滑动窗口中地图点坐标
        std::map<unsigned long, MapPoints::Ptr> actiMaps = mMap->GetActivateMapPoints();
        for(auto &m : actiMaps){
            MapPoints::Ptr actimap = m.second;
            // 取出该地图点的第一次观测到它的关键帧
            auto obs = actimap->GetActivateObservation();
            if(obs.empty())
                continue;
            auto firstObsKfId = obs.begin()->first;
            assert(actiKFs.find(firstObsKfId) != actiKFs.end());
            auto firstKF = actiKFs.at(firstObsKfId);
            auto Pc = firstKF->GetPose() * actimap->GetPose();
            assert(correctedActiKFPoses.find(firstObsKfId) != correctedActiKFPoses.end());
            SE3 T_acorrected = correctedActiKFPoses.at(firstObsKfId);
            auto P_corrected = T_acorrected.inverse() * Pc;
            actimap->SetPose(P_corrected);
        }
        // 4、修正滑动窗口中的关键帧位姿
        for(auto &m : actiKFs){
            auto kfid = m.first;
            auto kf = m.second;
            kf->SetPose(correctedActiKFPoses.at(kfid));
        }

        // 5、地图点融合，将LoopKF的地图点替代CurKF的地图点
        for(auto iter = mValidFeaMatches.begin(); iter != mValidFeaMatches.end(); iter++){
            auto loopfeaId = iter->first;
            auto curfeaId = iter->second;
            auto loopMap = mLoopcloseKF->mvpFeatureLeft.at(loopfeaId)->mpMapPoint.lock();
            if(loopMap == nullptr){
                LOG(INFO) << "出现错误!! loopMap为空。";
                assert(loopMap != nullptr);
            }
            auto curMap = mCurKF->mvpFeatureLeft.at(curfeaId)->mpMapPoint.lock();
            // 如果curMap不为空
            if(curMap != nullptr){
                for(auto &obs : curMap->GetObservation()){
                    // 先将curMap观测到的关键帧和特征点， 使loopMap也要观测到
                    auto kfId = obs.first;
                    auto fea = obs.second;
                    loopMap->AddObservation(kfId, fea);
                    // 再将LoopKF的地图点替代CurKF的地图点
                    fea.lock()->mpMapPoint = loopMap;
                    // 最后将curMap从地图点中剔除
                    mMap->RemoveMapPoints(curMap);
                }
            }else{  // curMap为空
                // 将loopMap赋给curKF对应的特征点
                mCurKF->mvpFeatureLeft.at(curfeaId)->mpMapPoint = loopMap;
            }
        }

        // 6、优化位姿
        LOG(INFO) << "开启GlobalBA...";
        Optimizer::GlobalBundleAdjustment(mMap, mLoopcloseKF, mCurKF);
        LOG(INFO) << "结束GlobalBA...";

        backend->Resume();
    }


    void LoopClosing::Running(){
        while(mbLoopRunning.load()){
            if(checkNewKF()){
                ProcessKF();
                bool bcomputeLoopKF = false;
                // LOG(INFO) << "KFDB size: " << KFDB.size();
                if(KFDB.size() > 0){
                    if(FindLoopcloseKF()){
                        bcomputeLoopKF = CaluCorrectedPose();
                        if(bcomputeLoopKF){
                            // 矫正位姿
                           CorrectPose();
                        }
                    }
                }
                
                if(!bcomputeLoopKF){
                    KFDB.insert({mCurKF->mKeyFrameId, mCurKF});
                    mLastKF = mCurKF;
                }
            }
            usleep(1000); 
        }
        std::cout << "LoopClosingThread Stop!" << std::endl;
    }





}