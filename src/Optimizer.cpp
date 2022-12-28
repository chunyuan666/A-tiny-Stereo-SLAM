#include "Optimizer.h"
#include "KeyFrame.h"

namespace myslam{
    unsigned long Optimizer::PoseOptimization(Frame::Ptr &Frame) {
        // 用G2O来估计位姿
        // Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        //　2.添加顶点
        VertexPose *vertex = new VertexPose();
        vertex->setId(0);
        vertex->setEstimate(Frame->GetPose());
        vertex->setFixed(false);
        optimizer.addVertex(vertex);

        // std::cout << "K:\n" << Frame->mK << std::endl;

        //　3.添加边
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features;
        //edges.reserve(Frame->mvpFeatureLeft.size());
        //features.reserve(Frame->mvpFeatureLeft.size());
        for(auto &fea : Frame->mvpFeatureLeft){
            // if(fea->mpMapPoint.expired())   continue;
            assert(!fea->mpMapPoint.expired());
            auto map = fea->mpMapPoint.lock();
            if(map->mbIsOutlier==false){
                auto *edge = new EdgeProjectionPoseOnly(map->GetPose(), Frame->mK);
                edge->setId(index);
                edge->setVertex(0, vertex);
                edge->setMeasurement(cvPoint2Vec2(fea->mKeyPoint.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                rk->setDelta(sqrt(5.991));
                edge->setRobustKernel(rk);
                features.push_back(fea);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }
        //　4.迭代
        double chi2_th = 5.991; //95%的置信区间
        int nOutLiers = 0;
        int iterations = 4;
        for(int iter = 0; iter < iterations; iter++){
            vertex->setEstimate(vertex->estimate());
            // optimizer.setVerbose ( true );
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            nOutLiers = 0;
            for(int i = 0; i < edges.size(); i++) {
                auto e = edges[i];
                auto fea = features[i];
                double chi2 = e->chi2();
                // 如果这条误差边是来自于outlier
                if (fea->IsOutLier) {
                    e->computeError();
                }
                // 该点不可靠
                if (chi2 > chi2_th) {
                    fea->IsOutLier = true;
                    e->setLevel(1); //下次不优化
                    nOutLiers++;
                } else {
                    fea->IsOutLier = false;
                    e->setLevel(0);
                }
                if(iter==iterations/2)
                    e->setRobustKernel(nullptr);
            }
        }
        Frame->SetPose(vertex->estimate());
        //LOG(INFO) << "\n" << "第" << Frame->mFrameId << "帧" << "估计后位姿：\n" << vertex->estimate().matrix();
        return edges.size()-nOutLiers;
    }

    void Optimizer::OptimizeActivateMap(Map::Ptr &Map, const Camera::Ptr &camera) {
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(
        g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        auto KFs = Map->GetActivateKeyFrames();
        auto MPs = Map->GetActivateMapPoints();

        std::map<unsigned long, VertexPose *> Vertex_KFs;
        std::map<unsigned long, VertexXYZ *> Vertex_Mps;

        // 增加顶点，相机位姿
        unsigned long maxKFid = 0;
        for(auto &Keyframe : KFs){
            auto kf = Keyframe.second;
            auto vertex_pose = new VertexPose();
            vertex_pose->setId(static_cast<int>(kf->mKeyFrameId));
            //LOG(INFO) << "\nkf_pose: \n" << kf->GetPose().matrix();
            vertex_pose->setEstimate(kf->GetPose());
            if(KFs.size()>1 && kf->mKeyFrameId == 0){
                vertex_pose->setFixed(true);
                assert(kf->GetPose().log().norm()!=0);
            }    
            optimizer.addVertex(vertex_pose);
            maxKFid = std::max(kf->mKeyFrameId, maxKFid);
            Vertex_KFs.insert(std::make_pair(kf->mKeyFrameId, vertex_pose));
        }

        int index = 1;
        double chi2_th = 5.991;
        std::map<Feature::Ptr, EdgeProjection *> FeatsAndEdges;
        // 增加顶点，地图点的坐标
        for(auto &MapPoint : MPs){
            auto mp_id = MapPoint.first;
            auto mp = MapPoint.second;
            if(mp==nullptr || mp->mbIsOutlier)
                continue;
            if(Vertex_Mps.find(mp->mid) == Vertex_Mps.end()){
                auto vertex_XYZ = new VertexXYZ();
                vertex_XYZ->setId(static_cast<int>(maxKFid +1 + mp->mid));
                //LOG(INFO) << "\nmp_pose: \n" << mp->GetPose().matrix();
                vertex_XYZ->setEstimate(mp->GetPose());
                vertex_XYZ->setMarginalized(true);
                optimizer.addVertex(vertex_XYZ);
                Vertex_Mps.insert(std::make_pair(mp->mid, vertex_XYZ));
            }
            auto observations = mp->GetActivateObservation();
            for(auto &obs : observations){
                auto kfId = obs.first;
                auto feat = obs.second.lock();
                if(feat==nullptr)
                    continue;
                assert(KFs.find(kfId) != KFs.end());
                auto *e = new EdgeProjection(camera->GetK(), camera->GetCameraPose());
                e->setId(index);
                e->setVertex(0, Vertex_KFs[kfId]);
                e->setVertex(1, Vertex_Mps[mp_id]);
                e->setMeasurement(cvPoint2Vec2(feat->mKeyPoint.pt));
                e->setInformation(Mat22d::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(sqrt(chi2_th));
                e->setRobustKernel(rk);
                optimizer.addEdge(e);
                index++;
                FeatsAndEdges.insert(std::make_pair(feat, e));
            }
        }
        int cntOutlier = 0, cntInlier = 0;

        optimizer.initializeOptimization(0);
        optimizer.optimize(5);

        for(auto &fe : FeatsAndEdges){
            auto e = fe.second;
            double chi2 = e->chi2();
            // LOG(INFO) << "chi2: " << chi2;
            if(e->chi2() > chi2_th){
                e->setLevel(1);
                // cntOutlier ++;
            }else{
                // cntInlier++;
                e->setLevel(0);
            }
             e->setRobustKernel(0);
        }

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        for(auto &fe : FeatsAndEdges){
            auto feat = fe.first;
            auto e = fe.second;
            double chi2 = e->chi2();
            if(e->chi2() > chi2_th){
                cntOutlier ++;
                feat->IsOutLier = true;
            }else{
                cntInlier++;
                feat->IsOutLier = false;
            }
        }
                
        LOG(INFO) << "OUTLIERS nums is:  " << cntOutlier;
        LOG(INFO) << "INLIERS nums is:  " << cntInlier;
        // 处理外点
        // 遍历当前边和特征点
        for(auto &fe : FeatsAndEdges){
            // 找出外点的特征
            auto feat = fe.first;
            auto mp = feat->mpMapPoint.lock();
            if(feat->IsOutLier){
                // 取消Feature对该点的观测
                //mp->RemoveActiveObservation(feat);
                mp->RemoveActiveObservation(feat);
                mp->RemoveObservation(feat);
                if(mp->GetActivateObsCnt()==0){
                    // 释放该地图点的指针,feat不再持有该地图点的指针
                    feat->mpMapPoint.reset();
                    // 设置为外点
                    mp->mbIsOutlier = true;
                }
                feat->IsOutLier = false;
            }
        }
        
        //设置当前帧的位姿
        for(auto &v : Vertex_KFs){
            KFs[v.first]->SetPose(v.second->estimate());
            // LOG(INFO) << "矫正后的位姿： id: " << KFs[v.first]->mFrameId << ", KFID:"
            // << KFs[v.first]->mKeyFrameId << "\n" << KFs[v.first]->GetPose().matrix3x4(); 
        }
        for(auto &m : Vertex_Mps){
        //     assert(MPs.find(m.first) != MPs.end());
        //     assert(MPs[m.first]);
            MPs[m.first]->SetPose(m.second->estimate());
        }
        //在地图中删除观测为0的点和外点
        Map->CullOldActivateMapPoint();
        Map->RemoveOutlierMapPoints();
    }

    void Optimizer::GlobalBundleAdjustment(Map::Ptr &Map, KeyFrame::Ptr &mLoopKF, KeyFrame::Ptr &CurKF){
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
        typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer; 
        optimizer.setAlgorithm(solver); 

        auto allKFs = Map->GetAllKeyFrames();
        auto activateKFs = Map->GetActivateKeyFrames();

        // 1、添加关键帧位姿顶点
        std::map<unsigned long, VertexPose*> vertex_kf;
        for(auto &m : allKFs){
            auto kfId = m.first;
            auto kf = m.second;
            VertexPose *v_kfpose = new VertexPose();
            v_kfpose->setId(kfId);
            v_kfpose->setEstimate(kf->GetPose());
            v_kfpose->setMarginalized(false);

            // 将第一帧(Id:0)、LoopKF、已经矫正过的活跃帧(activateKF)固定
            if(kfId == 0 || kfId == mLoopKF->mKeyFrameId || 
                activateKFs.find(kfId) != activateKFs.end()){
                v_kfpose->setFixed(true);
            }

            optimizer.addVertex(v_kfpose);
            vertex_kf.insert({kfId, v_kfpose});
        }

        // 2、添加边
        int index = 0;
        std::map<int, EdgePoseGraph*> Edges;
        for(auto &m : allKFs){
            auto kfId = m.first;
            auto kf = m.second;
            assert(vertex_kf.find(kfId) != vertex_kf.end());

            // 2.1、相邻帧约束
            auto lastKF = kf->mLastKF;
            if(lastKF){
                EdgePoseGraph *e = new EdgePoseGraph();
                e->setId(index);
                e->setVertex(0, vertex_kf.at(kfId));
                e->setVertex(1, vertex_kf.at(lastKF->mKeyFrameId));
                e->setMeasurement(kf->mRelativePoseToLastKF);
                e->setInformation(Mat66d::Identity());
                optimizer.addEdge(e);
                Edges.insert({index, e});
                index++;
            }

            // 2.2、与LoopKF的约束
            auto loopKF = kf->mLoopKF;
            if(loopKF){
                EdgePoseGraph *e = new EdgePoseGraph();
                e->setId(index);
                e->setVertex(0, vertex_kf.at(kfId));
                e->setVertex(1, vertex_kf.at(loopKF->mKeyFrameId));
                e->setMeasurement(kf->mRelativePoseToLoopKF);
                e->setInformation(Mat66d::Identity());
                optimizer.addEdge(e);
                Edges.insert({index, e});
                index++;
            }
        }

        // 3、优化
        optimizer.initializeOptimization();
        optimizer.optimize(20);


        // 4、矫正关键帧位姿和地图点位姿
        auto allMaps = Map->GetAllMapPoints();
        auto activateMaps = Map->GetActivateMapPoints();

        for(auto iter = allMaps.begin(); iter != allMaps.end();){
            if(activateMaps.find(iter->first) != activateMaps.end()){
                iter = allMaps.erase(iter);
            }else{
                iter++;
            }
        }
        for(auto &m : allMaps){
            auto mp = m.second;
            // 获取第一次观测到它的关键帧
            assert(!mp->GetObservation().empty());
            auto obs = mp->GetObservation();
            auto firstobsKFId = obs.begin()->first;
            if(allKFs.find(firstobsKFId) == allKFs.end()){
                continue;
            }
            auto firstobsKF = allKFs.at(firstobsKFId);
            
            assert(vertex_kf.find(firstobsKFId) != vertex_kf.end());
            SE3 T_corrected = vertex_kf.at(firstobsKFId)->estimate();
            auto Pcam = firstobsKF->GetPose() * mp->GetPose();
            auto Pcorrected = T_corrected.inverse() * Pcam;
            mp->SetPose(Pcorrected);
        }

        for(auto &v : vertex_kf){
            allKFs.at(v.first)->SetPose(v.second->estimate());
        }
    }

    
}









