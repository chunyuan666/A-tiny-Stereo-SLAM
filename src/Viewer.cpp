#include "Common.h"
#include "Viewer.h"

namespace myslam{
    

    void Viewer::LoopRunning(){
        pangolin::CreateWindowAndBind("MYSLAM", 1024,768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);

        pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

        pangolin::View& vis_display = pangolin::CreateDisplay()
                    .SetBounds(0.0,1.0,pangolin::Attach::Pix(175),1.0,-1024.0f/768.0f)
                   .SetHandler(new pangolin::Handler3D(vis_camera));

        bool bFollow = true;
        LOG(INFO) << "ViewerThread Begin...";
        while(!pangolin::ShouldQuit() && mViewerIsRunning){
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            std::unique_lock<std::mutex> lock(mViewerDataMutex);
            if(mCurrentFrame){
                if (menuFollowCamera && bFollow){
                    FollowCurrentFrame(vis_camera);
                }else if(!menuFollowCamera && bFollow){
                    bFollow = false;
                }else if(menuFollowCamera && !bFollow){
                    FollowCurrentFrame(vis_camera);
                    vis_camera.SetModelViewMatrix(
                        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                    bFollow = true;
                }
                cv::Mat imgL, imgC;
                // if(mLastFrame){
                //     imgL = PlotFrameImage(mLastFrame);
                //     cv::imshow("mLastFrame", imgL);
                // }  
    
                imgC = PlotFrameImage(mCurrentFrame);
                cv::resize(imgC, imgC, cv::Size(), 0.5, 0.5);
                cv::imshow("mCurrentFrame", imgC);
                cv::waitKey(1);
            }

            vis_display.Activate(vis_camera);
            if(mCurrentFrame){
                DrawFrame(mCurrentFrame, green);
            }
            if (mMap){
                DrawKFsAndMPs(menuShowKeyFrames, menuShowPoints);
            }

            if(mMap){
                DrawLine();
            }
            
            pangolin::FinishFrame();
            usleep(1000);
        }
        std::cout << "ViewerThread Stop!" << std::endl;
    }

    void Viewer::DrawLine(){
        auto allKFs = mMap->GetAllKeyFrames();
        if(allKFs.size() < 2){
            return;
        }
        std::vector< Frame::Ptr > AllKeyFrames;
        for(auto &it: allKFs){
            AllKeyFrames.push_back(it.second);
        }
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(255, 0, 0);
        for(int i=0; i<AllKeyFrames.size()-1; i++){
            auto kf = AllKeyFrames.at(i);
            auto nextkf = AllKeyFrames.at(i+1);
            auto t1 = kf->GetPose().inverse().translation();
            auto t2 = nextkf->GetPose().inverse().translation();
            glVertex3f(static_cast<GLfloat>(t1.x()), static_cast<GLfloat>(t1.y()), static_cast<GLfloat>(t1.z()));
            glVertex3f(static_cast<GLfloat>(t2.x()), static_cast<GLfloat>(t2.y()), static_cast<GLfloat>(t2.z()));
            //LOG(INFO) << "kf_id:" << kf->id_;
        }
        glEnd();
    }

    void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera){
        SE3 Twc = mCurrentFrame->GetPose().inverse();
        pangolin::OpenGlMatrix m(Twc.matrix());
        vis_camera.Follow(m, true);

    }

    cv::Mat Viewer::PlotFrameImage(Frame::Ptr Frame){
        cv::Mat img_out;
        cv::cvtColor(Frame->mImgLeft, img_out, cv::COLOR_GRAY2BGR);
        for (size_t i = 0, N = Frame->mvpFeatureLeft.size(); i < N; ++i){
                auto feat = Frame->mvpFeatureLeft[i];
                cv::circle(img_out, feat->mKeyPoint.pt, 2, cv::Scalar(0,255,0), 2);
        }
        std::string id = IntToString(Frame->mFrameId);
        putText(img_out, id, cv::Point(0, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
        return img_out;
    }

    void Viewer::DrawFrame(Frame::Ptr frame, const float* color){
        SE3 Twc = frame->GetPose().inverse();
        const float sz = 1.0;
        const int line_width = 2.0;
        const float fx = 400;
        const float fy = 400;
        const float cx = 512;
        const float cy = 384;
        const float width = 1080;
        const float height = 768;

        glPushMatrix();

        Sophus::Matrix4f m = Twc.matrix().template cast<float>();
        glMultMatrixf((GLfloat*)m.data());

        if (color == nullptr) {
            glColor3f(1, 0, 0);
        } else
            glColor3f(color[0], color[1], color[2]);

        glLineWidth(line_width);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();
        glPopMatrix();
    }

    void Viewer::DrawKFsAndMPs(const bool menuShowKeyFrames, const bool menuShowPoints){
        if(mMap->GetAllKeyFrames().empty() || mMap->GetActivateMapPoints().empty())
            return;
        if (menuShowKeyFrames){
            for (auto& kf: mMap->GetAllKeyFrames()){
                DrawFrame(kf.second, blue);
            }
        }

        if(menuShowPoints){
            glPointSize(1.5);
            glBegin(GL_POINTS);
            for (auto mp : mMap->GetAllMapPoints()) {
                if(mp.second->GetActivateObsCnt()>0)
                {
                    auto pos = mp.second->GetPose();
                    glColor3f(255, 0, 0);
                    glVertex3d(pos[0], pos[1], pos[2]);
                }else{
                    auto pos = mp.second->GetPose();
                    glColor3f(0, 0, 0);
                    glVertex3d(pos[0], pos[1], pos[2]);
                }
            }
            // for (auto& mp : mMap->GetActivateMapPoints()) {
            //     auto pos = mp.second->GetPose();
            //     glColor3f(255, 0, 0);
            //     glVertex3d(pos[0], pos[1], pos[2]);
            // }
        }
        glEnd();
    }
}