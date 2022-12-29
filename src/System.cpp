#include "System.h"
#include <fstream>
#include <memory>

using namespace std;

namespace myslam{

    
/**
 * @brief 构造函数
 * @details 加载相机参数、其他参数，参数保存在yaml文件中,
 *          创造前端、后端、回环检测、地图、可视化组件
 * @param 
 */
System::System(const std::string config_file){

    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    fs_read = fs;
    // fs_read.open(config_file, cv::FileStorage::WRITE);
    
    GetCameraPara();

    GetORBExtractorPara();

    mFrontend = std::make_shared<Frontend>();
    mFrontend->SetORBExtractor(mORBExtractorInit, mORBExtractor);
    mFrontend->SetCamera(mCameraLeft, mCameraRight);

    int TrackingGood, TrackingBad, InitGood;
    float thDepth;
    fs_read["numFeatures.initGood"] >> InitGood;
    fs_read["numFeatures.trackingBad"] >> TrackingBad;
    fs_read["numFeatures.trackingGood"] >> TrackingGood;
    fs_read["ThDepth"] >> thDepth;
    mFrontend->SetTrackingPara(InitGood, TrackingGood, TrackingBad, thDepth);

    mMap = std::make_shared<Map>();
    int num_activateMap;
    fs_read["Map.activeMap.size"] >> num_activateMap;
    mMap->SetNumActivateMap(num_activateMap);
    
    mFrontend->SetMap(mMap);

    mBackend = std::make_shared<Backend>();
    mBackend->SetMap(mMap);
    mBackend->SetCamera(mCameraLeft);
    
    mFrontend->SetBackend(mBackend);

    bool bshow;
    fs_read["Viewer.bShow"] >> bshow;
    if(bshow){
         SetViewerPara();
         mViewer->SetMap(mMap);
    }
           
    mFrontend->SetViewer(mViewer);
    
    int nlevels; float similarThe1, similarThe2;
    fs_read["ORBextractor.nLevels"] >> nlevels;
    fs_read["LCD.similarityScoreThreshold.high"] >> similarThe1;
    fs_read["LCD.similarityScoreThreshold.low"] >> similarThe2;
    mLoopClosing = std::make_shared<LoopClosing>(nlevels, similarThe1, similarThe2);
    mLoopClosing->setORBExtractor(mORBExtractor);
    mLoopClosing->setMap(mMap);
    mLoopClosing->setBackend(mBackend);

    mBackend->SetLoopClosing(mLoopClosing);

}

bool System::Run(const cv::Mat &ImgLeft, const cv::Mat &ImgRight, const double &TimeStamp){
    bool IsSeccess = mFrontend->Running(ImgLeft, ImgRight, TimeStamp);
    return IsSeccess;
}

void System::GetCameraPara(){

    float fx_left, fy_left, cx_left, cy_left, bf;
    assert(fs_read.isOpened());
    fs_read["Camera.left.fx"] >> fx_left;
    fs_read["Camera.left.fy"] >> fy_left;
    fs_read["Camera.left.cx"] >> cx_left;
    fs_read["Camera.left.cy"] >> cy_left;
    fs_read["Camera.bf"] >> bf;
    Vector3d t_left = Vector3d::Zero();
    SE3 pose_left = SE3( SO3(), t_left );

    float fx_right, fy_right, cx_right, cy_right, baseline;
    fs_read["Camera.right.fx"] >> fx_right;
    fs_read["Camera.right.fy"] >> fy_right;
    fs_read["Camera.right.cx"] >> cx_right;
    fs_read["Camera.right.cy"] >> cy_right;
    baseline = bf / fx_right;
    Vector3d t_right = Vector3d(baseline, 0, 0);
    SE3 pose_right = SE3( SO3(), t_right );

    mCameraLeft = std::make_shared<Camera>( fx_left, fy_left, cx_left, cy_left, baseline, pose_left );
    LOG(INFO) << "CamreaLeft_Pose:" << "\n" << mCameraLeft->GetCameraPose().matrix();
    mCameraRight = std::make_shared<Camera>( fx_right, fy_right, cx_right, cy_right, baseline, pose_right );
    LOG(INFO) << "CamreaRight_Pose:" << "\n" << mCameraRight->GetCameraPose().matrix();

}


void System::GetORBExtractorPara(){
    int nInitFeatures,  nNewFeatures, nLevels;
    int iniThFAST, minThFAST;
    float scaleFactor;
    fs_read["ORBextractor.nInitFeatures"] >> nInitFeatures;
    fs_read["ORBextractor.nNewFeatures"] >> nNewFeatures;
    fs_read["ORBextractor.nLevels"] >> nLevels;
    fs_read["ORBextractor.iniThFAST"] >> iniThFAST;
    fs_read["ORBextractor.minThFAST"] >> minThFAST;
    fs_read["ORBextractor.scaleFactor"] >> scaleFactor;

    mORBExtractor = std::make_shared<ORBExtractor>( nNewFeatures,
                    scaleFactor, nLevels, iniThFAST, minThFAST );
    mORBExtractorInit = std::make_shared<ORBExtractor>(nInitFeatures,
                    scaleFactor, nLevels, iniThFAST, minThFAST );
}

void System::SetViewerPara(){
    float ViewpointX, ViewpointY, ViewpointZ, ViewpointF;
    float GraphLineWidth, PointSize, CameraSize, CameraLineWidth, KeyFrameSize,KeyFrameLineWidth;
    fs_read["Viewer.ViewpointX"] >> ViewpointX;
    fs_read["Viewer.ViewpointY"] >> ViewpointY;
    fs_read["Viewer.ViewpointZ"] >> ViewpointZ;
    fs_read["Viewer.ViewpointF"] >> ViewpointF;
    fs_read["Viewer.KeyFrameSize"] >> KeyFrameSize;
    fs_read["Viewer.KeyFrameLineWidth"] >> KeyFrameLineWidth;
    fs_read["Viewer.GraphLineWidth"] >> GraphLineWidth;
    fs_read["Viewer.PointSize"] >> PointSize;
    fs_read["Viewer.CameraSize"] >> CameraSize;
    fs_read["Viewer.CameraLineWidth"] >> CameraLineWidth;
    mViewer = std::make_shared<Viewer>(ViewpointX,ViewpointY,ViewpointZ,ViewpointF,GraphLineWidth,
                        PointSize, CameraSize, CameraLineWidth,  KeyFrameSize, KeyFrameLineWidth);
}

/**
 * @brief 载入kitti数据集
*/
void System::LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps){
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof()){
        string s;
        getline(fTimes,s);
        if(!s.empty()){
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++){
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

    void System::SaveTrajectory(){
        std::string filePath = "trajectory.txt";
        std::ofstream outfile;
        outfile.open(filePath, std::ios_base::out|std::ios_base::trunc);
        outfile << std::fixed;
        auto KFs = mMap->GetAllKeyFrames();
        SE3 Two = KFs.begin()->second->GetPose().inverse();
        auto Riter = mFrontend->mlReferKFs.begin();
        for(auto iter = mFrontend->mRelativeToRefPose.begin(); 
                 iter!=mFrontend->mRelativeToRefPose.end(); iter++, Riter++){
            auto Refkf = *Riter;
            SE3 Trw = Refkf->GetPose() * Two;
            SE3 Tcw = (*iter) * Trw;
            SE3 Twc = Tcw.inverse();
            Vec3d twc = Twc.translation();
            Mat33d Rwc = Twc.rotationMatrix();
            outfile << setprecision(9) << Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
            Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
            Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << endl;
        }
        outfile.close();
    }

    System::~System(){
        fs_read.release();
        std::cout << "fs_read析构!"<<endl;
    }
}