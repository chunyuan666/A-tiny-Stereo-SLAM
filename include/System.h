#ifndef __SYSTEM_H
#define __SYSTEM_H

#include "Common.h"
#include "Frontend.h"
#include "Backend.h"
#include "Viewer.h"
#include "Camera.h"
#include "string"
#include "ORBExtractor.h"
#include "Map.h"
#include "LoopClosing.h"

using namespace std;

namespace myslam{


class System{

public:
    /**
     * @brief 构造函数
     * @details 加载相机参数、其他参数，参数保存在yaml文件中,
     *          创造前端、后端、回环检测、地图、可视化组件
     * @param 
    */
    System(const std::string config_file);

    void Stop(){
        if(mBackend)
            mBackend->Stop();
        if(mViewer)
            mViewer->Close();
        if(mLoopClosing)
            mLoopClosing->Stop();
        mBackend.reset();
        mFrontend.reset();
        mViewer.reset();
        mCameraLeft.reset();
        mCameraRight.reset();
        mORBExtractor.reset();
        mORBExtractorInit.reset();
        mMap.reset();
        mLoopClosing.reset();
        
    }

    /**
     * @brief 获得相机参数
    */
    void GetCameraPara();

    /**
     * @brief 获取ORB特征提取参数
    */
    void GetORBExtractorPara();

    /**
     * @brief 载入kitti数据集
    */
    static void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

    /**
     * @brief 运行入口
     * 
     * @param[cv::Mat] ImgLeft　左目图片
    */
    bool Run(const cv::Mat &ImgLeft, const cv::Mat &ImgRight, const double &TimeStamp);

    void SetViewerPara();

    /**
     * @brief 析构函数
     * @details 将读取配置文件析构
    */
   ~System();

public:
    typedef std::shared_ptr<System> Ptr;

    Frontend::Ptr mFrontend;
    Backend::Ptr mBackend;
    Viewer::Ptr mViewer;
    Camera::Ptr mCameraLeft, mCameraRight;
    ORBExtractor::Ptr mORBExtractor, mORBExtractorInit;
    Map::Ptr mMap;
    LoopClosing::Ptr mLoopClosing;
    


private:
    cv::FileStorage fs_read;

};










}

#endif //__SYSTEM_H