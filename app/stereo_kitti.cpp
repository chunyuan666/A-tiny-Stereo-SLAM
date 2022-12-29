#include "System.h"
#include <iostream>
#include "Common.h"
#include <chrono>
#include <X11/Xlib.h>

using namespace std;
using namespace myslam;

int main(int argc, char **argv){

//    if(argc != 3){
//        LOG(INFO) << "argc: " << argc;
//        for(int i = 0; i < argc; i++){
//            LOG(INFO) << "argv[" << i << "]: " << argv[i];
//        }
//        std::cerr << endl << "Usage:  ./bin/run_kitti_stereo   path_to_config   path_to_sequence" << std::endl;
//        return 1;
//    }
    XInitThreads();
    LOG(INFO) << "argc: " << argc;
    for(int i = 0; i < argc; i++){
        LOG(INFO) << "argv[" << i << "]: " << argv[i];
    }
    std::string strConfigPath(argv[1]);
    std::string strSequencePath(argv[2]);
    LOG(INFO) << "configFile:" << strConfigPath;
    // load sequence frames
    std::vector<std::string> vstrImageLeft, vstrImageRight;
    std::vector<double> vTimestamps;

    System::Ptr slam(new System(strConfigPath));

    slam->LoadImages(strSequencePath, vstrImageLeft, vstrImageRight, vTimestamps);
    const unsigned long nImages = vstrImageLeft.size();

   LOG(INFO) << "nImages: " << nImages << endl;

   for(int ni = 0; ni < nImages; ni++){
    //    if(ni < 2600) continue;
       cv::Mat ImgLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_GRAYSCALE);
       cv::Mat ImgRight = cv::imread(vstrImageRight[ni], cv::IMREAD_GRAYSCALE);
       double TimeStamp = vTimestamps[ni];

       if(ImgLeft.empty()){
            std::cerr << std::endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << std::endl;
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        //　开始运行
        bool isGood = slam->Run(ImgLeft, ImgRight, TimeStamp);
        auto t2 = std::chrono::steady_clock::now();
        double time_track = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1).count();
        double T = 0.0;
        T = vTimestamps[ni+1]-TimeStamp;
        if(time_track < T){
            // usleep((T-time_track)*1e6);
        }
        
        if(!isGood)
            break; 
   }

    slam->SaveTrajectory();
    LOG(INFO) << "Trajectory has been Save: ./trajectory.txt";

    slam->Stop();
    
    std::cout << "System stop." << std::endl;

    

    return 0;

}