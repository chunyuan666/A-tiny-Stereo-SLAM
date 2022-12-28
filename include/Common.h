#ifndef MYSLAM_COMMON_H
#define MYSLAM_COMMON_H

#include <iostream>

#include <atomic>
#include <mutex>

#include <memory>
#include <string>
#include <list>
#include <thread>
#include <sstream>
#include <map>

// glog
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pangolin/pangolin.h>

#include <g2o/types/slam3d/types_slam3d.h>


namespace myslam{
    typedef Sophus::SE3d SE3;
    typedef Sophus::SO3d SO3;


    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix<double, 2, 1> Vec2d;
    typedef Eigen::Matrix<double, 3, 1> Vec3d;
    typedef Eigen::Matrix<double, 4, 1> Vec4d;
    typedef Eigen::Matrix<double, 6, 1> Vec6d;
    typedef Eigen::Matrix<double, 3, 3> Mat33d;
    typedef Eigen::Matrix<double, 2, 2> Mat22d;
    typedef Eigen::Matrix<double, 3, 4> Mat34d;
    typedef Eigen::Matrix<double, 6, 6> Mat66d;

    #define SE3_Identity SE3(SO3(), Vector3d::Zero())

    inline Vec2d cvPoint2Vec2(const cv::Point2f& point){
        return Vec2d(point.x, point.y);
    }

    inline cv::Mat SE3toCvMat(const SE3 &m){
        Eigen::Matrix<double,4,4> eigMat = m.matrix();
        cv::Mat cvMat(4,4,CV_32F);
        for(int i=0;i<4;i++)
            for(int j=0; j<4; j++)
                cvMat.at<float>(i,j)=eigMat(i,j);
        return cvMat.clone();
    }

    inline std::string IntToString(int a)
    {
        std::ostringstream temp;
        temp << a;
        return temp.str();
    }


}

#endif