#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "Common.h"

namespace myslam{

class Camera{

public:
    /**
    * @brief 相机构造函数
    * @details
    */
    Camera(double fx, double fy, double cx, double cy, double baseline, const SE3 &pose):
            mfx(fx), mfy(fy), mcx(cx), mcy(cy), mbaseline(baseline), mPose(pose) {
        mK << mfx, 0,  mcx,
              0,   mfy, mcy,
              0,   0,   1;
            }
    /**
     * ＠brief 获取相机位姿
     * */
    SE3 GetCameraPose(){
        return mPose;
    }

    /**
     * @brief 将相机坐标系投影到像素坐标系
     * */
    cv::Point2f Camera2Pixel(const Vector3d &p_cam) const{
        float u = mfx * p_cam[0] / p_cam[2] + mcx;
        float v = mfy * p_cam[1] / p_cam[2] + mcy;
        // LOG(INFO) << "p_x: \n" << Vec2d(u,v).matrix();
        return {u, v};
    }

    /**
     * @brief 投影函数，将世界坐标投影到像素平面坐标
     * @details 参数T_cw的初始值为单位SE3 = [R|t] = [I | 0]
     * */
    cv::Point2f World2Pixel(const Vector3d &p_world, const SE3 &T_cw = SE3_Identity){
        Vector3d p_cam = mPose.inverse() * (T_cw * p_world);
        // LOG(INFO) << "mpose: " << mPose.matrix3x4();
        // LOG(INFO) << "T_CW: \n" << T_cw.matrix() << "\n" << "p_w: \n" << p_world.matrix();
        return Camera2Pixel(p_cam);
    }

    cv::Point3d pixel2cam(const cv::Point2d &p, const double &depth) {
    return cv::Point3d
        (
            (p.x - mcx) / mfx * depth,
            (p.y - mcy) / mfy * depth,
            depth
        );
}


    /**
     * ＠brief 获得参数K
     * */
    Mat33d GetK(){
        return mK;
    }

public:
    typedef std::shared_ptr<Camera> Ptr;

public:
    float mfx, mfy, mcx, mcy, mbaseline;
    Mat33d mK;
    SE3 mPose;
    SE3 mPose_INV;
};


}

#endif