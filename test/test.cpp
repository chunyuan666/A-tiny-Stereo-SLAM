/* 实现线性三角化方法(Linear triangulation methods), 给定匹配点
 * 以及相机投影矩阵(至少2对），计算对应的三维点坐标。给定相机内外参矩阵时，
 * 图像上每个点实际上对应三维中一条射线，理想情况下，利用两条射线相交便可以
 * 得到三维点的坐标。但是实际中，由于计算或者检测误差，无法保证两条射线的
 * 相交性，因此需要建立新的数学模型（如最小二乘）进行求解。
 *
 * 考虑两个视角的情况，假设空间中的三维点P的齐次坐标为X=[x,y,z,1]',对应地在
 * 两个视角的投影点分别为p1和p2，它们的图像坐标为
 *          x1=[x1, y1, 1]', x2=[x2, y2, 1]'.
 *
 * 两幅图像对应的相机投影矩阵为P1, P2 (P1,P2维度是3x4),理想情况下
 *             x1=P1X, x2=P2X
 *
 * 考虑第一个等式，在其两侧分别叉乘x1,可以得到
 *             x1 x (P1X) = 0
 *
 * 将P1X表示成[P11X, P21X, P31X]',其中P11，P21，P31分别是投影矩阵P1的第
 * 1～3行，我们可以得到
 *
 *          x1(P13X) - P11X     = 0
 *          y1(P13X) - P12X     = 0
 *          x1(P12X) - y1(P11X) = 0
 * 其中第三个方程可以由前两个通过线性变换得到，因此我们只考虑全两个方程。每一个
 * 视角可以提供两个约束，联合第二个视角的约束，我们可以得到
 *
 *                   AX = 0,
 * 其中
 *           [x1P13 - P11]
 *       A = [y1P13 - P12]
 *           [x2P23 - P21]
 *           [y2P23 - P22]
 *
 * 当视角个数多于2个的时候，可以采用最小二乘的方式进行求解，理论上，在不存在外点的
 * 情况下，视角越多估计的三维点坐标越准确。当存在外点(错误的匹配点）时，则通常采用
 * RANSAC的鲁棒估计方法进行求解。
 */

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


typedef Eigen::Matrix<double, 3, 4> Mat34d;
typedef Eigen::Matrix<double, 3, 3> Mat33d;
typedef Eigen::Vector3d Vector3d;

using namespace std;
using namespace cv;

void Triangulation(const Mat34d &P1, const Mat34d &P2,
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
        cv::Mat V = vt.row(3).t();
        //为了符合其次坐标的形式，使最后一维为1
        V = V.rowRange(0,3)/V.at<float>(3);
        cv::cv2eigen(V, X3D);
    }

void selfTrian(const Mat &R, const Mat &t, const Mat &K,
                                 const cv::KeyPoint &Kp1, const cv::KeyPoint &Kp2, 
                                 Point3d &points)
{
    Mat33d K_;
    K_ << K.at<double>(0,0),K.at<double>(0,1),K.at<double>(0,2),
        K.at<double>(1,0),K.at<double>(1,1),K.at<double>(1,2),
        K.at<double>(2,0),K.at<double>(2,1),K.at<double>(2,2);
    Mat34d T1;
    T1 << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0;
    T1 = K_ * T1;
    Mat34d T2;
    T2 << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0);
    T2 = K_ * T2;
    Vector3d X3D;
    // cout << "T2: \n" << T2.matrix() << endl;
    Triangulation(T1, T2, Kp1, Kp2, X3D);
    points.x = X3D[0];
    points.y = X3D[1];
    points.z = X3D[2];
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

//相机坐标系转UV系
Point2f cam2pixel(const Point3d &p, const Mat &K)
{
    float d = p.z;
    return Point2f( K.at<double>(0, 0) * p.x / d + K.at<double>(0, 2),
                    K.at<double>(1, 1) * p.x / d + K.at<double>(1, 2));
}

void Triangulation2(const Mat &R, const Mat &t, const Mat &K,
                                 const cv::KeyPoint &Kp1, const cv::KeyPoint &Kp2, 
                                 Point3d &points){
    // Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    // cout << "T2:\n" << T2<<endl;
    vector<Point2f> pts_1, pts_2;
    pts_1.push_back(pixel2cam(Kp1.pt, K));
    pts_2.push_back(pixel2cam(Kp2.pt, K));
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    Mat x = pts_4d.col(0);
    x /= x.at<float>(3, 0); // 归一化
    Point3d p(
    x.at<float>(0, 0),
    x.at<float>(1, 0),
    x.at<float>(2, 0)
    );
    points = p;
}

int main(){
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat R = (Mat_<double>(3, 3) << 
            0.9985961798781877, -0.05169917220143662, 0.01152671359827862,
            0.05139607508976053, 0.9983603445075083, 0.02520051547522452,
            -0.01281065954813537, -0.02457271064688494, 0.9996159607036126);
    Mat t = (Mat_<double>(3,1) << -0.8220841067933339,
                                 -0.0326974270640541,
                                 0.5684264241053518);
    KeyPoint kp1(245, 211, 0);
    KeyPoint kp2(231, 219, 0);

    Point3d points, points2;

    Triangulation2(R, t, K, kp1, kp2, points);

    cout << "opencv:" << points <<endl;
    // Mat pt2_trans = R * (Mat_<double>(3, 1) << points.x, points.y, points.z) + t;
    // Point3d pc(pt2_trans.at<double>(0, 0),pt2_trans.at<double>(1, 0),pt2_trans.at<double>(2, 0) );
    // Point2f uv = cam2pixel(pc, K);
    // cout << "重投影:" << uv <<endl;
    selfTrian(R, t, K, kp1, kp2, points2);
    cout << "self:" << points2 <<endl;

    return 0;
}

/*
int main(int argc, char** argv)
{
	Vec2f p1;
	p1[0] = 0.289986; p1[1] = -0.0355493;

	Vec2f p2;
	p2[0] = 0.316154; p2[1] = 0.0898488;

    cv::KeyPoint kp1(p1[0],p1[1],7);
    cv::KeyPoint kp2(p2[0],p2[1],7);


	Mat P1(3, 4, CV_64FC1);
	Mat P2(3, 4, CV_64FC1);
    Mat34d P1_eigen, P2_eigen;
    

	P1.at<double>(0, 0) = 0.919653;    P1.at<double>(0, 1) = -0.000621866; P1.at<double>(0, 2) = -0.00124006; P1.at<double>(0, 3) = 0.00255933;
	P1.at<double>(1, 0) = 0.000609954; P1.at<double>(1, 1) = 0.919607; P1.at<double>(1, 2) = -0.00957316; P1.at<double>(1, 3) = 0.0540753;
	P1.at<double>(2, 0) = 0.00135482;  P1.at<double>(2, 1) = 0.0104087; P1.at<double>(2, 2) = 0.999949;    P1.at<double>(2, 3) = -0.127624;

	P2.at<double>(0, 0) = 0.920039;    P2.at<double>(0, 1) = -0.0117214;  P2.at<double>(0, 2) = 0.0144298;   P2.at<double>(0, 3) = 0.0749395;
	P2.at<double>(1, 0) = 0.0118301;   P2.at<double>(1, 1) = 0.920129;  P2.at<double>(1, 2) = -0.00678373; P2.at<double>(1, 3) = 0.862711;
	P2.at<double>(2, 0) = -0.0155846;  P2.at<double>(2, 1) = 0.00757181; P2.at<double>(2, 2) = 0.999854;   P2.at<double>(2, 3) = -0.0887441;

    cv::cv2eigen(P1, P1_eigen);
    cv::cv2eigen(P2, P2_eigen);

    // cout << "P1_eigen:\n" << P1_eigen<<endl;
    // cout << "P2_eigen:\n" << P2_eigen<<endl;

	//构造A矩阵
	Mat A(4, 4, CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		A.at<double>(0, i) = p1[0] * P1.at<double>(2, i) - P1.at<double>(0, i);
		A.at<double>(1, i) = p1[1] * P1.at<double>(2, i) - P1.at<double>(1, i);
		A.at<double>(2, i) = p2[0] * P2.at<double>(2, i) - P2.at<double>(0, i);
		A.at<double>(3, i) = p2[1] * P2.at<double>(2, i) - P2.at<double>(1, i);
	}

    cout << "opencv A:\n" << A <<endl;

	Mat U, S, V;
	SVD svd;
	svd.compute(A, S, U, V, 4);
    
	
	Vec3f X;
	X[0] = V.at<double>(0, 3) / V.at<double>(3, 3);
	X[1] = V.at<double>(1, 3) / V.at<double>(3, 3);
	X[2] = V.at<double>(2, 3) / V.at<double>(3, 3);
    cout << "V_opencv:\n" << V <<endl;

    Vector3d X3D;
    Triangulation(P1_eigen, P2_eigen, kp1, kp2, X3D);
    cout << " opencv:trianglede point is :" << X[0] << " " << X[1] << " " << X[2] << endl;
	cout << " myself:trianglede point is :" << X3D[0] << " " << X3D[1] << " " << X3D[2] << endl;

	return 0;
}
*/