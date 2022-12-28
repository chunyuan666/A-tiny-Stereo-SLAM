#ifndef MYSLAM_ORBEXTRACTOE_H
#define MYSLAM_ORBEXTRACTOE_H

#include "Common.h"

namespace myslam {

class ORBExtractor{
public:
    /**
     * @brief 构造函数
     * */
    ORBExtractor(int nFeatures, float scaleFactor, int nLevels, int iniThFAST, int minThFAST);

    /**
     * @brief 提取特征点
    */
    void Detect(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints);

    void ComputePyramid(cv::Mat image);

    void CalcuKeyPoints(cv::InputArray img, std::vector<cv::KeyPoint> &kps, std::vector<cv::KeyPoint> &outkps);

    void CalcuDescriptors(cv::Mat img, std::vector<cv::KeyPoint> &kps, cv::OutputArray descriptors);
public:
    typedef std::shared_ptr<ORBExtractor> Ptr;

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<cv::Mat> mvMaskPyramid;

private:

    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
    
    int mnFeatures;  // 初始化时最小提取特征的个数　300,创造新的一帧需要提取的个数 200
    float mscaleFactor; // 缩放系数 1.2
    int mnLevels;       // 金字塔层数 8
    int miniThFAST;     // 初始提取FAST角点的阈值　20
    int mminThFAST;     // 提取失败时最小的提取阈值 7

    std::vector<float> mvScaleFactor;       // 金字塔每层的缩放系数
    std::vector<float> mvInvScaleFactor;    // 金字塔每层的缩放系数的倒数
    std::vector<float> mvLevelSigma2;       // 每层金字塔的∑^2, 缩放系数的平方
    std::vector<float> mvInvLevelSigma2;    // 每层金字塔的∑^2的倒数

    std::vector<int> umax;

    std::vector<int> mnFeaturesPerLevel; //每层金字塔分配的特征点数目

    std::vector<cv::Point> pattern;

        
};


class ExtractorNode
{
    public:
    /** @brief 构造函数 */
    ExtractorNode():bNoMore(false){}

    /**
     * @brief 在八叉树分配特征点的过程中，实现一个节点分裂为4个节点的操作
     * 
     * @param[out] n1   分裂的节点1
     * @param[out] n2   分裂的节点2
     * @param[out] n3   分裂的节点3
     * @param[out] n4   分裂的节点4
     */
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    ///保存有当前节点的特征点
    std::vector<cv::KeyPoint> vKeys;
    ///当前节点所对应的图像坐标边界
    cv::Point2i UL, UR, BL, BR;
    //存储提取器节点的列表（其实就是双向链表）的一个迭代器,可以参考[http://www.runoob.com/cplusplus/cpp-overloading.html]
    //这个迭代器提供了访问总节点列表的方式，需要结合cpp文件进行分析
    std::list<ExtractorNode>::iterator lit;
    
    ///如果节点中只有一个特征点的话，说明这个节点不能够再进行分裂了，这个标志置位
    //这个节点中如果没有特征点的话，这个节点就直接被删除了
    bool bNoMore;
};
}

#endif