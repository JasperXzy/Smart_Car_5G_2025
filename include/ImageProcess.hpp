/**
 * @file ImageProcess.hpp
 * @brief 图像处理模块
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace smartcar {

/**
 * @brief 图像处理类
 */
class ImageProcess {
public:
    ImageProcess();
    ~ImageProcess() = default;
    
    /**
     * @brief 初始化图像处理器
     * @param width 图像宽度
     * @param height 图像高度
     * @return true=成功, false=失败
     */
    bool init(int width, int height);
    
    /**
     * @brief Sobel 边缘检测
     * @param src 源图像
     * @param dst 输出图像
     * @param ksize Sobel 核大小
     */
    void sobelEdge(const cv::Mat& src, cv::Mat& dst, int ksize = 3);
    
    /**
     * @brief 查找黑线中心
     * @param gray 灰度图像
     * @param search_y 搜索行号
     * @param search_start_x 搜索起始 X
     * @return 黑线中心 X 坐标，-1表示未找到
     */
    int findLineCenter(const cv::Mat& gray, int search_y, int search_start_x);
    
    /**
     * @brief 斑马线检测
     * @param src 源图像
     * @param hsv_lower HSV 下界
     * @param hsv_upper HSV 上界
     * @param roi 感兴趣区域 (x, y, width, height)
     * @return 检测到的白色条纹数量
     */
    int detectZebraCrossing(const cv::Mat& src, 
                           const cv::Scalar& hsv_lower,
                           const cv::Scalar& hsv_upper,
                           const cv::Rect& roi);
    
    /**
     * @brief 开运算
     * @param src 源图像
     * @param dst 输出图像
     * @param kernel_size 核大小
     */
    void morphologyOpen(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5);
    
    /**
     * @brief 闭运算
     * @param src 源图像
     * @param dst 输出图像
     * @param kernel_size 核大小
     */
    void morphologyClose(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5);
    
    /**
     * @brief 高斯模糊
     * @param src 源图像
     * @param dst 输出图像
     * @param kernel_size 核大小
     */
    void gaussianBlur(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5);
    
    /**
     * @brief 绘制检测结果
     * @param img 图像
     * @param center_x 中心点 X 坐标
     * @param search_y 搜索行 Y 坐标
     */
    void drawLineCenter(cv::Mat& img, int center_x, int search_y);
    
    /**
     * @brief 图像去畸变
     * @param src 源图像
     * @param dst 输出图像
     * @param camera_matrix 相机内参矩阵
     * @param dist_coeffs 畸变系数
     */
    void undistort(const cv::Mat& src, cv::Mat& dst,
                   const cv::Mat& camera_matrix,
                   const cv::Mat& dist_coeffs);
    
private:
    int width_;
    int height_;
    bool initialized_;
    
    // 缓存用的矩阵
    cv::Mat temp_gray_;
    cv::Mat temp_binary_;
    cv::Mat temp_edges_;
};

} // namespace smartcar
